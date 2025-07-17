#!/usr/bin/env python3
"""Convert JSON checkpoint to Pickle format with progress monitoring"""

import json
import pickle
import gzip
import os
import time
import sys
from datetime import datetime

def convert_checkpoint(json_path):
    """Convert a JSON checkpoint to pickle format"""
    if not os.path.exists(json_path):
        print(f"Error: File not found: {json_path}")
        return
    
    # Get file info
    file_size = os.path.getsize(json_path)
    print(f"Converting checkpoint: {json_path}")
    print(f"File size: {file_size / (1024**3):.2f} GB")
    
    # Output filename
    pickle_path = json_path.replace('.json', '.pkl.gz')
    
    # Load JSON with progress updates
    print("\nLoading JSON file...")
    start_time = time.time()
    
    try:
        # For very large files, we need to load in chunks
        # But JSON doesn't support streaming, so we just have to wait
        print("This may take several minutes for large files...")
        print("Progress: Reading file...", end='', flush=True)
        
        with open(json_path, 'r') as f:
            # Read the entire file first (faster than json.load for huge files)
            content = f.read()
            print(" Done!")
            
            print("Progress: Parsing JSON...", end='', flush=True)
            data = json.loads(content)
            print(" Done!")
        
        load_time = time.time() - start_time
        print(f"JSON loaded in {load_time:.1f} seconds")
        
        # Convert lists back to sets for the new format
        print("\nConverting data structures...")
        if isinstance(data.get("visited_wallets"), list):
            data["visited_wallets"] = set(data["visited_wallets"])
            print(f"  - Converted {len(data['visited_wallets'])} visited wallets to set")
        
        if "contract_to_owners" in data:
            total_contracts = len(data["contract_to_owners"])
            print(f"  - Converting {total_contracts} contract->owners mappings...", end='', flush=True)
            data["contract_to_owners"] = {k: set(v) for k, v in data["contract_to_owners"].items()}
            print(" Done!")
        
        if "wallet_to_contracts" in data:
            total_wallets = len(data["wallet_to_contracts"])
            print(f"  - Converting {total_wallets} wallet->contracts mappings...", end='', flush=True)
            data["wallet_to_contracts"] = {k: set(v) for k, v in data["wallet_to_contracts"].items()}
            print(" Done!")
        
        if "edges" in data:
            total_edges = len(data["edges"])
            print(f"  - Converting {total_edges} edge sets...", end='', flush=True)
            data["edges"] = {k: set(v) for k, v in data["edges"].items()}
            print(" Done!")
        
        if "seed_wallets" in data and isinstance(data["seed_wallets"], list):
            data["seed_wallets"] = set(data["seed_wallets"])
            print(f"  - Converted {len(data['seed_wallets'])} seed wallets to set")
        
        # Save as pickle with compression
        print(f"\nSaving to pickle format: {pickle_path}")
        start_time = time.time()
        
        with gzip.open(pickle_path, 'wb', compresslevel=1) as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        save_time = time.time() - start_time
        new_size = os.path.getsize(pickle_path)
        
        print(f"Pickle saved in {save_time:.1f} seconds")
        print(f"New file size: {new_size / (1024**2):.1f} MB")
        print(f"Compression ratio: {(1 - new_size/file_size) * 100:.1f}% smaller")
        
        # Test loading the pickle file
        print("\nTesting pickle load speed...")
        start_time = time.time()
        with gzip.open(pickle_path, 'rb') as f:
            test_data = pickle.load(f)
        test_time = time.time() - start_time
        print(f"Pickle loads in {test_time:.1f} seconds!")
        
        print(f"\n✅ Conversion complete!")
        print(f"Total time saved on future loads: ~{load_time - test_time:.0f} seconds")
        
        # Show stats
        print("\nCheckpoint contents:")
        print(f"  - Visited wallets: {len(data.get('visited_wallets', []))}")
        print(f"  - Contracts tracked: {len(data.get('contract_to_owners', {}))}")
        print(f"  - Wallets tracked: {len(data.get('wallet_to_contracts', {}))}")
        print(f"  - Total edges: {sum(len(v) for v in data.get('edges', {}).values())}")
        
        return pickle_path
        
    except json.JSONDecodeError as e:
        print(f"\nError: Failed to parse JSON: {e}")
        print("The file might be corrupted or incomplete.")
        return None
    except MemoryError:
        print(f"\nError: Not enough memory to load the checkpoint!")
        print("Try closing other applications or using a machine with more RAM.")
        return None
    except Exception as e:
        print(f"\nError: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        # Look for JSON checkpoints in the current directory
        import glob
        json_files = []
        for pattern in ["checkpoints*/checkpoint_*.json", "checkpoint_*.json"]:
            json_files.extend(glob.glob(pattern))
        
        if not json_files:
            print("No JSON checkpoint files found!")
            print("\nUsage: python convert_checkpoint.py <checkpoint.json>")
            return
        
        print("Found JSON checkpoints:")
        for i, f in enumerate(json_files):
            size = os.path.getsize(f) / (1024**3)
            print(f"{i+1}. {f} ({size:.2f} GB)")
        
        try:
            choice = int(input("\nWhich checkpoint to convert? (number): ")) - 1
            if 0 <= choice < len(json_files):
                json_path = json_files[choice]
            else:
                print("Invalid choice!")
                return
        except (ValueError, KeyboardInterrupt):
            print("\nCancelled.")
            return
    else:
        json_path = sys.argv[1]
    
    convert_checkpoint(json_path)

if __name__ == "__main__":
    main() 