#!/usr/bin/env python3
"""Test checkpoint performance: JSON vs Pickle+Gzip"""

import time
import json
import pickle
import gzip
import os
from collections import defaultdict
import tempfile

def create_test_data(num_wallets=10000, num_contracts=5000):
    """Create test data similar to NFT network data"""
    print(f"Creating test data with {num_wallets} wallets and {num_contracts} contracts...")
    
    # Create test data
    visited_wallets = set(f"wallet_{i}" for i in range(num_wallets))
    
    contract_to_owners = {}
    for i in range(num_contracts):
        # Each contract has 10-100 owners
        num_owners = 10 + (i % 90)
        contract_to_owners[f"contract_{i}"] = set(f"wallet_{j}" for j in range(i, min(i + num_owners, num_wallets)))
    
    wallet_to_contracts = defaultdict(set)
    for contract, owners in contract_to_owners.items():
        for owner in owners:
            wallet_to_contracts[owner].add(contract)
    
    edges = defaultdict(set)
    for i in range(num_wallets):
        # Each wallet connected to 5-20 other wallets
        num_connections = 5 + (i % 15)
        for j in range(num_connections):
            target = (i + j + 1) % num_wallets
            edges[f"wallet_{i}"].add(f"wallet_{target}")
    
    data = {
        "visited_wallets": visited_wallets,
        "contract_to_owners": contract_to_owners,
        "wallet_to_contracts": dict(wallet_to_contracts),
        "edges": dict(edges),
        "queue": [(f"wallet_{i}", i % 3, float(i)) for i in range(1000)],
        "stats": {"api_calls": 123456, "cache_hits": 789012}
    }
    
    return data

def test_json_checkpoint(data, filename):
    """Test JSON checkpoint save/load"""
    print("\n=== Testing JSON Format ===")
    
    # Convert sets to lists for JSON
    json_data = {
        "visited_wallets": list(data["visited_wallets"]),
        "contract_to_owners": {k: list(v) for k, v in data["contract_to_owners"].items()},
        "wallet_to_contracts": {k: list(v) for k, v in data["wallet_to_contracts"].items()},
        "edges": {k: list(v) for k, v in data["edges"].items()},
        "queue": data["queue"],
        "stats": data["stats"]
    }
    
    # Save
    start_time = time.time()
    with open(filename, 'w') as f:
        json.dump(json_data, f)
    save_time = time.time() - start_time
    file_size = os.path.getsize(filename) / (1024**2)  # MB
    print(f"Save time: {save_time:.2f} seconds")
    print(f"File size: {file_size:.2f} MB")
    
    # Load
    start_time = time.time()
    with open(filename, 'r') as f:
        loaded_data = json.load(f)
    
    # Convert lists back to sets
    restored_data = {
        "visited_wallets": set(loaded_data["visited_wallets"]),
        "contract_to_owners": {k: set(v) for k, v in loaded_data["contract_to_owners"].items()},
        "wallet_to_contracts": {k: set(v) for k, v in loaded_data["wallet_to_contracts"].items()},
        "edges": {k: set(v) for k, v in loaded_data["edges"].items()},
        "queue": loaded_data["queue"],
        "stats": loaded_data["stats"]
    }
    load_time = time.time() - start_time
    print(f"Load time: {load_time:.2f} seconds")
    print(f"Total time: {save_time + load_time:.2f} seconds")
    
    return save_time, load_time, file_size

def test_pickle_checkpoint(data, filename):
    """Test Pickle+Gzip checkpoint save/load"""
    print("\n=== Testing Pickle+Gzip Format ===")
    
    # Save (no conversion needed!)
    start_time = time.time()
    with gzip.open(filename, 'wb', compresslevel=1) as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    save_time = time.time() - start_time
    file_size = os.path.getsize(filename) / (1024**2)  # MB
    print(f"Save time: {save_time:.2f} seconds")
    print(f"File size: {file_size:.2f} MB")
    
    # Load
    start_time = time.time()
    with gzip.open(filename, 'rb') as f:
        restored_data = pickle.load(f)
    load_time = time.time() - start_time
    print(f"Load time: {load_time:.2f} seconds")
    print(f"Total time: {save_time + load_time:.2f} seconds")
    
    return save_time, load_time, file_size

def main():
    print("NFT Network Checkpoint Performance Test")
    print("=======================================")
    
    # Create test data
    data = create_test_data(num_wallets=50000, num_contracts=10000)
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as json_file:
        json_filename = json_file.name
    with tempfile.NamedTemporaryFile(suffix='.pkl.gz', delete=False) as pickle_file:
        pickle_filename = pickle_file.name
    
    try:
        # Test JSON
        json_save, json_load, json_size = test_json_checkpoint(data, json_filename)
        
        # Test Pickle
        pickle_save, pickle_load, pickle_size = test_pickle_checkpoint(data, pickle_filename)
        
        # Summary
        print("\n=== SUMMARY ===")
        print(f"JSON total time: {json_save + json_load:.2f}s ({json_size:.2f} MB)")
        print(f"Pickle total time: {pickle_save + pickle_load:.2f}s ({pickle_size:.2f} MB)")
        print(f"\nSpeedup: {(json_save + json_load) / (pickle_save + pickle_load):.1f}x faster")
        print(f"Size reduction: {(1 - pickle_size/json_size) * 100:.1f}% smaller")
        
    finally:
        # Cleanup
        if os.path.exists(json_filename):
            os.unlink(json_filename)
        if os.path.exists(pickle_filename):
            os.unlink(pickle_filename)

if __name__ == "__main__":
    main() 