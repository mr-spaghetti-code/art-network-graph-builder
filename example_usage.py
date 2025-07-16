#!/usr/bin/env python3
"""
Example usage of the NFT Network Builder

This script demonstrates how to use the NFT network builder in both modes:
1. Starting from an NFT contract (analyzes all holders)
2. Starting from wallet address(es) (analyzes specific wallets)
"""

import subprocess
import sys
import os

def run_example(command, description):
    """Run an example command and show what it does"""
    print(f"\n{'='*60}")
    print(f"Example: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")
    
    response = input("\nRun this example? (y/n): ").lower()
    if response == 'y':
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
        except KeyboardInterrupt:
            print("\nInterrupted by user")
    else:
        print("Skipped")

def main():
    print("NFT Network Builder - Example Usage")
    print("===================================")
    
    # Check if API key is set
    if not os.getenv("ALCHEMY_API_KEY"):
        print("\n⚠️  Warning: ALCHEMY_API_KEY not found in environment!")
        print("Please create a .env file with your API key first.")
        return
    
    examples = [
        # Contract mode examples
        {
            "command": ["python", "nft_network_builder.py", "--contract", "0x8a90CAb2b38dba80c64b7734e58Ee1dB38B8992e"],
            "description": "Build network from Doodles NFT contract"
        },
        {
            "command": ["python", "nft_network_builder.py", "0xb47e3cd837ddf8e4c57f05d70ab865de6e193bbb"],
            "description": "Build network from CryptoPunks (backward compatibility - no flag needed)"
        },
        
        # Wallet mode examples
        {
            "command": ["python", "nft_network_builder.py", "--wallet", "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"],
            "description": "Build network starting from Vitalik's wallet"
        },
        {
            "command": ["python", "nft_network_builder.py", "--wallet", 
                       "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
                       "0x220866B1A2219f40e72f5c628B65D54268cA3A9D",
                       "0x0c23fc0ef06716d2f8ba19bc4bed56d045581f2d"],
            "description": "Build network from multiple notable wallets"
        },
        
        # Helper script example
        {
            "command": ["python", "run_nft_network.py", "bayc"],
            "description": "Use helper script with BAYC shortcut"
        }
    ]
    
    print("\nAvailable examples:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['description']}")
    
    print("\n0. Run all examples")
    print("q. Quit")
    
    while True:
        choice = input("\nSelect an example (0-5 or q): ").lower()
        
        if choice == 'q':
            break
        elif choice == '0':
            for example in examples:
                run_example(example["command"], example["description"])
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(examples):
                    example = examples[idx]
                    run_example(example["command"], example["description"])
                else:
                    print("Invalid choice")
            except ValueError:
                print("Invalid choice")

if __name__ == "__main__":
    main() 