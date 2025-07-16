#!/usr/bin/env python3
"""
NFT Network Builder - Build social graphs from NFT collections

Usage:
    python run_nft_network.py [contract_address]
    
Examples:
    # CryptoPunks
    python run_nft_network.py 0xb47e3cd837ddf8e4c57f05d70ab865de6e193bbb
    
    # Bored Ape Yacht Club
    python run_nft_network.py 0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D
    
    # Mutant Ape Yacht Club
    python run_nft_network.py 0x60E4d786628Fea6478F785A6d7e704777c86a7c6
    
    # Azuki
    python run_nft_network.py 0xED5AF388653567Af2F388E6224dC7C4b3241C544
    
    # Doodles
    python run_nft_network.py 0x8a90CAb2b38dba80c64b7734e58Ee1dB38B8992e
    
    # Cool Cats
    python run_nft_network.py 0x1A92f7381B9F03921564a437210bB9396471050C
"""

import subprocess
import sys
import os

# Popular NFT contract addresses
POPULAR_CONTRACTS = {
    "punks": "0xb47e3cd837ddf8e4c57f05d70ab865de6e193bbb",
    "bayc": "0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D", 
    "mayc": "0x60E4d786628Fea6478F785A6d7e704777c86a7c6",
    "azuki": "0xED5AF388653567Af2F388E6224dC7C4b3241C544",
    "doodles": "0x8a90CAb2b38dba80c64b7734e58Ee1dB38B8992e",
    "coolcats": "0x1A92f7381B9F03921564a437210bB9396471050C",
    "clonex": "0x49cF6f5d44E70224e2E23fDcdd2C053F30aDA28B",
    "worldofwomen": "0xe785E82358879F061BC3dcAC6f0444462D4b5330",
    "pudgypenguins": "0xBd3531dA5CF5857e7CfAA92426877b022e612cf8",
    "meebits": "0x7Bd29408f11D2bFC23c34f18275bBf23bB716Bc7"
}

def main():
    # Check if we have the nft_network_builder.py file
    if not os.path.exists("nft_network_builder.py"):
        print("Error: nft_network_builder.py not found!")
        print("Make sure you have the NFT network builder script in the current directory.")
        return
    
    # Get contract address
    if len(sys.argv) > 1:
        contract_input = sys.argv[1].lower()
        
        # Check if it's a nickname
        if contract_input in POPULAR_CONTRACTS:
            contract_address = POPULAR_CONTRACTS[contract_input]
            print(f"Using {contract_input} contract: {contract_address}")
        else:
            # Assume it's a contract address
            contract_address = sys.argv[1]
    else:
        print(__doc__)
        print("\nPopular contract shortcuts:")
        for name, address in POPULAR_CONTRACTS.items():
            print(f"  {name}: {address}")
        return
    
    # Run the network builder
    print(f"\nBuilding NFT network for contract: {contract_address}")
    print("This may take a while depending on the collection size...\n")
    
    try:
        # Use the --contract flag for clarity
        subprocess.run([sys.executable, "nft_network_builder.py", "--contract", contract_address], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running network builder: {e}")
        return
    except KeyboardInterrupt:
        print("\nNetwork building interrupted by user.")
        return

if __name__ == "__main__":
    main() 