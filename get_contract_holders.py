#!/usr/bin/env python3
"""
Script to fetch and output all holders of an NFT contract
Usage: python get_contract_holders.py <contract_address> [--output <filename>]
"""

import asyncio
import sys
import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from nft_network_builder import NFTGraphBuilder
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


async def get_all_holders(contract_address: str, output_file: str = None):
    """Fetch all holders of an NFT contract"""
    
    # Get API key
    api_key = os.getenv("ALCHEMY_API_KEY")
    if not api_key:
        logger.error("ALCHEMY_API_KEY environment variable not found!")
        logger.error("Please create a .env file with: ALCHEMY_API_KEY=your_actual_key_here")
        logger.error("Get a free API key at https://www.alchemy.com/")
        return
    
    logger.info(f"Fetching holders for contract: {contract_address}")
    logger.info("This may take a while for large collections...")
    
    start_time = time.time()
    
    # Use the NFTGraphBuilder to fetch holders
    async with NFTGraphBuilder(api_key, rate_limit=20) as builder:
        try:
            # Fetch all holders
            holders = await builder.get_all_contract_holders(contract_address)
            
            if not holders:
                logger.warning("No holders found for this contract")
                return
            
            # Sort holders for consistent output
            holders_list = sorted(list(holders))
            
            # Calculate statistics
            elapsed_time = time.time() - start_time
            
            # Print summary
            logger.info(f"\n{'='*60}")
            logger.info(f"Contract: {contract_address}")
            logger.info(f"Total unique holders: {len(holders_list):,}")
            logger.info(f"Time taken: {elapsed_time:.2f} seconds")
            logger.info(f"API calls made: {builder.api_calls}")
            logger.info(f"{'='*60}\n")
            
            # Output to file if requested
            if output_file:
                output_data = {
                    "contract_address": contract_address,
                    "total_holders": len(holders_list),
                    "timestamp": datetime.now().isoformat(),
                    "time_taken_seconds": elapsed_time,
                    "api_calls": builder.api_calls,
                    "holders": holders_list
                }
                
                # Determine file format from extension
                if output_file.endswith('.json'):
                    with open(output_file, 'w') as f:
                        json.dump(output_data, f, indent=2)
                    logger.info(f"Saved holder data to {output_file}")
                    
                elif output_file.endswith('.txt'):
                    with open(output_file, 'w') as f:
                        f.write(f"# Holders for contract {contract_address}\n")
                        f.write(f"# Total: {len(holders_list)}\n")
                        f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
                        for holder in holders_list:
                            f.write(f"{holder}\n")
                    logger.info(f"Saved holder addresses to {output_file}")
                    
                elif output_file.endswith('.csv'):
                    import csv
                    with open(output_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['wallet_address'])
                        for holder in holders_list:
                            writer.writerow([holder])
                    logger.info(f"Saved holder addresses to {output_file}")
                    
                else:
                    # Default to text format
                    with open(output_file, 'w') as f:
                        for holder in holders_list:
                            f.write(f"{holder}\n")
                    logger.info(f"Saved holder addresses to {output_file}")
            
            else:
                # Print first 10 and last 10 holders to console
                logger.info("First 10 holders:")
                for i, holder in enumerate(holders_list[:10]):
                    print(f"{i+1:4d}. {holder}")
                
                if len(holders_list) > 20:
                    print("      ...")
                    logger.info(f"\nLast 10 holders:")
                    start_idx = len(holders_list) - 10
                    for i, holder in enumerate(holders_list[-10:]):
                        print(f"{start_idx + i + 1:4d}. {holder}")
                elif len(holders_list) > 10:
                    logger.info(f"\nRemaining holders:")
                    for i, holder in enumerate(holders_list[10:], 10):
                        print(f"{i+1:4d}. {holder}")
                
                logger.info("\nTip: Use --output <filename> to save all addresses to a file")
                
        except Exception as e:
            logger.error(f"Error fetching holders: {e}")
            return


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python get_contract_holders.py <contract_address> [--output <filename>]")
        print("\nExamples:")
        print("  python get_contract_holders.py 0xb47e3cd837ddf8e4c57f05d70ab865de6e193bbb")
        print("  python get_contract_holders.py 0xb47e3cd837ddf8e4c57f05d70ab865de6e193bbb --output holders.json")
        print("  python get_contract_holders.py 0xb47e3cd837ddf8e4c57f05d70ab865de6e193bbb --output holders.txt")
        print("  python get_contract_holders.py 0xb47e3cd837ddf8e4c57f05d70ab865de6e193bbb --output holders.csv")
        sys.exit(1)
    
    contract_address = sys.argv[1]
    
    # Check for output file
    output_file = None
    if len(sys.argv) > 2 and sys.argv[2] == "--output":
        if len(sys.argv) > 3:
            output_file = sys.argv[3]
        else:
            print("Error: --output requires a filename")
            sys.exit(1)
    
    # Validate contract address format
    if not contract_address.startswith("0x") or len(contract_address) != 42:
        print(f"Warning: '{contract_address}' doesn't look like a valid Ethereum address")
        print("Valid addresses start with '0x' and are 42 characters long")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Run the async function
    asyncio.run(get_all_holders(contract_address, output_file))


if __name__ == "__main__":
    main() 