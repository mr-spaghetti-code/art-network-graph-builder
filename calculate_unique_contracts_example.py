#!/usr/bin/env python3
"""
Example script to calculate the number of unique contracts in an NFT network
"""

from checkpoint_to_graph import OptimizedCheckpointToGraph
import argparse
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Calculate unique contracts in NFT network")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", 
                       help="Directory containing checkpoint files")
    parser.add_argument("--checkpoint-file", help="Specific checkpoint file to use")
    parser.add_argument("--filter-discovered-only", action="store_true",
                       help="Only analyze contracts for discovered wallets")
    
    args = parser.parse_args()
    
    # Create converter
    converter = OptimizedCheckpointToGraph(checkpoint_dir=args.checkpoint_dir)
    
    # Choose checkpoint
    if args.checkpoint_file:
        checkpoint_path = args.checkpoint_file
    else:
        # List available checkpoints
        checkpoints = converter.list_checkpoints()
        
        if not checkpoints:
            logger.error("No checkpoints found!")
            return
        
        # Use most recent
        checkpoint_path = checkpoints[0]["path"]
        logger.info(f"Using most recent checkpoint: {checkpoints[0]['filename']}")
    
    # Load checkpoint
    try:
        logger.info("\nLoading checkpoint data...")
        checkpoint_data = converter.load_checkpoint(checkpoint_path)
        
        # Calculate contracts for all wallets in checkpoint
        logger.info("\n=== Contract Analysis (All Wallets) ===")
        all_stats = converter.calculate_unique_contracts(checkpoint_data)
        
        for key, value in all_stats.items():
            if key == "top_10_contracts":
                logger.info(f"\n{key}:")
                for i, (contract, count) in enumerate(value, 1):
                    logger.info(f"  {i}. {contract}: held by {count} wallets")
            elif isinstance(value, float):
                logger.info(f"{key}: {value:.2f}")
            else:
                logger.info(f"{key}: {value:,}")
        
        # If filtering, also show stats for discovered wallets only
        if args.filter_discovered_only:
            logger.info("\n\nBuilding filtered graph...")
            graph = converter.build_graph_from_checkpoint(checkpoint_data, filter_discovered_only=True)
            
            logger.info("\n=== Contract Analysis (Discovered Wallets Only) ===")
            filtered_stats = converter.calculate_unique_contracts(checkpoint_data, graph)
            
            for key, value in filtered_stats.items():
                if key == "top_10_contracts":
                    logger.info(f"\n{key}:")
                    for i, (contract, count) in enumerate(value, 1):
                        logger.info(f"  {i}. {contract}: held by {count} wallets")
                elif isinstance(value, float):
                    logger.info(f"{key}: {value:.2f}")
                else:
                    logger.info(f"{key}: {value:,}")
        
    except Exception as e:
        logger.error(f"Failed to analyze contracts: {e}")
        raise

if __name__ == "__main__":
    main() 