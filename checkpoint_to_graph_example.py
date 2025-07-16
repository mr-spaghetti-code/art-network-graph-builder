#!/usr/bin/env python3
"""
Example usage of optimized checkpoint_to_graph.py

This script demonstrates different ways to use the optimized checkpoint to graph converter.
"""

from checkpoint_to_graph import OptimizedCheckpointToGraph
import logging
import time

# Set up logging for the example
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_basic_usage():
    """Basic usage: convert most recent checkpoint to graph"""
    logger.info("=== Basic Usage Example ===")
    
    # Create converter with default settings (parallel processing enabled)
    converter = OptimizedCheckpointToGraph("./checkpoints")
    
    # List available checkpoints
    checkpoints = converter.list_checkpoints()
    
    if not checkpoints:
        logger.error("No checkpoints found!")
        return
    
    # Use the most recent checkpoint
    latest_checkpoint = checkpoints[0]
    logger.info(f"Using checkpoint: {latest_checkpoint['filename']}")
    
    # Load and convert
    checkpoint_data = converter.load_checkpoint(latest_checkpoint['path'])
    graph = converter.build_graph_from_checkpoint(checkpoint_data)
    
    # Save in multiple formats
    converter.save_graph(graph, "example_output", formats=["gexf", "graphml"])
    
    logger.info("Conversion complete!")

def example_with_analysis():
    """Example with graph analysis"""
    logger.info("\n=== Analysis Example ===")
    
    converter = OptimizedCheckpointToGraph("./checkpoints")
    checkpoints = converter.list_checkpoints()
    
    if not checkpoints:
        return
    
    # Load checkpoint
    checkpoint_data = converter.load_checkpoint(checkpoints[0]['path'])
    graph = converter.build_graph_from_checkpoint(checkpoint_data)
    
    # Analyze the graph
    stats = converter.analyze_graph(graph)
    
    logger.info("\nGraph Statistics:")
    for key, value in stats.items():
        if key == "top_5_nodes":
            logger.info(f"\n{key}:")
            for wallet, degree in value:
                logger.info(f"  - {wallet}: {degree} connections")
        else:
            logger.info(f"{key}: {value}")

def example_parallel_vs_sequential():
    """Example comparing parallel vs sequential processing"""
    logger.info("\n=== Parallel vs Sequential Comparison ===")
    
    checkpoints = OptimizedCheckpointToGraph("./checkpoints").list_checkpoints()
    if not checkpoints:
        return
    
    checkpoint_path = checkpoints[0]['path']
    
    # Test with parallel processing
    logger.info("\n1. With parallel processing:")
    start = time.perf_counter()
    converter_parallel = OptimizedCheckpointToGraph(use_parallel=True, num_workers=4)
    data = converter_parallel.load_checkpoint(checkpoint_path)
    graph = converter_parallel.build_graph_from_checkpoint(data)
    parallel_time = time.perf_counter() - start
    
    # Test without parallel processing
    logger.info("\n2. Without parallel processing:")
    start = time.perf_counter()
    converter_sequential = OptimizedCheckpointToGraph(use_parallel=False)
    data = converter_sequential.load_checkpoint(checkpoint_path)
    graph = converter_sequential.build_graph_from_checkpoint(data)
    sequential_time = time.perf_counter() - start
    
    logger.info(f"\nPerformance comparison:")
    logger.info(f"Parallel processing time: {parallel_time:.2f} seconds")
    logger.info(f"Sequential processing time: {sequential_time:.2f} seconds")
    logger.info(f"Speedup: {sequential_time/parallel_time:.2f}x")

def example_large_file_streaming():
    """Example with streaming for large files"""
    logger.info("\n=== Large File Streaming Example ===")
    
    converter = OptimizedCheckpointToGraph()
    checkpoints = converter.list_checkpoints()
    
    if not checkpoints:
        return
    
    # Find largest checkpoint
    largest = max(checkpoints, key=lambda x: x['size_mb'])
    logger.info(f"Processing largest checkpoint: {largest['filename']} ({largest['size_mb']:.1f} MB)")
    
    # Load with streaming (if file > 100MB)
    checkpoint_data = converter.load_checkpoint(largest['path'], stream_large_files=True)
    graph = converter.build_graph_from_checkpoint(checkpoint_data)
    
    # Save in efficient format
    converter.save_graph(graph, "large_graph_output", formats=["pickle"])  # Binary format is faster

def example_custom_settings():
    """Example with custom performance settings"""
    logger.info("\n=== Custom Settings Example ===")
    
    # Create converter with custom settings
    converter = OptimizedCheckpointToGraph(
        checkpoint_dir="./checkpoints",
        use_parallel=True,
        num_workers=8,  # Use 8 workers
        chunk_size=5000  # Smaller chunks for better load balancing
    )
    
    checkpoints = converter.list_checkpoints()
    if not checkpoints:
        return
    
    # Process checkpoint
    checkpoint_data = converter.load_checkpoint(checkpoints[0]['path'])
    graph = converter.build_graph_from_checkpoint(checkpoint_data)
    
    # Quick analysis
    logger.info(f"Graph has {graph.number_of_nodes():,} nodes and {graph.number_of_edges():,} edges")
    
    # Save in multiple formats in parallel
    converter.save_graph(graph, "custom_output", formats=["gexf", "graphml", "json", "pickle"])

if __name__ == "__main__":
    # Run examples
    example_basic_usage()
    example_with_analysis()
    # example_parallel_vs_sequential()  # Uncomment to test performance comparison
    # example_large_file_streaming()     # Uncomment for large file handling
    # example_custom_settings()          # Uncomment for custom settings demo 