#!/usr/bin/env python3
"""
Performance comparison script for checkpoint_to_graph optimization
"""

import time
import json
import os
from checkpoint_to_graph import OptimizedCheckpointToGraph
import argparse

def create_test_checkpoint(num_wallets=10000, avg_connections=50, output_file="test_checkpoint.json"):
    """Create a test checkpoint file for benchmarking"""
    print(f"Creating test checkpoint with {num_wallets} wallets...")
    
    import random
    
    # Generate wallet addresses
    wallets = [f"0x{i:040x}" for i in range(num_wallets)]
    
    # Generate edges
    edges = {}
    for wallet in wallets:
        # Random number of connections
        num_connections = random.randint(avg_connections//2, avg_connections*2)
        connections = random.sample(wallets, min(num_connections, len(wallets)-1))
        # Remove self-connections
        connections = [w for w in connections if w != wallet]
        edges[wallet] = connections
    
    # Generate wallet to contracts mapping
    wallet_to_contracts = {}
    contracts = [f"contract_{i}" for i in range(1000)]
    
    for wallet in wallets:
        # Each wallet interacts with random contracts
        num_contracts = random.randint(1, 20)
        wallet_to_contracts[wallet] = random.sample(contracts, num_contracts)
    
    # Create checkpoint data
    checkpoint_data = {
        "edges": edges,
        "wallet_to_contracts": wallet_to_contracts,
        "stats": {
            "total_wallets": num_wallets,
            "total_edges": sum(len(conns) for conns in edges.values()),
            "avg_connections": avg_connections
        }
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(checkpoint_data, f)
    
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Test checkpoint created: {output_file} ({file_size:.1f} MB)")
    
    return output_file

def benchmark_conversion(checkpoint_file, use_parallel=True, analyze=True):
    """Benchmark the conversion process"""
    converter = OptimizedCheckpointToGraph(
        use_parallel=use_parallel,
        chunk_size=10000
    )
    
    results = {}
    
    # Load checkpoint
    start = time.perf_counter()
    checkpoint_data = converter.load_checkpoint(checkpoint_file, stream_large_files=False)
    results['load_time'] = time.perf_counter() - start
    
    # Build graph
    start = time.perf_counter()
    graph = converter.build_graph_from_checkpoint(checkpoint_data)
    results['build_time'] = time.perf_counter() - start
    
    # Analyze graph
    if analyze:
        start = time.perf_counter()
        stats = converter.analyze_graph(graph)
        results['analyze_time'] = time.perf_counter() - start
        results['stats'] = stats
    
    # Save graph (only one format for benchmarking)
    start = time.perf_counter()
    converter.save_graph(graph, "benchmark_output", formats=["gexf"])
    results['save_time'] = time.perf_counter() - start
    
    results['total_time'] = sum(v for k, v in results.items() if k.endswith('_time'))
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Benchmark checkpoint_to_graph performance")
    parser.add_argument("--create-test", action="store_true",
                       help="Create a test checkpoint file")
    parser.add_argument("--num-wallets", type=int, default=10000,
                       help="Number of wallets for test checkpoint")
    parser.add_argument("--avg-connections", type=int, default=50,
                       help="Average connections per wallet")
    parser.add_argument("--checkpoint", default="test_checkpoint.json",
                       help="Checkpoint file to use")
    parser.add_argument("--compare", action="store_true",
                       help="Compare parallel vs sequential processing")
    
    args = parser.parse_args()
    
    # Create test checkpoint if requested
    if args.create_test:
        checkpoint_file = create_test_checkpoint(
            num_wallets=args.num_wallets,
            avg_connections=args.avg_connections,
            output_file=args.checkpoint
        )
    else:
        checkpoint_file = args.checkpoint
    
    if not os.path.exists(checkpoint_file):
        print(f"Checkpoint file not found: {checkpoint_file}")
        print("Use --create-test to create a test checkpoint")
        return
    
    print(f"\nBenchmarking with checkpoint: {checkpoint_file}")
    print("=" * 60)
    
    if args.compare:
        # Compare parallel vs sequential
        print("\n1. Testing with PARALLEL processing:")
        parallel_results = benchmark_conversion(checkpoint_file, use_parallel=True)
        
        print("\n2. Testing with SEQUENTIAL processing:")
        sequential_results = benchmark_conversion(checkpoint_file, use_parallel=False)
        
        # Print comparison
        print("\n" + "=" * 60)
        print("PERFORMANCE COMPARISON")
        print("=" * 60)
        print(f"{'Operation':<20} {'Parallel (s)':<15} {'Sequential (s)':<15} {'Speedup':<10}")
        print("-" * 60)
        
        for key in ['load_time', 'build_time', 'analyze_time', 'save_time', 'total_time']:
            if key in parallel_results and key in sequential_results:
                p_time = parallel_results[key]
                s_time = sequential_results[key]
                speedup = s_time / p_time if p_time > 0 else 0
                print(f"{key.replace('_', ' ').title():<20} {p_time:<15.3f} {s_time:<15.3f} {speedup:<10.2f}x")
        
        # Print graph stats
        if 'stats' in parallel_results:
            print("\n" + "=" * 60)
            print("GRAPH STATISTICS")
            print("=" * 60)
            stats = parallel_results['stats']
            for key, value in stats.items():
                if key != "top_5_nodes":
                    print(f"{key}: {value}")
    else:
        # Single run
        results = benchmark_conversion(checkpoint_file, use_parallel=True)
        
        print("\nPERFORMANCE RESULTS")
        print("=" * 60)
        for key in ['load_time', 'build_time', 'analyze_time', 'save_time', 'total_time']:
            if key in results:
                print(f"{key.replace('_', ' ').title():<20}: {results[key]:.3f} seconds")
        
        if 'stats' in results:
            print("\nGraph Statistics:")
            stats = results['stats']
            for key, value in stats.items():
                if key != "top_5_nodes":
                    print(f"  {key}: {value}")
    
    # Cleanup
    if args.create_test and os.path.exists("benchmark_output.gexf"):
        os.remove("benchmark_output.gexf")
        print("\nCleaned up temporary files")

if __name__ == "__main__":
    main() 