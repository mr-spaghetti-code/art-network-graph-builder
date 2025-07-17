#!/usr/bin/env python3
"""
Visualize NFT network with seed nodes highlighted
"""

import networkx as nx
import matplotlib.pyplot as plt
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def visualize_network_with_seeds(graph_file: str, output_file: str = None, max_nodes: int = 100):
    """
    Visualize network graph with seed nodes highlighted
    
    Args:
        graph_file: Path to GEXF or GraphML file
        output_file: Path to save visualization (optional)
        max_nodes: Maximum nodes to visualize (for performance)
    """
    logger.info(f"Loading graph from {graph_file}")
    
    # Load graph based on file extension
    if graph_file.endswith('.gexf'):
        G = nx.read_gexf(graph_file)
    elif graph_file.endswith('.graphml'):
        G = nx.read_graphml(graph_file)
    else:
        raise ValueError("Unsupported file format. Use .gexf or .graphml")
    
    logger.info(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # If graph is too large, sample it
    if G.number_of_nodes() > max_nodes:
        logger.info(f"Graph too large, sampling {max_nodes} nodes...")
        # Find seed nodes first
        seed_nodes = [n for n, attrs in G.nodes(data=True) if attrs.get('seed', False)]
        
        # Sample nodes, prioritizing seed nodes
        sampled_nodes = set(seed_nodes[:min(len(seed_nodes), max_nodes // 2)])
        
        # Add non-seed nodes
        non_seed_nodes = [n for n in G.nodes() if n not in seed_nodes]
        remaining_slots = max_nodes - len(sampled_nodes)
        if remaining_slots > 0 and non_seed_nodes:
            import random
            sampled_nodes.update(random.sample(non_seed_nodes, min(remaining_slots, len(non_seed_nodes))))
        
        G = G.subgraph(sampled_nodes).copy()
        logger.info(f"Sampled graph: {G.number_of_nodes()} nodes")
    
    # Prepare node colors and sizes
    node_colors = []
    node_sizes = []
    seed_count = 0
    
    for node, attrs in G.nodes(data=True):
        if attrs.get('seed', False):
            node_colors.append('red')  # Seed nodes in red
            node_sizes.append(300)     # Larger size for seed nodes
            seed_count += 1
        else:
            node_colors.append('lightblue')  # Regular nodes in light blue
            node_sizes.append(100)           # Smaller size for regular nodes
    
    logger.info(f"Found {seed_count} seed nodes in visualization")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(G, k=1/G.number_of_nodes()**0.5, iterations=50)
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3, width=0.5)
    
    # Add labels for seed nodes only (to avoid clutter)
    seed_labels = {n: n[:8] + "..." for n, attrs in G.nodes(data=True) if attrs.get('seed', False)}
    nx.draw_networkx_labels(G, pos, seed_labels, font_size=8)
    
    plt.title(f"NFT Network Visualization\nRed nodes = Seed wallets ({seed_count}), Blue nodes = Discovered wallets")
    plt.axis('off')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize NFT network with seed nodes highlighted")
    parser.add_argument("graph_file", help="Path to graph file (GEXF or GraphML)")
    parser.add_argument("-o", "--output", help="Output file for visualization (PNG, PDF, etc.)")
    parser.add_argument("-n", "--max-nodes", type=int, default=100, 
                       help="Maximum nodes to visualize (default: 100)")
    
    args = parser.parse_args()
    
    try:
        visualize_network_with_seeds(args.graph_file, args.output, args.max_nodes)
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 