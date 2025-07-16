#!/usr/bin/env python3
"""
Network visualization tool for NFT collector networks

Supports multiple visualization methods:
- Interactive HTML visualization (pyvis)
- Static matplotlib plots
- Plotly interactive plots
- Community detection and visualization
- Node importance highlighting
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import argparse
import logging
from pathlib import Path
import pickle
import json
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NetworkVisualizer:
    def __init__(self, graph_file: str):
        self.graph_file = graph_file
        self.graph = None
        self.load_graph()
        
    def load_graph(self):
        """Load graph from file based on extension"""
        logger.info(f"Loading graph from {self.graph_file}")
        
        ext = Path(self.graph_file).suffix.lower()
        
        try:
            if ext == '.gexf':
                self.graph = nx.read_gexf(self.graph_file)
            elif ext == '.graphml':
                self.graph = nx.read_graphml(self.graph_file)
            elif ext == '.gml':
                self.graph = nx.read_gml(self.graph_file)
            elif ext == '.pickle' or ext == '.pkl':
                with open(self.graph_file, 'rb') as f:
                    self.graph = pickle.load(f)
            elif ext == '.json':
                with open(self.graph_file, 'r') as f:
                    data = json.load(f)
                    self.graph = nx.node_link_graph(data)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
                
            logger.info(f"Graph loaded successfully:")
            logger.info(f"  - Nodes: {self.graph.number_of_nodes():,}")
            logger.info(f"  - Edges: {self.graph.number_of_edges():,}")
            
        except Exception as e:
            logger.error(f"Failed to load graph: {e}")
            raise
            
    def get_graph_stats(self) -> Dict:
        """Get basic statistics about the graph"""
        stats = {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "is_connected": nx.is_connected(self.graph),
        }
        
        if stats["num_nodes"] < 10000:  # Only for smaller graphs
            stats["diameter"] = nx.diameter(self.graph) if stats["is_connected"] else "N/A (disconnected)"
            stats["average_clustering"] = nx.average_clustering(self.graph)
        
        return stats
    
    def visualize_pyvis(self, output_file: str = "network.html", 
                       max_nodes: Optional[int] = None,
                       physics: bool = True):
        """Create interactive HTML visualization using pyvis"""
        try:
            from pyvis.network import Network
        except ImportError:
            logger.error("pyvis not installed. Install with: pip install pyvis")
            return
            
        logger.info("Creating interactive HTML visualization...")
        
        # Create subgraph if needed
        if max_nodes and self.graph.number_of_nodes() > max_nodes:
            logger.info(f"Graph too large, sampling {max_nodes} nodes...")
            nodes = self._get_important_nodes(max_nodes)
            subgraph = self.graph.subgraph(nodes)
        else:
            subgraph = self.graph
            
        # Create pyvis network
        net = Network(
            height="800px", 
            width="100%", 
            bgcolor="#222222", 
            font_color="white",
            notebook=False
        )
        
        # Add nodes and edges
        net.from_nx(subgraph)
        
        # Configure physics
        if physics:
            net.barnes_hut(
                gravity=-80000,
                central_gravity=0.3,
                spring_length=250,
                spring_strength=0.01,
                damping=0.09
            )
        else:
            net.toggle_physics(False)
            
        # Add options
        net.set_options("""
        var options = {
          "edges": {
            "color": {
              "inherit": true
            },
            "smooth": false
          },
          "nodes": {
            "font": {
              "size": 12
            }
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100
          }
        }
        """)
        
        # Save
        net.save_graph(output_file)
        logger.info(f"Interactive visualization saved to {output_file}")
        
    def visualize_matplotlib(self, output_file: str = "network.png",
                           layout: str = "spring",
                           max_nodes: Optional[int] = 1000,
                           node_size_attr: Optional[str] = None,
                           color_communities: bool = False):
        """Create static visualization using matplotlib"""
        logger.info("Creating static matplotlib visualization...")
        
        # Create subgraph if needed
        if max_nodes and self.graph.number_of_nodes() > max_nodes:
            logger.info(f"Graph too large, sampling {max_nodes} nodes...")
            nodes = self._get_important_nodes(max_nodes)
            subgraph = self.graph.subgraph(nodes)
        else:
            subgraph = self.graph
            
        # Create figure
        plt.figure(figsize=(20, 20))
        
        # Calculate layout
        logger.info(f"Calculating {layout} layout...")
        if layout == "spring":
            pos = nx.spring_layout(subgraph, k=1/np.sqrt(subgraph.number_of_nodes()), iterations=50)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(subgraph)
        elif layout == "circular":
            pos = nx.circular_layout(subgraph)
        elif layout == "random":
            pos = nx.random_layout(subgraph)
        else:
            pos = nx.spring_layout(subgraph)
            
        # Node sizes
        if node_size_attr and node_size_attr in next(iter(subgraph.nodes(data=True)))[1]:
            node_sizes = [subgraph.nodes[node].get(node_size_attr, 1) * 10 for node in subgraph.nodes()]
        else:
            degrees = dict(subgraph.degree())
            node_sizes = [degrees[node] * 10 for node in subgraph.nodes()]
            
        # Node colors
        if color_communities:
            communities = self._detect_communities(subgraph)
            node_colors = [communities.get(node, 0) for node in subgraph.nodes()]
        else:
            node_colors = 'lightblue'
            
        # Draw graph
        nx.draw_networkx_nodes(subgraph, pos, 
                              node_size=node_sizes,
                              node_color=node_colors,
                              cmap=plt.cm.viridis,
                              alpha=0.7)
        
        nx.draw_networkx_edges(subgraph, pos,
                              edge_color='gray',
                              alpha=0.2,
                              width=0.5)
        
        # Add labels for high-degree nodes
        high_degree_nodes = sorted(subgraph.degree(), key=lambda x: x[1], reverse=True)[:20]
        labels = {node: node[:8] + "..." for node, _ in high_degree_nodes}
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)
        
        plt.title(f"Network Visualization ({subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges)")
        plt.axis('off')
        plt.tight_layout()
        
        # Save
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"Static visualization saved to {output_file}")
        
    def visualize_plotly(self, output_file: str = "network_plotly.html",
                        max_nodes: Optional[int] = 5000):
        """Create interactive plotly visualization"""
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.error("plotly not installed. Install with: pip install plotly")
            return
            
        logger.info("Creating interactive plotly visualization...")
        
        # Create subgraph if needed
        if max_nodes and self.graph.number_of_nodes() > max_nodes:
            logger.info(f"Graph too large, sampling {max_nodes} nodes...")
            nodes = self._get_important_nodes(max_nodes)
            subgraph = self.graph.subgraph(nodes)
        else:
            subgraph = self.graph
            
        # Calculate layout
        logger.info("Calculating layout...")
        pos = nx.spring_layout(subgraph, k=1/np.sqrt(subgraph.number_of_nodes()), iterations=50)
        
        # Create edge traces
        edge_trace = []
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=0.5, color='#888'),
                hoverinfo='none'
            ))
            
        # Create node trace
        node_x = []
        node_y = []
        for node in subgraph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlOrRd',
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                )
            )
        )
        
        # Color nodes by degree
        node_adjacencies = []
        node_text = []
        for node in subgraph.nodes():
            adjacencies = len(list(subgraph.neighbors(node)))
            node_adjacencies.append(adjacencies)
            node_text.append(f'{node}<br># connections: {adjacencies}')
            
        node_trace.marker.color = node_adjacencies
        node_trace.text = node_text
        
        # Create figure
        fig = go.Figure(data=edge_trace + [node_trace],
                       layout=go.Layout(
                           title=f'Network Graph ({subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges)',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white'
                       ))
        
        # Save
        fig.write_html(output_file)
        logger.info(f"Plotly visualization saved to {output_file}")
        
    def analyze_communities(self, output_prefix: str = "communities"):
        """Detect and visualize communities in the network"""
        logger.info("Detecting communities...")
        
        # Use Louvain method for community detection
        try:
            import community as community_louvain
            communities = community_louvain.best_partition(self.graph)
        except ImportError:
            logger.warning("python-louvain not installed. Using alternative method.")
            # Fallback to connected components
            communities = {}
            for i, component in enumerate(nx.connected_components(self.graph)):
                for node in component:
                    communities[node] = i
                    
        # Count community sizes
        community_sizes = {}
        for node, comm in communities.items():
            community_sizes[comm] = community_sizes.get(comm, 0) + 1
            
        # Log statistics
        logger.info(f"Found {len(community_sizes)} communities")
        largest_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("Top 10 communities by size:")
        for comm_id, size in largest_communities:
            logger.info(f"  Community {comm_id}: {size} nodes")
            
        # Visualize largest communities
        for i, (comm_id, size) in enumerate(largest_communities[:3]):
            if size > 10:  # Only visualize communities with more than 10 nodes
                nodes = [n for n, c in communities.items() if c == comm_id]
                subgraph = self.graph.subgraph(nodes)
                
                plt.figure(figsize=(10, 10))
                pos = nx.spring_layout(subgraph)
                nx.draw(subgraph, pos, node_color='lightblue', 
                       node_size=50, with_labels=False,
                       edge_color='gray', alpha=0.5)
                plt.title(f"Community {comm_id} ({size} nodes)")
                plt.savefig(f"{output_prefix}_community_{i+1}.png", dpi=150, bbox_inches='tight')
                plt.close()
                
        logger.info(f"Community visualizations saved")
        
    def find_key_nodes(self, top_n: int = 20) -> List[Tuple[str, Dict]]:
        """Find the most important nodes based on various centrality measures"""
        logger.info("Finding key nodes...")
        
        # Calculate centrality measures
        degree_cent = nx.degree_centrality(self.graph)
        
        # For smaller graphs, calculate more expensive metrics
        if self.graph.number_of_nodes() < 5000:
            betweenness_cent = nx.betweenness_centrality(self.graph, k=min(100, self.graph.number_of_nodes()))
            closeness_cent = nx.closeness_centrality(self.graph)
            try:
                eigenvector_cent = nx.eigenvector_centrality(self.graph, max_iter=100)
            except:
                eigenvector_cent = {}
        else:
            betweenness_cent = {}
            closeness_cent = {}
            eigenvector_cent = {}
            
        # Combine scores
        key_nodes = []
        for node in self.graph.nodes():
            scores = {
                'degree': degree_cent.get(node, 0),
                'betweenness': betweenness_cent.get(node, 0),
                'closeness': closeness_cent.get(node, 0),
                'eigenvector': eigenvector_cent.get(node, 0),
                'num_contracts': self.graph.nodes[node].get('num_contracts', 0)
            }
            key_nodes.append((node, scores))
            
        # Sort by degree
        key_nodes.sort(key=lambda x: x[1]['degree'], reverse=True)
        
        # Log top nodes
        logger.info(f"Top {top_n} nodes by degree centrality:")
        for i, (node, scores) in enumerate(key_nodes[:top_n]):
            logger.info(f"  {i+1}. {node[:16]}... (degree: {scores['degree']:.4f}, contracts: {scores['num_contracts']})")
            
        return key_nodes[:top_n]
        
    def _get_important_nodes(self, n: int) -> List:
        """Get n most important nodes based on degree and other metrics"""
        # Start with highest degree nodes
        degrees = dict(self.graph.degree())
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        important_nodes = [node for node, _ in sorted_nodes[:n//2]]
        
        # Add random nodes to maintain some network structure
        remaining_nodes = list(set(self.graph.nodes()) - set(important_nodes))
        if remaining_nodes:
            import random
            random_sample = random.sample(remaining_nodes, min(n//2, len(remaining_nodes)))
            important_nodes.extend(random_sample)
            
        return important_nodes[:n]
        
    def _detect_communities(self, graph) -> Dict:
        """Detect communities in the graph"""
        try:
            import community as community_louvain
            return community_louvain.best_partition(graph)
        except ImportError:
            # Simple connected components as fallback
            communities = {}
            for i, component in enumerate(nx.connected_components(graph)):
                for node in component:
                    communities[node] = i
            return communities

def main():
    parser = argparse.ArgumentParser(description="Visualize NFT collector network")
    parser.add_argument("graph_file", help="Path to graph file (gexf, graphml, pickle, etc.)")
    parser.add_argument("--output", default="network_visualization",
                       help="Output file prefix")
    parser.add_argument("--method", choices=["pyvis", "matplotlib", "plotly", "all"],
                       default="pyvis",
                       help="Visualization method")
    parser.add_argument("--max-nodes", type=int,
                       help="Maximum nodes to visualize (for large graphs)")
    parser.add_argument("--layout", choices=["spring", "kamada_kawai", "circular", "random"],
                       default="spring",
                       help="Graph layout algorithm")
    parser.add_argument("--analyze", action="store_true",
                       help="Perform network analysis")
    parser.add_argument("--communities", action="store_true",
                       help="Detect and visualize communities")
    parser.add_argument("--key-nodes", type=int, default=20,
                       help="Number of key nodes to identify")
    
    args = parser.parse_args()
    
    # Create visualizer
    try:
        viz = NetworkVisualizer(args.graph_file)
    except Exception as e:
        logger.error(f"Failed to load graph: {e}")
        return
        
    # Show basic stats
    stats = viz.get_graph_stats()
    logger.info("\n=== Network Statistics ===")
    for key, value in stats.items():
        logger.info(f"{key}: {value}")
    logger.info("========================\n")
    
    # Perform analysis if requested
    if args.analyze:
        key_nodes = viz.find_key_nodes(args.key_nodes)
        
    # Detect communities if requested
    if args.communities:
        viz.analyze_communities(f"{args.output}_communities")
        
    # Create visualizations
    if args.method == "all":
        methods = ["pyvis", "matplotlib", "plotly"]
    else:
        methods = [args.method]
        
    for method in methods:
        try:
            if method == "pyvis":
                viz.visualize_pyvis(
                    f"{args.output}_interactive.html",
                    max_nodes=args.max_nodes or 10000
                )
            elif method == "matplotlib":
                viz.visualize_matplotlib(
                    f"{args.output}_static.png",
                    layout=args.layout,
                    max_nodes=args.max_nodes or 1000,
                    color_communities=args.communities
                )
            elif method == "plotly":
                viz.visualize_plotly(
                    f"{args.output}_plotly.html",
                    max_nodes=args.max_nodes or 5000
                )
        except Exception as e:
            logger.error(f"Failed to create {method} visualization: {e}")
            
    logger.info("\nVisualization complete!")

if __name__ == "__main__":
    main() 