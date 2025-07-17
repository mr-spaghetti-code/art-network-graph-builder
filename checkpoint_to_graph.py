import json
import os
import networkx as nx
from typing import Dict, Set, List, Tuple, Optional, Generator
import argparse
from datetime import datetime
import logging
import time
from functools import wraps
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from collections import defaultdict, deque
import numpy as np
import ijson  # For streaming JSON parsing
import pickle
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Performance monitoring decorator
def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f"{func.__name__} took {end - start:.3f} seconds")
        return result
    return wrapper

class OptimizedCheckpointToGraph:
    def __init__(self, checkpoint_dir: str = "./checkpoints", 
                 use_parallel: bool = True,
                 num_workers: Optional[int] = None,
                 chunk_size: int = 10000,
                 holders_file: Optional[str] = None):
        self.checkpoint_dir = checkpoint_dir
        self.use_parallel = use_parallel
        self.num_workers = num_workers or mp.cpu_count()
        self.chunk_size = chunk_size
        self.holders_file = holders_file
    
    def load_holders(self) -> Set[str]:
        """Load holder addresses from holders file"""
        holders = set()
        
        if self.holders_file and os.path.exists(self.holders_file):
            logger.info(f"Loading holders from {self.holders_file}")
            with open(self.holders_file, 'r') as f:
                for line in f:
                    address = line.strip().lower()  # Normalize to lowercase
                    if address:
                        holders.add(address)
            logger.info(f"Loaded {len(holders)} holder addresses")
        else:
            logger.warning(f"Holders file not found or not specified: {self.holders_file}")
            
        return holders
        
    @timed
    def list_checkpoints(self) -> List[dict]:
        """List all available checkpoint files - optimized version"""
        checkpoint_path = Path(self.checkpoint_dir)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint directory '{self.checkpoint_dir}' does not exist")
            return []
            
        checkpoints = []
        # Use os.scandir for better performance than os.listdir
        with os.scandir(self.checkpoint_dir) as entries:
            for entry in entries:
                if entry.name.startswith("checkpoint_") and entry.name.endswith(".json"):
                    try:
                        # Extract timestamp more efficiently
                        timestamp_str = entry.name[11:-5]  # Remove "checkpoint_" and ".json"
                        timestamp = int(timestamp_str)
                        stat = entry.stat()
                        
                        checkpoints.append({
                            "filename": entry.name,
                            "path": entry.path,
                            "timestamp": timestamp,
                            "datetime": datetime.fromtimestamp(timestamp),
                            "size_mb": stat.st_size / (1024 * 1024)
                        })
                    except (ValueError, OSError):
                        continue
        
        # Sort by timestamp, newest first
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
        return checkpoints
    
    @timed
    def load_checkpoint(self, checkpoint_path: str, stream_large_files: bool = True) -> dict:
        """Load checkpoint data with option for streaming large files"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
        
        try:
            # For files larger than 100MB, consider streaming
            if stream_large_files and file_size > 100:
                logger.info(f"Large file detected ({file_size:.1f} MB), using streaming parser")
                return self._load_checkpoint_streaming(checkpoint_path)
            else:
                with open(checkpoint_path, 'r') as f:
                    data = json.load(f)
            
            logger.info(f"Checkpoint loaded successfully")
            if "stats" in data:
                logger.info(f"Stats from checkpoint:")
                for key, value in data["stats"].items():
                    logger.info(f"  - {key}: {value}")
            
            # Log seed wallets if present
            if "seed_wallets" in data:
                logger.info(f"Found {len(data['seed_wallets'])} seed wallets in checkpoint")
                    
            return data
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def _load_checkpoint_streaming(self, checkpoint_path: str) -> dict:
        """Stream parse large JSON files to reduce memory usage"""
        data = {"edges": {}, "wallet_to_contracts": {}, "stats": {}, "seed_wallets": []}
        
        try:
            with open(checkpoint_path, 'rb') as f:
                parser = ijson.parse(f)
                current_section = None
                current_key = None
                
                for prefix, event, value in parser:
                    if prefix == "edges" and event == "map_key":
                        current_section = "edges"
                        current_key = value
                        data["edges"][current_key] = []
                    elif prefix.startswith("edges.") and event == "string":
                        if current_section == "edges" and current_key:
                            data["edges"][current_key].append(value)
                    elif prefix == "wallet_to_contracts" and event == "map_key":
                        current_section = "wallet_to_contracts"
                        current_key = value
                        data["wallet_to_contracts"][current_key] = []
                    elif prefix.startswith("wallet_to_contracts.") and event == "string":
                        if current_section == "wallet_to_contracts" and current_key:
                            data["wallet_to_contracts"][current_key].append(value)
                    elif prefix.startswith("seed_wallets.item") and event == "string":
                        data["seed_wallets"].append(value)
                    elif prefix.startswith("stats."):
                        key = prefix.split(".")[-1]
                        if event in ["string", "number"]:
                            data["stats"][key] = value
                            
        except ImportError:
            logger.warning("ijson not available, falling back to standard JSON loading")
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
                
        return data
    
    def _process_edge_batch(self, batch_data: Tuple[List[Tuple[str, str]], Dict[str, Set[str]]]) -> List[Tuple[str, str, int]]:
        """Process a batch of edges to calculate weights"""
        edge_pairs, wallet_contracts_dict = batch_data
        edges_with_weights = []
        
        for wallet1, wallet2 in edge_pairs:
            contracts1 = wallet_contracts_dict.get(wallet1, set())
            contracts2 = wallet_contracts_dict.get(wallet2, set())
            shared = len(contracts1 & contracts2) if contracts1 and contracts2 else 0
            weight = shared if shared > 0 else 1
            edges_with_weights.append((wallet1, wallet2, weight))
            
        return edges_with_weights
    
    @timed
    def build_graph_from_checkpoint(self, checkpoint_data: dict, filter_discovered_only: bool = False) -> nx.Graph:
        """Optimized graph building with bulk operations and parallel processing
        
        Args:
            checkpoint_data: The checkpoint data dictionary
            filter_discovered_only: If True, only include edges between wallets that have contract data
        """
        logger.info("Building graph from checkpoint data...")
        
        edges_dict = checkpoint_data.get("edges", {})
        wallet_to_contracts = checkpoint_data.get("wallet_to_contracts", {})
        
        # Load seed wallets from holders file if available, otherwise from checkpoint
        if self.holders_file:
            seed_wallets = self.load_holders()
        else:
            seed_wallets = set(checkpoint_data.get("seed_wallets", []))
            
        if seed_wallets:
            logger.info(f"Found {len(seed_wallets)} seed wallets")
        
        # Pre-convert all contract lists to sets for O(1) lookup
        logger.info("Pre-processing contract data...")
        wallet_contracts_sets = {
            wallet: set(contracts) if isinstance(contracts, list) else contracts
            for wallet, contracts in wallet_to_contracts.items()
        }
        
        # If filtering, create set of discovered wallets
        if filter_discovered_only:
            discovered_wallets = set(wallet_to_contracts.keys())
            logger.info(f"Filtering to only discovered wallets: {len(discovered_wallets)} wallets")
        
        # Collect all unique edges
        logger.info("Collecting edges...")
        all_edges = []
        processed_pairs = set()
        
        for wallet, connections in edges_dict.items():
            # Ensure connections is a list/set
            if isinstance(connections, list):
                conn_iter = connections
            elif isinstance(connections, set):
                conn_iter = connections
            else:
                conn_iter = list(connections)
                
            for connected_wallet in conn_iter:
                # Skip if filtering and either wallet is not discovered
                if filter_discovered_only:
                    if wallet not in discovered_wallets or connected_wallet not in discovered_wallets:
                        continue
                
                # Create canonical edge representation to avoid duplicates
                edge_pair = (min(wallet, connected_wallet), max(wallet, connected_wallet))
                
                if edge_pair not in processed_pairs:
                    processed_pairs.add(edge_pair)
                    all_edges.append(edge_pair)
        
        logger.info(f"Found {len(all_edges)} unique edges")
        
        # Calculate edge weights in parallel if enabled
        if self.use_parallel and len(all_edges) > 1000:
            logger.info(f"Processing edges in parallel with {self.num_workers} workers...")
            edges_with_weights = self._parallel_edge_processing(all_edges, wallet_contracts_sets)
        else:
            logger.info("Processing edges sequentially...")
            edges_with_weights = self._sequential_edge_processing(all_edges, wallet_contracts_sets)
        
        # Create graph with bulk edge addition
        logger.info("Creating graph with bulk edge addition...")
        G = nx.Graph()
        
        # Add all edges at once - much faster than individual additions
        G.add_weighted_edges_from(edges_with_weights)
        
        # Bulk add node attributes
        logger.info("Adding node attributes...")
        node_attrs = {}
        seed_node_count = 0
        
        for wallet in G.nodes():
            attrs = {}
            
            # Add contract count if available
            if wallet in wallet_to_contracts:
                attrs['num_contracts'] = len(wallet_to_contracts[wallet])
            
            # Add seed attributes if this is a seed wallet
            # Normalize wallet address for comparison
            if wallet.lower() in seed_wallets:
                attrs['seed'] = True
                attrs['Label'] = 'Seed'  # Capital L for Gephi
                attrs['label'] = 'Seed'  # Keep lowercase for backward compatibility
                seed_node_count += 1
            else:
                # Label all non-seed nodes as "Other"
                attrs['Label'] = 'Other'  # Capital L for Gephi
                attrs['label'] = 'Other'  # Keep lowercase for backward compatibility
            
            # Always add node attributes since every node now has at least a label
            node_attrs[wallet] = attrs
        
        nx.set_node_attributes(G, node_attrs)
        
        if seed_node_count > 0:
            logger.info(f"Marked {seed_node_count} nodes as seed nodes")
        
        # Count edges without contract data
        skipped_edges = sum(1 for _, _, w in edges_with_weights if w == 1)
        
        logger.info(f"Graph construction complete:")
        logger.info(f"  - Nodes: {G.number_of_nodes():,}")
        logger.info(f"  - Edges: {G.number_of_edges():,}")
        logger.info(f"  - Edges without contract data: {skipped_edges:,}")
        
        return G
    
    def _parallel_edge_processing(self, all_edges: List[Tuple[str, str]], 
                                 wallet_contracts_sets: Dict[str, Set[str]]) -> List[Tuple[str, str, int]]:
        """Process edges in parallel for weight calculation"""
        # Split edges into chunks for parallel processing
        edge_chunks = [all_edges[i:i + self.chunk_size] 
                      for i in range(0, len(all_edges), self.chunk_size)]
        
        # Prepare data for each worker
        batch_data = [(chunk, wallet_contracts_sets) for chunk in edge_chunks]
        
        edges_with_weights = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._process_edge_batch, data) for data in batch_data]
            
            for future in futures:
                edges_with_weights.extend(future.result())
                
        return edges_with_weights
    
    def _sequential_edge_processing(self, all_edges: List[Tuple[str, str]], 
                                   wallet_contracts_sets: Dict[str, Set[str]]) -> List[Tuple[str, str, int]]:
        """Sequential edge processing for smaller datasets"""
        edges_with_weights = []
        
        for i, (wallet1, wallet2) in enumerate(all_edges):
            contracts1 = wallet_contracts_sets.get(wallet1, set())
            contracts2 = wallet_contracts_sets.get(wallet2, set())
            
            if contracts1 and contracts2:
                shared = len(contracts1 & contracts2)
                weight = shared if shared > 0 else 1
            else:
                weight = 1
                
            edges_with_weights.append((wallet1, wallet2, weight))
            
            if (i + 1) % 10000 == 0:
                logger.info(f"Processed {i + 1:,} edges...")
                
        return edges_with_weights
    
    @timed
    def calculate_unique_contracts(self, checkpoint_data: dict, G: Optional[nx.Graph] = None) -> dict:
        """Calculate the number of unique contracts in the network
        
        Args:
            checkpoint_data: The checkpoint data dictionary containing wallet_to_contracts
            G: Optional graph - if provided, only count contracts for wallets in the graph
            
        Returns:
            Dictionary with contract statistics
        """
        logger.info("Calculating unique contracts...")
        
        wallet_to_contracts = checkpoint_data.get("wallet_to_contracts", {})
        
        # If graph is provided, filter to only wallets in the graph
        if G is not None:
            graph_wallets = set(G.nodes())
            wallet_to_contracts = {
                wallet: contracts 
                for wallet, contracts in wallet_to_contracts.items() 
                if wallet in graph_wallets
            }
            logger.info(f"Filtering to {len(wallet_to_contracts)} wallets present in graph")
        
        # Collect all unique contracts
        all_contracts = set()
        total_contract_instances = 0
        wallets_with_contracts = 0
        contract_counts = []
        
        for wallet, contracts in wallet_to_contracts.items():
            if contracts:
                wallets_with_contracts += 1
                contract_list = contracts if isinstance(contracts, list) else list(contracts)
                all_contracts.update(contract_list)
                total_contract_instances += len(contract_list)
                contract_counts.append(len(contract_list))
        
        # Calculate statistics
        stats = {
            "unique_contracts": len(all_contracts),
            "total_contract_instances": total_contract_instances,
            "wallets_with_contracts": wallets_with_contracts,
            "avg_contracts_per_wallet": total_contract_instances / wallets_with_contracts if wallets_with_contracts > 0 else 0,
        }
        
        if contract_counts:
            stats.update({
                "max_contracts_per_wallet": max(contract_counts),
                "min_contracts_per_wallet": min(contract_counts),
                "median_contracts_per_wallet": np.median(contract_counts),
            })
        
        # Find most popular contracts
        if all_contracts:
            contract_popularity = defaultdict(int)
            for wallet, contracts in wallet_to_contracts.items():
                contract_list = contracts if isinstance(contracts, list) else list(contracts)
                for contract in contract_list:
                    contract_popularity[contract] += 1
            
            # Get top 10 most popular contracts
            top_contracts = sorted(contract_popularity.items(), key=lambda x: x[1], reverse=True)[:10]
            stats["top_10_contracts"] = top_contracts
            
            # Average popularity
            popularities = list(contract_popularity.values())
            stats["avg_contract_popularity"] = np.mean(popularities)
            stats["max_contract_popularity"] = max(popularities)
        
        logger.info(f"Found {stats['unique_contracts']:,} unique contracts")
        logger.info(f"Total contract instances: {stats['total_contract_instances']:,}")
        logger.info(f"Wallets with contracts: {stats['wallets_with_contracts']:,}")
        
        return stats
    
    @timed
    def analyze_graph(self, G: nx.Graph) -> dict:
        """Optimized graph analysis using numpy and efficient algorithms"""
        logger.info("Analyzing graph...")
        
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        
        stats = {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "density": nx.density(G),
        }
        
        # Count seed nodes
        seed_nodes = [n for n, attrs in G.nodes(data=True) if attrs.get('seed', False)]
        if seed_nodes:
            stats["num_seed_nodes"] = len(seed_nodes)
            stats["seed_node_percentage"] = (len(seed_nodes) / num_nodes) * 100
        
        # Use numpy for efficient degree statistics
        degrees = np.array([d for _, d in G.degree()])
        
        if len(degrees) > 0:
            stats["avg_degree"] = np.mean(degrees)
            stats["max_degree"] = int(np.max(degrees))
            stats["min_degree"] = int(np.min(degrees))
            stats["median_degree"] = np.median(degrees)
            stats["std_degree"] = np.std(degrees)
            
            # Find top nodes more efficiently
            top_indices = np.argpartition(degrees, -min(5, len(degrees)))[-5:]
            nodes = list(G.nodes())
            stats["top_5_nodes"] = [(nodes[i], int(degrees[i])) 
                                   for i in sorted(top_indices, key=lambda x: degrees[x], reverse=True)]
            
            # Find top seed nodes by degree if any exist
            if seed_nodes:
                seed_degrees = [(node, G.degree(node)) for node in seed_nodes]
                seed_degrees.sort(key=lambda x: x[1], reverse=True)
                stats["top_5_seed_nodes"] = seed_degrees[:5]
        
        # Efficient connected components analysis
        if num_nodes < 100000:  # Only for smaller graphs
            stats["num_connected_components"] = nx.number_connected_components(G)
            
            # Get component sizes efficiently
            component_sizes = [len(c) for c in nx.connected_components(G)]
            if component_sizes:
                stats["largest_component_size"] = max(component_sizes)
                stats["smallest_component_size"] = min(component_sizes)
                stats["avg_component_size"] = np.mean(component_sizes)
        else:
            logger.info("Skipping component analysis for large graph")
            stats["num_connected_components"] = "Skipped (graph too large)"
        
        return stats
    
    @timed
    def save_graph(self, G: nx.Graph, base_filename: str, formats: List[str] = None):
        """Optimized graph saving with parallel writes for multiple formats"""
        if formats is None:
            formats = ["gexf", "graphml"]
        
        def save_format(fmt: str):
            output_file = f"{base_filename}.{fmt}"
            logger.info(f"Saving graph as {output_file}")
            
            try:
                if fmt == "gexf":
                    nx.write_gexf(G, output_file)
                elif fmt == "graphml":
                    nx.write_graphml(G, output_file)
                elif fmt == "gml":
                    nx.write_gml(G, output_file)
                elif fmt == "json":
                    # Use streaming JSON write for large graphs
                    data = nx.node_link_data(G)
                    with open(output_file, 'w') as f:
                        json.dump(data, f)
                elif fmt == "pickle":
                    # Fast binary format
                    with open(output_file, 'wb') as f:
                        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
                elif fmt == "react":
                    # React-friendly JSON format
                    nodes = []
                    for node in G.nodes():
                        # Use number of contracts as val, fallback to degree if not available
                        val = G.nodes[node].get('num_contracts', G.degree(node))
                        node_data = {
                            "id": node,
                            "name": node,
                            "val": val
                        }
                        
                        # Add seed attribute if present
                        if G.nodes[node].get('seed', False):
                            node_data['seed'] = True
                            node_data['label'] = 'Seed'
                            node_data['Label'] = 'Seed'  # Capital L for Gephi compatibility
                        else:
                            node_data['label'] = 'Other'
                            node_data['Label'] = 'Other'  # Capital L for Gephi compatibility
                        
                        nodes.append(node_data)
                    
                    links = []
                    for source, target in G.edges():
                        links.append({
                            "source": source,
                            "target": target
                        })
                    
                    react_data = {
                        "nodes": nodes,
                        "links": links
                    }
                    
                    with open(output_file, 'w') as f:
                        json.dump(react_data, f, indent=2)
                else:
                    logger.warning(f"Unknown format: {fmt}")
                    return
                    
                logger.info(f"Successfully saved as {output_file}")
            except Exception as e:
                logger.error(f"Failed to save as {fmt}: {e}")
        
        # Save formats in parallel if multiple formats requested
        if len(formats) > 1 and self.use_parallel:
            with ThreadPoolExecutor(max_workers=min(len(formats), 4)) as executor:
                executor.map(save_format, formats)
        else:
            for fmt in formats:
                save_format(fmt)

def main():
    parser = argparse.ArgumentParser(description="Convert NFT network checkpoint to graph file")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", 
                       help="Directory containing checkpoint files")
    parser.add_argument("--checkpoint-file", help="Specific checkpoint file to use")
    parser.add_argument("--output", default="nft_network_from_checkpoint",
                       help="Base filename for output (without extension)")
    parser.add_argument("--formats", nargs="+", 
                       default=["gexf", "graphml", "react"],
                       help="Output formats (gexf, graphml, gml, json, pickle, react)")
    parser.add_argument("--analyze", action="store_true",
                       help="Print detailed graph analysis")
    parser.add_argument("--no-parallel", action="store_true",
                       help="Disable parallel processing")
    parser.add_argument("--workers", type=int,
                       help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--chunk-size", type=int, default=10000,
                       help="Chunk size for parallel processing")
    parser.add_argument("--stream-large-files", action="store_true",
                       help="Use streaming parser for large JSON files (requires ijson)")
    parser.add_argument("--filter-discovered-only", action="store_true",
                       help="Only include edges between wallets with contract data (discovered wallets)")
    parser.add_argument("--holders-file", default="holders.txt",
                       help="File containing holder addresses (one per line) to mark as seed nodes")
    
    args = parser.parse_args()
    
    # Create optimized converter
    converter = OptimizedCheckpointToGraph(
        checkpoint_dir=args.checkpoint_dir,
        use_parallel=not args.no_parallel,
        num_workers=args.workers,
        chunk_size=args.chunk_size,
        holders_file=args.holders_file
    )
    
    # Record total time
    total_start = time.perf_counter()
    
    # Choose checkpoint
    if args.checkpoint_file:
        checkpoint_path = args.checkpoint_file
    else:
        # List available checkpoints
        checkpoints = converter.list_checkpoints()
        
        if not checkpoints:
            logger.error("No checkpoints found!")
            return
        
        logger.info(f"\nAvailable checkpoints:")
        for i, cp in enumerate(checkpoints):
            logger.info(f"{i+1}. {cp['filename']} ({cp['size_mb']:.1f} MB) - {cp['datetime']}")
        
        # Use most recent by default
        if len(checkpoints) == 1:
            choice = 0
            logger.info(f"\nUsing the only available checkpoint")
        else:
            try:
                choice_input = input(f"\nSelect checkpoint (1-{len(checkpoints)}) [default: 1]: ").strip()
                choice = int(choice_input) - 1 if choice_input else 0
                if choice < 0 or choice >= len(checkpoints):
                    raise ValueError()
            except:
                logger.info("Using most recent checkpoint")
                choice = 0
        
        checkpoint_path = checkpoints[choice]["path"]
    
    # Load and convert
    try:
        checkpoint_data = converter.load_checkpoint(checkpoint_path, args.stream_large_files)
        graph = converter.build_graph_from_checkpoint(checkpoint_data, filter_discovered_only=args.filter_discovered_only)
        
        # Analyze if requested
        if args.analyze:
            stats = converter.analyze_graph(graph)
            logger.info("\n=== Graph Analysis ===")
            for key, value in stats.items():
                if key in ["top_5_nodes", "top_5_seed_nodes"]:
                    logger.info(f"{key}:")
                    for wallet, degree in value:
                        logger.info(f"  - {wallet[:8]}...: {degree} connections")
                else:
                    logger.info(f"{key}: {value}")
            logger.info("====================\n")
            
            # Also analyze contracts
            contract_stats = converter.calculate_unique_contracts(checkpoint_data, graph)
            logger.info("\n=== Contract Analysis ===")
            for key, value in contract_stats.items():
                if key == "top_10_contracts":
                    logger.info(f"{key}:")
                    for contract, count in value:
                        logger.info(f"  - {contract[:8]}...: held by {count} wallets")
                elif isinstance(value, float):
                    logger.info(f"{key}: {value:.2f}")
                else:
                    logger.info(f"{key}: {value}")
            logger.info("========================\n")
        
        # Save graph
        converter.save_graph(graph, args.output, args.formats)
        
        total_time = time.perf_counter() - total_start
        logger.info(f"\nConversion complete! Total time: {total_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Failed to convert checkpoint: {e}")
        raise

if __name__ == "__main__":
    main() 