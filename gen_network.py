import asyncio
import aiohttp
from collections import defaultdict, deque
import time
import networkx as nx
from typing import Set, Dict, List, Tuple, Optional
import pickle
import json
import os
from datetime import datetime
from dataclasses import dataclass
import heapq
import mmh3
from bitarray import bitarray
import aiofiles
import ujson  # faster json
from functools import lru_cache
import logging
import glob
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class BloomFilter:
    """space-efficient probabilistic set membership testing"""
    def __init__(self, size: int = 10000000, num_hashes: int = 7):
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)
        
    def add(self, item: str):
        for i in range(self.num_hashes):
            index = mmh3.hash(item, i) % self.size
            self.bit_array[index] = 1
            
    def __contains__(self, item: str):
        for i in range(self.num_hashes):
            index = mmh3.hash(item, i) % self.size
            if self.bit_array[index] == 0:
                return False
        return True

@dataclass
class WalletPriority:
    """priority queue item for processing high-value wallets first"""
    wallet: str
    depth: int
    priority: float  # based on estimated value/connections
    
    def __lt__(self, other):
        return self.priority > other.priority  # higher priority first

class AsyncRateLimiter:
    """token bucket rate limiter"""
    def __init__(self, rate: int = 10, per: float = 1.0, burst: int = 20):
        self.rate = rate
        self.per = per
        self.allowance = burst
        self.last_check = time.monotonic()
        self.lock = asyncio.Lock()
        
    async def acquire(self):
        async with self.lock:
            current = time.monotonic()
            time_passed = current - self.last_check
            self.last_check = current
            self.allowance += time_passed * (self.rate / self.per)
            
            if self.allowance > self.rate:
                self.allowance = self.rate
                
            if self.allowance < 1.0:
                sleep_time = (1.0 - self.allowance) * (self.per / self.rate)
                await asyncio.sleep(sleep_time)
                self.allowance = 0.0
            else:
                self.allowance -= 1.0

class NFTGraphBuilder:
    def __init__(self, api_key: str, rate_limit: int = 10, checkpoint_dir: str = "./checkpoints"):
        self.api_key = api_key
        self.base_url = f"https://eth-mainnet.g.alchemy.com/nft/v3/{api_key}"
        self.rate_limiter = AsyncRateLimiter(rate=rate_limit, burst=rate_limit*2)
        
        # multi-level caching
        self.visited_wallets = BloomFilter(size=50000000)  # 50M wallets
        self.visited_wallets_exact = set()  # exact set for accuracy
        self.contract_to_owners: Dict[str, Set[str]] = {}
        self.wallet_to_contracts: Dict[str, Set[str]] = {}
        self.edges: Dict[str, Set[str]] = defaultdict(set)
        
        # connection pooling
        self.session: Optional[aiohttp.ClientSession] = None
        self.connector = aiohttp.TCPConnector(
            limit=100,  # total connection pool
            limit_per_host=30,  # per-host limit
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        
        # performance metrics
        self.api_calls = 0
        self.cache_hits = 0
        self.start_time = time.time()
        
        # checkpoint management
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = 250  # Save checkpoint every 50 wallets
        self.wallets_processed_since_checkpoint = 0
        self.max_checkpoints = 10  # Keep only the last 10 checkpoints
        
        # pruning thresholds
        self.MAX_OWNERS_PER_CONTRACT = 10000
        self.MAX_NFTS_PER_WALLET = 1000
        self.MIN_SHARED_CONTRACTS = 2  # min contracts to create edge
        
        # known spam contracts to skip
        self.SPAM_CONTRACTS = {
            "0x57f1887a8bf19b14fc0df6fd9b2acc9af147ea85",  # ENS
            "0x495f947276749ce646f68ac8c248420045cb7b5e",  # opensea shared
        }
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(connector=self.connector)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint file in the checkpoint directory"""
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_*.json"))
        if not checkpoint_files:
            return None
        
        # Sort by timestamp in filename
        checkpoint_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        latest = checkpoint_files[-1]
        logger.info(f"Found latest checkpoint: {latest}")
        return latest
    
    async def load_checkpoint_async(self, checkpoint_path: str) -> Optional[Dict]:
        """Load checkpoint data from file"""
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return None
        
        logger.info(f"Loading checkpoint from {checkpoint_path}...")
        try:
            # For large files, read in chunks to avoid memory issues
            file_size = os.path.getsize(checkpoint_path)
            logger.info(f"Checkpoint file size: {file_size / (1024**3):.2f} GB")
            
            async with aiofiles.open(checkpoint_path, 'r') as f:
                content = await f.read()
                data = ujson.loads(content)
                
            logger.info(f"Checkpoint loaded successfully")
            return data
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return None
    
    def restore_state_from_checkpoint(self, checkpoint_data: Dict) -> List[WalletPriority]:
        """Restore the builder state from checkpoint data"""
        logger.info("Restoring state from checkpoint...")
        
        # Restore visited wallets
        visited_wallets_list = checkpoint_data.get("visited_wallets", [])
        for wallet in visited_wallets_list:
            self.visited_wallets.add(wallet)
            self.visited_wallets_exact.add(wallet)
        logger.info(f"Restored {len(visited_wallets_list)} visited wallets")
        
        # Restore contract to owners mapping
        contract_to_owners_data = checkpoint_data.get("contract_to_owners", {})
        self.contract_to_owners = {k: set(v) for k, v in contract_to_owners_data.items()}
        logger.info(f"Restored {len(self.contract_to_owners)} contract-to-owners mappings")
        
        # Restore wallet to contracts mapping
        wallet_to_contracts_data = checkpoint_data.get("wallet_to_contracts", {})
        self.wallet_to_contracts = {k: set(v) for k, v in wallet_to_contracts_data.items()}
        logger.info(f"Restored {len(self.wallet_to_contracts)} wallet-to-contracts mappings")
        
        # Restore edges
        edges_data = checkpoint_data.get("edges", {})
        for wallet, connections in edges_data.items():
            self.edges[wallet] = set(connections)
        logger.info(f"Restored edges for {len(self.edges)} wallets")
        
        # Restore stats
        stats = checkpoint_data.get("stats", {})
        self.api_calls = stats.get("api_calls", 0)
        self.cache_hits = stats.get("cache_hits", 0)
        logger.info(f"Restored stats: {self.api_calls} API calls, {self.cache_hits} cache hits")
        
        # Restore priority queue
        queue_data = checkpoint_data.get("queue", [])
        priority_queue = []
        for wallet, depth, priority in queue_data:
            heapq.heappush(priority_queue, WalletPriority(wallet, depth, priority))
        logger.info(f"Restored priority queue with {len(priority_queue)} items")
        
        return priority_queue
    
    def estimate_wallet_priority(self, wallet: str, depth: int) -> float:
        """estimate wallet importance for priority queue"""
        # simple heuristic: prefer wallets with more contracts
        num_contracts = len(self.wallet_to_contracts.get(wallet, set()))
        return num_contracts / (depth + 1)  # favor shallow depth
    
    @lru_cache(maxsize=10000)
    def should_skip_contract(self, contract: str) -> bool:
        """fast contract filtering"""
        return contract.lower() in self.SPAM_CONTRACTS
    
    async def batch_get_contract_owners(self, contracts: List[str]) -> Dict[str, Set[str]]:
        """batch process multiple contracts concurrently"""
        tasks = []
        contracts_to_fetch = []
        
        for contract in contracts:
            if contract in self.contract_to_owners:
                self.cache_hits += 1
                continue
            if self.should_skip_contract(contract):
                continue
            contracts_to_fetch.append(contract)
            tasks.append(self.get_contract_owners(contract))
        
        if not tasks:
            return {}
        
        logger.info(f"Fetching owners for {len(contracts_to_fetch)} contracts")
        
        # process in chunks to avoid overwhelming
        chunk_size = 10
        results = {}
        
        for i in range(0, len(tasks), chunk_size):
            chunk_tasks = tasks[i:i+chunk_size]
            chunk_contracts = contracts_to_fetch[i:i+chunk_size]
            
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
            
            for contract, result in zip(chunk_contracts, chunk_results):
                if isinstance(result, Exception):
                    logger.error(f"error fetching {contract[:8]}: {result}")
                    continue
                results[contract] = result
                
        return results
    
    async def get_wallet_nfts_optimized(self, address: str) -> List[Dict]:
        """optimized nft fetching with early termination using Alchemy"""
        logger.debug(f"Fetching NFTs for wallet {address[:8]}...")
        all_contracts = []
        page_key = None
        pages_fetched = 0
        max_pages = 5  # limit pagination for performance
        
        while pages_fetched < max_pages:
            await self.rate_limiter.acquire()
            self.api_calls += 1
            
            url = f"{self.base_url}/getContractsForOwner"
            params = {
                "owner": address,
                "withMetadata": "false"
            }
            
            if page_key:
                params["pageKey"] = page_key
                
            try:
                logger.debug(f"API call #{self.api_calls}: getContractsForOwner for {address[:8]} (page {pages_fetched + 1})")
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 429:
                        logger.info("Rate limit hit, waiting 5 seconds...")
                        await asyncio.sleep(5)
                        continue
                    
                    # Debug logging
                    text = await response.text()
                    
                    if response.status != 200:
                        logger.error(f"Non-200 status for wallet {address[:8]}: {response.status}")
                        logger.error(f"Response body: {text}")
                        return []
                    
                    try:
                        result = ujson.loads(text)
                    except Exception as e:
                        logger.error(f"Failed to parse JSON for wallet {address[:8]}: {e}")
                        logger.error(f"Response text: {text[:500]}...")
                        return []
                    
                    total = result.get("totalCount", 0)
                    if total > self.MAX_NFTS_PER_WALLET:
                        logger.debug(f"Wallet {address[:8]} has {total} NFTs - skipping (whale/exchange)")
                        return []  # skip whales/exchanges
                    
                    contracts = result.get("contracts", [])
                    all_contracts.extend(contracts)
                    pages_fetched += 1
                    logger.debug(f"Retrieved {len(contracts)} contracts, total so far: {len(all_contracts)}")
                    
                    page_key = result.get("pageKey")
                    if not page_key or len(all_contracts) >= total:
                        break
                        
            except Exception as e:
                logger.warning(f"Error fetching NFTs for {address[:8]}: {e}")
                return []
        
        logger.debug(f"Wallet {address[:8]} has {len(all_contracts)} NFT contracts")
        return all_contracts
    
    async def get_contract_owners(self, contract_address: str) -> Set[str]:
        """get owners with aggressive caching and early termination using Alchemy"""
        if contract_address in self.contract_to_owners:
            self.cache_hits += 1
            return self.contract_to_owners[contract_address]
        
        logger.debug(f"Fetching owners for contract {contract_address[:8]}...")
        owners = set()
        page_key = None
        pages_fetched = 0
        max_pages = 3  # limit for performance
        
        while pages_fetched < max_pages:
            await self.rate_limiter.acquire()
            self.api_calls += 1
            
            url = f"{self.base_url}/getOwnersForContract"
            params = {
                "contractAddress": contract_address
            }
            
            if page_key:
                params["pageKey"] = page_key
                
            try:
                logger.debug(f"API call #{self.api_calls}: getOwnersForContract for {contract_address[:8]} (page {pages_fetched + 1})")
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 429:
                        logger.info("Rate limit hit, waiting 5 seconds...")
                        await asyncio.sleep(5)
                        continue
                    
                    # Debug logging
                    logger.debug(f"Response status: {response.status}")
                    logger.debug(f"Response headers: {response.headers}")
                    
                    # Get raw text first
                    text = await response.text()
                    logger.debug(f"Raw response for {contract_address[:8]}: {text[:500]}...")  # First 500 chars
                    
                    # Check if response is valid
                    if response.status != 200:
                        logger.error(f"Non-200 status for {contract_address[:8]}: {response.status}")
                        logger.error(f"Response body: {text}")
                        return set()
                    
                    try:
                        result = ujson.loads(text)
                    except Exception as e:
                        logger.error(f"Failed to parse JSON for {contract_address[:8]}: {e}")
                        logger.error(f"Response text: {text}")
                        return set()
                    
                    # Check if result is a dict
                    if not isinstance(result, dict):
                        logger.error(f"Unexpected response type for {contract_address[:8]}: {type(result)}")
                        logger.error(f"Response: {result}")
                        return set()
                    
                    # early termination for huge collections
                    owner_list = result.get("owners", [])
                    if pages_fetched == 0 and len(owner_list) > self.MAX_OWNERS_PER_CONTRACT:
                        logger.debug(f"Contract {contract_address[:8]} has {len(owner_list)} owners - skipping (huge collection)")
                        return set()
                    
                    for owner_data in owner_list:
                        # Handle both string addresses and object format
                        if isinstance(owner_data, str):
                            owner_address = owner_data.lower()
                        elif isinstance(owner_data, dict):
                            owner_address = owner_data.get("ownerAddress", "").lower()
                        else:
                            logger.warning(f"Unexpected owner data type: {type(owner_data)}")
                            continue
                            
                        if owner_address:
                            owners.add(owner_address)
                    
                    pages_fetched += 1
                    logger.debug(f"Found {len(owner_list)} owners, total so far: {len(owners)}")
                    
                    page_key = result.get("pageKey")
                    if not page_key:
                        break
                        
            except Exception as e:
                logger.warning(f"Error fetching owners for {contract_address[:8]}: {e}")
                break
        
        logger.debug(f"Contract {contract_address[:8]} has {len(owners)} owners")
        self.contract_to_owners[contract_address] = owners
        return owners
    
    async def process_wallet_batch(self, wallet_batch: List[Tuple[str, int]], priority_queue: List):
        """process multiple wallets concurrently"""
        logger.info(f"Processing batch of {len(wallet_batch)} wallets")
        tasks = []
        
        for wallet, depth in wallet_batch:
            if wallet in self.visited_wallets:
                logger.debug(f"Skipping already visited wallet {wallet[:8]}")
                continue
                
            self.visited_wallets.add(wallet)
            self.visited_wallets_exact.add(wallet)
            tasks.append(self.process_single_wallet(wallet, depth, priority_queue))
        
        if tasks:
            logger.info(f"Starting {len(tasks)} concurrent wallet processing tasks")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log any errors
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error processing wallet: {result}")
            
            self.wallets_processed_since_checkpoint += len(tasks)
            logger.info(f"Batch complete. Total wallets processed: {len(self.visited_wallets_exact)}")
    
    async def process_single_wallet(self, wallet: str, depth: int, priority_queue: List):
        """process individual wallet"""
        logger.info(f"Processing wallet {wallet[:8]} at depth {depth}")
        contracts_data = await self.get_wallet_nfts_optimized(wallet)
        if not contracts_data:
            logger.debug(f"No NFTs found for wallet {wallet[:8]}")
            return
            
        contracts = {contract["address"].lower() for contract in contracts_data if contract.get("address")}
        self.wallet_to_contracts[wallet] = contracts
        logger.info(f"Wallet {wallet[:8]} owns NFTs from {len(contracts)} contracts")
        
        # batch fetch all contract owners
        owner_results = await self.batch_get_contract_owners(list(contracts))
        
        # build edges efficiently
        new_connections = set()
        for contract, owners in owner_results.items():
            for owner in owners:
                if owner != wallet and owner:
                    # only create edge if significant overlap
                    shared = len(self.wallet_to_contracts.get(owner, set()) & contracts)
                    if shared >= self.MIN_SHARED_CONTRACTS or depth < 2:
                        self.edges[wallet].add(owner)
                        self.edges[owner].add(wallet)
                        new_connections.add(owner)
        
        # add to priority queue
        new_count = 0
        for owner in new_connections:
            if owner not in self.visited_wallets:
                priority = self.estimate_wallet_priority(owner, depth + 1)
                heapq.heappush(priority_queue, WalletPriority(owner, depth + 1, priority))
                new_count += 1
        
        logger.info(f"Wallet {wallet[:8]} connected to {len(new_connections)} other wallets, added {new_count} to queue")
    
    async def build_graph(self, start_wallet: str, max_depth: int = 3, max_nodes: int = 10000, resume_from_checkpoint: bool = False) -> nx.Graph:
        """build graph with performance optimizations"""
        priority_queue = []
        
        # Check if we should resume from checkpoint
        if resume_from_checkpoint:
            latest_checkpoint = self.get_latest_checkpoint()
            if latest_checkpoint:
                checkpoint_data = await self.load_checkpoint_async(latest_checkpoint)
                if checkpoint_data:
                    priority_queue = self.restore_state_from_checkpoint(checkpoint_data)
                    logger.info(f"Resumed from checkpoint with {len(self.visited_wallets_exact)} visited wallets")
                else:
                    logger.error("Failed to load checkpoint data, starting fresh")
                    priority_queue = [WalletPriority(start_wallet.lower(), 0, 100.0)]
            else:
                logger.info("No checkpoint found, starting fresh")
                priority_queue = [WalletPriority(start_wallet.lower(), 0, 100.0)]
        else:
            priority_queue = [WalletPriority(start_wallet.lower(), 0, 100.0)]
        
        logger.info(f"Starting graph build" + (" (resumed)" if resume_from_checkpoint and priority_queue else " from wallet " + start_wallet))
        logger.info(f"Settings: max_depth={max_depth}, max_nodes={max_nodes}, rate_limit={self.rate_limiter.rate}/s")
        
        try:
            nodes_processed = len(self.visited_wallets_exact)  # Start from checkpoint progress
            batch_size = 20  # process 20 wallets concurrently
            last_stats_time = time.time()
            
            while priority_queue and nodes_processed < max_nodes:
                logger.info(f"Queue size: {len(priority_queue)}, Nodes processed: {nodes_processed}")
                
                # get next batch from priority queue
                batch = []
                while priority_queue and len(batch) < batch_size:
                    item = heapq.heappop(priority_queue)
                    if item.depth < max_depth and item.wallet not in self.visited_wallets:
                        batch.append((item.wallet, item.depth))
                
                if batch:
                    await self.process_wallet_batch(batch, priority_queue)
                    nodes_processed += len(batch)
                    
                    # More frequent stats - every 10 nodes or 30 seconds
                    current_time = time.time()
                    if nodes_processed % 10 == 0 or (current_time - last_stats_time) > 30:
                        elapsed = current_time - self.start_time
                        rate = self.api_calls / elapsed if elapsed > 0 else 0
                        logger.info(f"=== STATS ===")
                        logger.info(f"Nodes processed: {nodes_processed}/{max_nodes}")
                        logger.info(f"API calls: {self.api_calls}")
                        logger.info(f"Cache hits: {self.cache_hits}")
                        logger.info(f"API rate: {rate:.1f}/s")
                        logger.info(f"Queue size: {len(priority_queue)}")
                        logger.info(f"Unique wallets discovered: {len(self.visited_wallets_exact)}")
                        logger.info(f"Edges created: {sum(len(v) for v in self.edges.values()) // 2}")
                        logger.info(f"=============")
                        last_stats_time = current_time
                    
                    # checkpoint
                    if self.wallets_processed_since_checkpoint >= self.checkpoint_interval:
                        await self.save_checkpoint_async(priority_queue)
                        self.wallets_processed_since_checkpoint = 0
                        
                else:
                    if priority_queue:
                        logger.warning(f"No valid items in batch, but queue has {len(priority_queue)} items")
                    else:
                        logger.info("Queue is empty, finishing")
                        
        except Exception as e:
            logger.error(f"Error in build_graph: {e}", exc_info=True)
            await self.save_checkpoint_async(priority_queue)
            raise
        
        # build final graph
        logger.info(f"Building NetworkX graph from collected data...")
        G = nx.Graph()
        edge_count = 0
        for wallet, connections in self.edges.items():
            for connected_wallet in connections:
                shared = len(self.wallet_to_contracts.get(wallet, set()) & 
                           self.wallet_to_contracts.get(connected_wallet, set()))
                if shared > 0:
                    G.add_edge(wallet, connected_wallet, weight=shared)
                    edge_count += 1
                    if edge_count % 1000 == 0:
                        logger.info(f"Added {edge_count} edges to graph...")
        
        logger.info(f"\n=== FINAL STATS ===")
        logger.info(f"Graph nodes: {G.number_of_nodes()}")
        logger.info(f"Graph edges: {G.number_of_edges()}")
        logger.info(f"Total API calls: {self.api_calls}")
        logger.info(f"Total cache hits: {self.cache_hits}")
        logger.info(f"Cache hit rate: {self.cache_hits / (self.api_calls + self.cache_hits) * 100:.1f}%")
        logger.info(f"Total runtime: {time.time() - self.start_time:.1f} seconds")
        logger.info(f"==================")
        
        return G
    
    async def save_checkpoint_async(self, priority_queue):
        """async checkpoint saving with cleanup of old checkpoints"""
        logger.info("Saving checkpoint...")
        checkpoint_data = {
            "visited_wallets": list(self.visited_wallets_exact),
            "contract_to_owners": {k: list(v) for k, v in self.contract_to_owners.items()},
            "wallet_to_contracts": {k: list(v) for k, v in self.wallet_to_contracts.items()},
            "edges": {k: list(v) for k, v in self.edges.items()},
            "queue": [(w.wallet, w.depth, w.priority) for w in priority_queue],
            "stats": {
                "api_calls": self.api_calls,
                "cache_hits": self.cache_hits,
                "nodes_processed": len(self.visited_wallets_exact)
            }
        }
        
        filename = f"{self.checkpoint_dir}/checkpoint_{int(time.time())}.json"
        async with aiofiles.open(filename, 'w') as f:
            await f.write(ujson.dumps(checkpoint_data))
        logger.info(f"Checkpoint saved to {filename}")
        
        # Clean up old checkpoints, keeping only the last max_checkpoints
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoint files, keeping only the last max_checkpoints"""
        try:
            checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_*.json"))
            if len(checkpoint_files) <= self.max_checkpoints:
                return  # No cleanup needed
            
            # Sort by timestamp in filename
            checkpoint_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            
            # Keep only the latest max_checkpoints
            files_to_delete = checkpoint_files[:-self.max_checkpoints]
            
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted old checkpoint: {os.path.basename(file_path)}")
                except Exception as e:
                    logger.warning(f"Failed to delete checkpoint {file_path}: {e}")
            
            if files_to_delete:
                logger.info(f"Cleaned up {len(files_to_delete)} old checkpoints, keeping {len(checkpoint_files) - len(files_to_delete)}")
                
        except Exception as e:
            logger.warning(f"Error during checkpoint cleanup: {e}")

# usage
async def main():
    # Load API key from environment variable
    api_key = os.getenv("ALCHEMY_API_KEY")
    
    if not api_key:
        logger.error("ALCHEMY_API_KEY environment variable not found!")
        logger.error("Please create a .env file with: ALCHEMY_API_KEY=your_actual_key_here")
        logger.error("Get a free API key at https://www.alchemy.com/")
        return
    
    start_wallet = "0xf5dd8ff9fc6a00da37e93e873ed64b9f62ba038d"
    
    # Set this to True to resume from the latest checkpoint
    resume_from_checkpoint = True
    
    logger.info("Starting NFT network builder with Alchemy...")
    
    async with NFTGraphBuilder(api_key, rate_limit=20) as builder:
        graph = await builder.build_graph(
            start_wallet, 
            max_depth=3,
            max_nodes=10000,  # limit for testing
            resume_from_checkpoint=resume_from_checkpoint
        )
        
        output_file = "nft_network_alchemy.gexf"
        logger.info(f"Saving graph to {output_file}")
        nx.write_gexf(graph, output_file)
        logger.info("Done!")

if __name__ == "__main__":
    asyncio.run(main())