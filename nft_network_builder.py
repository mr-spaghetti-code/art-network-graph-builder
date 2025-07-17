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
import gzip

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
        
        # error tracking
        self.consecutive_errors = 0
        self.total_errors = 0
        self.max_consecutive_errors = 10  # Pause after this many consecutive errors
        self.max_total_errors = 50  # Pause after this many total errors
        
        # checkpoint management
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = 500  # Save checkpoint every 500 wallets
        self.wallets_processed_since_checkpoint = 0
        self.max_checkpoints = 5  # Keep only the last 10 checkpoints
        
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
        # Look for both new pickle and old JSON checkpoints
        pickle_files = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_*.pkl.gz"))
        json_files = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_*.json"))
        
        checkpoint_files = pickle_files + json_files
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
            logger.info(f"Checkpoint file size: {file_size / (1024**2):.2f} MB")
            
            # Check if it's a pickle file (new format) or json (old format)
            if checkpoint_path.endswith('.pkl.gz'):
                # New compressed pickle format
                with gzip.open(checkpoint_path, 'rb') as f:
                    data = pickle.load(f)
            elif checkpoint_path.endswith('.json'):
                # Old JSON format (for backward compatibility)
                async with aiofiles.open(checkpoint_path, 'r') as f:
                    content = await f.read()
                    data = ujson.loads(content)
            else:
                logger.error(f"Unknown checkpoint format: {checkpoint_path}")
                return None
                
            logger.info(f"Checkpoint loaded successfully")
            return data
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return None
    
    def restore_state_from_checkpoint(self, checkpoint_data: Dict) -> Tuple[List[WalletPriority], Optional[set]]:
        """Restore the builder state from checkpoint data"""
        logger.info("Restoring state from checkpoint...")
        
        # Restore visited wallets
        visited_wallets_data = checkpoint_data.get("visited_wallets", [])
        # Handle both list (old JSON format) and set (new pickle format)
        if isinstance(visited_wallets_data, list):
            for wallet in visited_wallets_data:
                self.visited_wallets.add(wallet)
                self.visited_wallets_exact.add(wallet)
        else:  # It's already a set from pickle
            self.visited_wallets_exact = visited_wallets_data
            for wallet in visited_wallets_data:
                self.visited_wallets.add(wallet)
        logger.info(f"Restored {len(self.visited_wallets_exact)} visited wallets")
        
        # Restore contract to owners mapping
        contract_to_owners_data = checkpoint_data.get("contract_to_owners", {})
        # Handle both formats
        if contract_to_owners_data:
            # Check if it's old format (lists) or new format (sets)
            sample_value = next(iter(contract_to_owners_data.values()), None)
            if sample_value is not None and isinstance(sample_value, list):
                # Old format: values are lists, convert to sets
                self.contract_to_owners = {k: set(v) for k, v in contract_to_owners_data.items()}
            else:
                # New format: values are already sets
                self.contract_to_owners = contract_to_owners_data
        logger.info(f"Restored {len(self.contract_to_owners)} contract-to-owners mappings")
        
        # Restore wallet to contracts mapping
        wallet_to_contracts_data = checkpoint_data.get("wallet_to_contracts", {})
        # Handle both formats
        if wallet_to_contracts_data:
            # Check if it's old format (lists) or new format (sets)
            sample_value = next(iter(wallet_to_contracts_data.values()), None)
            if sample_value is not None and isinstance(sample_value, list):
                # Old format: values are lists, convert to sets
                self.wallet_to_contracts = {k: set(v) for k, v in wallet_to_contracts_data.items()}
            else:
                # New format: values are already sets
                self.wallet_to_contracts = wallet_to_contracts_data
        logger.info(f"Restored {len(self.wallet_to_contracts)} wallet-to-contracts mappings")
        
        # Restore edges
        edges_data = checkpoint_data.get("edges", {})
        # Handle both formats
        if edges_data:
            # Check if it's old format (lists) or new format (sets/defaultdict)
            sample_value = next(iter(edges_data.values()), None)
            if sample_value is not None and isinstance(sample_value, list):
                # Old format: values are lists, convert to sets
                for wallet, connections in edges_data.items():
                    self.edges[wallet] = set(connections)
            else:
                # New format: values are already sets
                # Convert regular dict back to defaultdict
                self.edges = defaultdict(set, edges_data)
        logger.info(f"Restored edges for {len(self.edges)} wallets")
        
        # Restore stats
        stats = checkpoint_data.get("stats", {})
        self.api_calls = stats.get("api_calls", 0)
        self.cache_hits = stats.get("cache_hits", 0)
        self.total_errors = stats.get("total_errors", 0)
        self.consecutive_errors = 0  # Reset consecutive errors on resume
        logger.info(f"Restored stats: {self.api_calls} API calls, {self.cache_hits} cache hits, {self.total_errors} total errors")
        
        # Restore priority queue
        queue_data = checkpoint_data.get("queue", [])
        priority_queue = []
        for wallet, depth, priority in queue_data:
            heapq.heappush(priority_queue, WalletPriority(wallet, depth, priority))
        logger.info(f"Restored priority queue with {len(priority_queue)} items")
        
        # Restore seed wallets
        seed_wallets = None
        if "seed_wallets" in checkpoint_data:
            seed_data = checkpoint_data["seed_wallets"]
            # Handle both list and set formats
            seed_wallets = set(seed_data) if isinstance(seed_data, list) else seed_data
            logger.info(f"Restored {len(seed_wallets)} seed wallets")
        
        return priority_queue, seed_wallets
    
    async def get_all_contract_holders(self, contract_address: str) -> Set[str]:
        """Fetch all holders of a given NFT contract"""
        logger.info(f"Fetching all holders of contract: {contract_address}")
        
        all_owners = set()
        page_key = None
        pages_fetched = 0
        max_pages = 100  # Most contracts have a reasonable number of holders
        
        while pages_fetched < max_pages:
            await self.rate_limiter.acquire()
            self.api_calls += 1
            
            url = f"{self.base_url}/getOwnersForContract"
            params = {
                "contractAddress": contract_address,
                "withTokenBalances": "false",
                "excludeFilters[]": "SPAM",
                "spamConfidenceLevel": "HIGH"
            }
            
            if page_key:
                params["pageKey"] = page_key
                
            try:
                logger.info(f"Fetching holders page {pages_fetched + 1}")
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 429:
                        logger.info("Rate limit hit, waiting 5 seconds...")
                        await asyncio.sleep(5)
                        continue
                    
                    text = await response.text()
                    
                    if response.status != 200:
                        logger.error(f"Error fetching contract holders: {response.status}")
                        logger.error(f"Response: {text}")
                        break
                    
                    result = ujson.loads(text)
                    owner_list = result.get("owners", [])
                    
                    for owner_data in owner_list:
                        if isinstance(owner_data, str):
                            owner_address = owner_data.lower()
                        elif isinstance(owner_data, dict):
                            owner_address = owner_data.get("ownerAddress", "").lower()
                        else:
                            continue
                            
                        if owner_address:
                            all_owners.add(owner_address)
                    
                    pages_fetched += 1
                    logger.info(f"Found {len(owner_list)} owners in this page, total: {len(all_owners)}")
                    
                    page_key = result.get("pageKey")
                    if not page_key:
                        break
                        
            except Exception as e:
                logger.error(f"Error fetching contract holders: {e}")
                break
        
        logger.info(f"Found total of {len(all_owners)} unique holders for contract {contract_address[:8]}...")
        
        # Cache the contract owners
        self.contract_to_owners[contract_address.lower()] = all_owners
        
        return all_owners
    
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
                "withMetadata": "false",
                "excludeFilters[]": "SPAM",
                "spamConfidenceLevel": "HIGH"
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
                        
                        # Track errors
                        self.consecutive_errors += 1
                        self.total_errors += 1
                        
                        # Check for rate limit error
                        if response.status == 403 and "capacity limit exceeded" in text.lower():
                            logger.error("API capacity limit exceeded! Need to pause and save checkpoint.")
                            raise Exception("API_CAPACITY_EXCEEDED")
                        
                        return []
                    
                    try:
                        result = ujson.loads(text)
                    except Exception as e:
                        logger.error(f"Failed to parse JSON for wallet {address[:8]}: {e}")
                        logger.error(f"Response text: {text[:500]}...")
                        self.consecutive_errors += 1
                        self.total_errors += 1
                        return []
                    
                    # Success - reset consecutive errors
                    self.consecutive_errors = 0
                    
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
                self.consecutive_errors += 1
                self.total_errors += 1
                
                # Re-raise critical errors
                if str(e) == "API_CAPACITY_EXCEEDED":
                    raise
                    
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
                "contractAddress": contract_address,
                "excludeFilters[]": "SPAM",
                "spamConfidenceLevel": "HIGH"
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
                        
                        # Track errors
                        self.consecutive_errors += 1
                        self.total_errors += 1
                        
                        # Check for rate limit error
                        if response.status == 403 and "capacity limit exceeded" in text.lower():
                            logger.error("API capacity limit exceeded! Need to pause and save checkpoint.")
                            raise Exception("API_CAPACITY_EXCEEDED")
                            
                        return set()
                    
                    try:
                        result = ujson.loads(text)
                    except Exception as e:
                        logger.error(f"Failed to parse JSON for {contract_address[:8]}: {e}")
                        logger.error(f"Response text: {text}")
                        self.consecutive_errors += 1
                        self.total_errors += 1
                        return set()
                    
                    # Check if result is a dict
                    if not isinstance(result, dict):
                        logger.error(f"Unexpected response type for {contract_address[:8]}: {type(result)}")
                        logger.error(f"Response: {result}")
                        self.consecutive_errors += 1
                        self.total_errors += 1
                        return set()
                    
                    # Success - reset consecutive errors
                    self.consecutive_errors = 0
                    
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
    
    async def build_graph(self, start_wallets=None, max_depth: int = 3, max_nodes: int = 10000, resume_from_checkpoint: bool = False, seed_wallets: set = None) -> nx.Graph:
        """build graph with performance optimizations
        
        Args:
            start_wallets: Can be a single wallet address (str), a list of wallet addresses, or None
            max_depth: Maximum depth to traverse from starting wallets
            max_nodes: Maximum number of nodes to process
            resume_from_checkpoint: Whether to resume from the latest checkpoint
            seed_wallets: Set of wallet addresses to mark as "Seed" nodes (for contract mode)
        """
        priority_queue = []
        
        # Handle different input types for start_wallets
        if isinstance(start_wallets, str):
            initial_wallets = [(start_wallets.lower(), 0, 100.0)]
        elif isinstance(start_wallets, (list, set)):
            # Give all starting wallets equal high priority
            initial_wallets = [(wallet.lower(), 0, 100.0) for wallet in start_wallets]
        else:
            initial_wallets = []
        
        # Check if we should resume from checkpoint
        restored_seed_wallets = None
        if resume_from_checkpoint:
            latest_checkpoint = self.get_latest_checkpoint()
            if latest_checkpoint:
                checkpoint_data = await self.load_checkpoint_async(latest_checkpoint)
                if checkpoint_data:
                    priority_queue, restored_seed_wallets = self.restore_state_from_checkpoint(checkpoint_data)
                    # Use restored seed wallets if available and no new seed wallets provided
                    if restored_seed_wallets and not seed_wallets:
                        seed_wallets = restored_seed_wallets
                    logger.info(f"Resumed from checkpoint with {len(self.visited_wallets_exact)} visited wallets")
                else:
                    logger.error("Failed to load checkpoint data, starting fresh")
                    for wallet, depth, priority in initial_wallets:
                        heapq.heappush(priority_queue, WalletPriority(wallet, depth, priority))
            else:
                logger.info("No checkpoint found, starting fresh")
                for wallet, depth, priority in initial_wallets:
                    heapq.heappush(priority_queue, WalletPriority(wallet, depth, priority))
        else:
            for wallet, depth, priority in initial_wallets:
                heapq.heappush(priority_queue, WalletPriority(wallet, depth, priority))
        
        logger.info(f"Starting graph build" + 
                   (f" (resumed)" if resume_from_checkpoint and len(self.visited_wallets_exact) > 0 
                    else f" with {len(initial_wallets)} starting wallet(s)"))
        logger.info(f"Settings: max_depth={max_depth}, max_nodes={max_nodes}, rate_limit={self.rate_limiter.rate}/s")
        
        # Initialize graph early in case we need to return it due to errors
        G = nx.Graph()
        
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
                    
                    # Check error thresholds
                    if self.consecutive_errors >= self.max_consecutive_errors:
                        logger.error(f"Hit {self.consecutive_errors} consecutive errors. Pausing to save checkpoint.")
                        await self.save_checkpoint_async(priority_queue, seed_wallets)
                        logger.error("Too many consecutive errors. Please check your API limits and resume later.")
                        return G
                        
                    if self.total_errors >= self.max_total_errors:
                        logger.error(f"Hit {self.total_errors} total errors. Pausing to save checkpoint.")
                        await self.save_checkpoint_async(priority_queue, seed_wallets)
                        logger.error("Too many total errors. Please check your API limits and resume later.")
                        return G
                    
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
                        logger.info(f"Errors: {self.consecutive_errors} consecutive, {self.total_errors} total")
                        logger.info(f"=============")
                        last_stats_time = current_time
                    
                    # checkpoint
                    if self.wallets_processed_since_checkpoint >= self.checkpoint_interval:
                        await self.save_checkpoint_async(priority_queue, seed_wallets)
                        self.wallets_processed_since_checkpoint = 0
                        
                else:
                    if priority_queue:
                        logger.warning(f"No valid items in batch, but queue has {len(priority_queue)} items")
                    else:
                        logger.info("Queue is empty, finishing")
                        
        except Exception as e:
            logger.error(f"Error in build_graph: {e}")
            
            # Always save checkpoint on error
            await self.save_checkpoint_async(priority_queue, seed_wallets)
            
            # Special handling for capacity exceeded
            if str(e) == "API_CAPACITY_EXCEEDED":
                logger.error("API capacity limit exceeded. Graph building paused.")
                logger.error("Please upgrade your API plan or wait for the next billing period.")
                logger.error("You can resume from checkpoint when ready.")
            else:
                logger.error(f"Unexpected error: {e}", exc_info=True)
            
            # Build partial graph from what we have so far
            logger.info("Building partial graph from collected data...")
        
        # build final graph (or continue building if we had an error)
        logger.info(f"Building NetworkX graph from collected data...")
        
        # G is already initialized above
        edge_count = 0
        
        # First, add all nodes with attributes
        for wallet in self.visited_wallets_exact:
            # Mark seed nodes if provided
            if seed_wallets and wallet in seed_wallets:
                G.add_node(wallet, seed=True, label="Seed")
            else:
                G.add_node(wallet)
        
        # Then add edges
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
    
    async def save_checkpoint_async(self, priority_queue, seed_wallets=None):
        """async checkpoint saving with cleanup of old checkpoints"""
        logger.info("Saving checkpoint...")
        start_time = time.time()
        
        # No need to convert sets to lists with pickle!
        checkpoint_data = {
            "visited_wallets": self.visited_wallets_exact,  # Keep as set
            "contract_to_owners": self.contract_to_owners,  # Keep sets as sets
            "wallet_to_contracts": self.wallet_to_contracts,  # Keep sets as sets
            "edges": self.edges,  # Keep sets as sets
            "queue": [(w.wallet, w.depth, w.priority) for w in priority_queue],
            "stats": {
                "api_calls": self.api_calls,
                "cache_hits": self.cache_hits,
                "nodes_processed": len(self.visited_wallets_exact),
                "total_errors": self.total_errors
            }
        }
        
        # Save seed wallets if provided
        if seed_wallets:
            checkpoint_data["seed_wallets"] = seed_wallets  # Keep as set
        
        filename = f"{self.checkpoint_dir}/checkpoint_{int(time.time())}.pkl.gz"
        
        # Use synchronous gzip for better performance
        # (aiofiles doesn't work well with gzip)
        logger.info(f"Writing checkpoint to {filename}...")
        with gzip.open(filename, 'wb', compresslevel=1) as f:  # compresslevel=1 for speed
            pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        elapsed = time.time() - start_time
        file_size = os.path.getsize(filename) / (1024**2)  # MB
        logger.info(f"Checkpoint saved to {filename} ({file_size:.1f} MB) in {elapsed:.1f} seconds")
        
        # Clean up old checkpoints, keeping only the last max_checkpoints
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoint files, keeping only the last max_checkpoints"""
        try:
            checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_*.pkl.gz"))
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
    
    # Parse command line arguments
    import sys
    if len(sys.argv) < 2:
        logger.error("Usage:")
        logger.error("  For NFT contract: python nft_network_builder.py --contract <contract_address>")
        logger.error("  For wallet(s): python nft_network_builder.py --wallet <wallet_address> [wallet2 wallet3 ...]")
        logger.error("  Default (CryptoPunks): python nft_network_builder.py")
        
        # Default to CryptoPunks for backward compatibility
        mode = "contract"
        input_value = "0xb47e3cd837ddf8e4c57f05d70ab865de6e193bbb"
        logger.info("Using default CryptoPunks contract")
    elif sys.argv[1] == "--contract":
        mode = "contract"
        if len(sys.argv) < 3:
            logger.error("Please provide a contract address after --contract")
            return
        input_value = sys.argv[2]
    elif sys.argv[1] == "--wallet":
        mode = "wallet"
        if len(sys.argv) < 3:
            logger.error("Please provide at least one wallet address after --wallet")
            return
        input_value = sys.argv[2:]  # Get all wallet addresses
    else:
        # Backward compatibility - assume it's a contract address
        mode = "contract"
        input_value = sys.argv[1]
    
    # Set this to True to resume from the latest checkpoint
    resume_from_checkpoint = True
    
    # Determine checkpoint directory and output file names
    if mode == "contract":
        contract_short = input_value[:10].lower().replace("0x", "")
        checkpoint_dir = f"./checkpoints_{contract_short}"
        output_file = f"nft_network_{contract_short}.gexf"
        stats_file = f"nft_network_stats_{contract_short}.json"
    else:
        # For wallet mode, use first wallet address for naming
        first_wallet = input_value[0] if isinstance(input_value, list) else input_value
        wallet_short = first_wallet[:10].lower().replace("0x", "")
        checkpoint_dir = f"./checkpoints_wallet_{wallet_short}"
        output_file = f"nft_network_wallet_{wallet_short}.gexf"
        stats_file = f"nft_network_stats_wallet_{wallet_short}.json"
    
    async with NFTGraphBuilder(api_key, rate_limit=20, checkpoint_dir=checkpoint_dir) as builder:
        if mode == "contract":
            # Contract mode: fetch all holders first
            logger.info(f"Starting NFT network builder for contract {input_value}")
            logger.info(f"Fetching all holders of contract {input_value}...")
            contract_holders = await builder.get_all_contract_holders(input_value)
            
            if not contract_holders:
                logger.error("Failed to fetch contract holders!")
                return
            
            logger.info(f"Starting graph build with {len(contract_holders)} contract holders")
            start_wallets = list(contract_holders)
            max_depth = 2  # Reduced depth since we're starting with many wallets
            max_nodes = 50000  # Large limit for comprehensive network
            
        else:
            # Wallet mode: start directly with provided wallets
            logger.info(f"Starting NFT network builder with {len(input_value)} wallet(s)")
            start_wallets = input_value if isinstance(input_value, list) else [input_value]
            max_depth = 3  # Can go deeper with fewer starting points
            max_nodes = 10000  # Smaller limit for wallet-based exploration
        
        # Build the graph
        # Pass seed_wallets only in contract mode
        seed_wallets_set = set(start_wallets) if mode == "contract" else None
        
        graph = await builder.build_graph(
            start_wallets=start_wallets,
            max_depth=max_depth,
            max_nodes=max_nodes,
            resume_from_checkpoint=resume_from_checkpoint,
            seed_wallets=seed_wallets_set
        )
        
        logger.info(f"Saving graph to {output_file}")
        nx.write_gexf(graph, output_file)
        
        # Save statistics
        stats = {
            "mode": mode,
            "input": input_value,
            "graph_nodes": graph.number_of_nodes(),
            "graph_edges": graph.number_of_edges(),
            "api_calls": builder.api_calls,
            "cache_hits": builder.cache_hits,
            "total_errors": builder.total_errors,
            "timestamp": datetime.now().isoformat()
        }
        
        if mode == "contract":
            stats["total_holders"] = len(start_wallets)
            # Count seed nodes in final graph
            seed_count = sum(1 for node, attrs in graph.nodes(data=True) if attrs.get('seed', False))
            stats["seed_nodes_in_graph"] = seed_count
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Saved statistics to {stats_file}")
        logger.info("Done!")

if __name__ == "__main__":
    asyncio.run(main())