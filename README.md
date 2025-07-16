# NFT Network Builder with Alchemy

This repository contains tools for building network graphs of NFT collectors by analyzing shared ownership patterns using the Alchemy API.

## ⚠️ API Key Required

**You must have an Alchemy API key to use these tools.** Get a free API key at [Alchemy.com](https://www.alchemy.com/).

## 🆕 Generic NFT Network Builder

**New Feature**: Build networks for ANY NFT contract on Ethereum!

### Quick Start

```bash
# Create .env file
echo "ALCHEMY_API_KEY=your_api_key_here" > .env

# Run with any contract address
python nft_network_builder.py 0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D

# Or use shortcuts for popular collections
python run_nft_network.py bayc
```

See [README_nft_network.md](README_nft_network.md) for detailed documentation on the generic NFT network builder.

## Migration from Moralis to Alchemy

This codebase has been migrated from Moralis to Alchemy API. Key changes include:

- **API Endpoints**: 
  - `get_wallet_nfts` → `getContractsForOwner`
  - `get_nft_owners` → `getOwnersForContract`
- **Response Format**: Adapted to Alchemy's JSON structure
- **Pagination**: Uses `pageKey` instead of `cursor`
- **Spam Filtering**: Leverages Alchemy's built-in spam detection

## Features

- **Efficient API Usage**: Rate limiting and connection pooling for optimal performance
- **Smart Caching**: Bloom filters and in-memory caching to avoid redundant API calls
- **Scalable Design**: Priority queue processing and early termination for large collections
- **Checkpoint System**: Automatic saving of progress for resumability
- **Spam Filtering**: Built-in filtering of spam contracts using Alchemy's spam detection

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

1. Get an Alchemy API key from [Alchemy.com](https://www.alchemy.com/)
2. Update the `api_key` variable in `gen_network.py`:

```python
api_key = "your-alchemy-api-key-here"  # Replace with your actual key
```

## Usage

Run the script:

```bash
python gen_network.py
```

The script will:
1. Start from a seed wallet address
2. Fetch all NFT contracts owned by that wallet
3. For each contract, fetch other owners
4. Build connections between wallets based on shared NFT ownership
5. Continue exploring the network up to the specified depth
6. Export the network as a GEXF file for visualization

## Configuration Options

- `max_depth`: How many hops from the starting wallet to explore (default: 3)
- `max_nodes`: Maximum number of wallets to process (default: 5000)
- `rate_limit`: API calls per second (default: 20)
- `MIN_SHARED_CONTRACTS`: Minimum shared contracts to create an edge (default: 2)

## Output

The tool generates a `nft_network_alchemy.gexf` file that can be opened in network visualization tools like:
- Gephi
- Cytoscape
- NetworkX (Python)

## Architecture

The tool uses:
- **Async/await** for concurrent API calls
- **Priority queue** to process high-value wallets first
- **Bloom filters** for space-efficient visited tracking
- **Checkpoint system** for fault tolerance

## API Endpoints Used

- `getContractsForOwner`: Fetches all NFT contracts owned by a wallet
- `getOwnersForContract`: Fetches all owners of a specific NFT contract

## Performance Optimizations

- Batch processing of multiple wallets concurrently
- Early termination for whale wallets and huge collections
- Aggressive caching of contract-to-owner mappings
- Connection pooling for HTTP requests
- Spam contract filtering 