# NFT Network Builder

A high-performance tool for building social network graphs from NFT collections on Ethereum. This tool analyzes NFT ownership patterns to create networks showing relationships between wallet addresses based on shared NFT collections.

## Features

- **Generic Contract Support**: Works with any ERC-721 NFT contract on Ethereum
- **Dual Mode Operation**: Start from either NFT contracts or wallet addresses
- **High Performance**: Concurrent processing, intelligent caching, and rate limiting
- **Smart Error Handling**: Automatically pauses and saves progress when hitting API limits
- **Checkpoint System**: Resume interrupted builds without losing progress
- **Scalable**: Handles large collections with thousands of holders
- **Network Analysis**: Outputs graphs in GEXF format for visualization

## Prerequisites

1. Python 3.8 or higher
2. An Alchemy API key (get one free at https://www.alchemy.com/)

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your Alchemy API key:
```bash
ALCHEMY_API_KEY=your_actual_key_here
```

## Usage

### Starting from an NFT Contract

Analyze all holders of a specific NFT contract:

```bash
# Using contract flag
python nft_network_builder.py --contract <contract_address>

# Backward compatibility (assumes contract if no flag)
python nft_network_builder.py <contract_address>

# Example with CryptoPunks
python nft_network_builder.py --contract 0xb47e3cd837ddf8e4c57f05d70ab865de6e193bbb
```

### Starting from Wallet Address(es)

Analyze the network starting from specific wallet(s):

```bash
# Single wallet
python nft_network_builder.py --wallet <wallet_address>

# Multiple wallets
python nft_network_builder.py --wallet <wallet1> <wallet2> <wallet3>

# Example
python nft_network_builder.py --wallet 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
```

### Using the Helper Script

The `run_nft_network.py` script provides shortcuts for popular collections:

```bash
# Using a shortcut
python run_nft_network.py bayc

# Or with full address
python run_nft_network.py 0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D
```

Available shortcuts:
- `punks` - CryptoPunks
- `bayc` - Bored Ape Yacht Club
- `mayc` - Mutant Ape Yacht Club
- `azuki` - Azuki
- `doodles` - Doodles
- `coolcats` - Cool Cats
- `clonex` - Clone X
- `worldofwomen` - World of Women
- `pudgypenguins` - Pudgy Penguins
- `meebits` - Meebits

## How It Works

1. **Fetch Contract Holders**: Gets all current holders of the specified NFT contract
2. **Build Network**: For each holder:
   - Fetches their other NFT holdings
   - Identifies connections with other holders based on shared collections
   - Creates edges weighted by the number of shared collections
3. **Export Graph**: Saves the network as a GEXF file for visualization

## Output Files

For contract mode:
- `nft_network_<contract_short>.gexf` - Network graph file
- `nft_network_stats_<contract_short>.json` - Statistics about the network
- `checkpoints_<contract_short>/` - Checkpoint files for resuming

For wallet mode:
- `nft_network_wallet_<wallet_short>.gexf` - Network graph file
- `nft_network_stats_wallet_<wallet_short>.json` - Statistics about the network
- `checkpoints_wallet_<wallet_short>/` - Checkpoint files for resuming

## Configuration

Key parameters in the code:
- `max_depth`: How many levels deep to traverse (default: 2)
- `max_nodes`: Maximum wallets to process (default: 50,000)
- `rate_limit`: API calls per second (default: 20)
- `checkpoint_interval`: Save progress every N wallets (default: 500)

## Performance Tips

1. **Large Collections**: The tool automatically handles rate limiting and checkpointing
2. **Resume Builds**: If interrupted, the tool will automatically resume from the last checkpoint
3. **API Limits**: Adjust `rate_limit` based on your Alchemy plan

## Error Handling

The tool includes smart error handling that:

1. **Tracks Errors**: Monitors both consecutive and total errors
2. **Auto-Pause**: Automatically pauses and saves checkpoint when:
   - 10 consecutive errors occur
   - 50 total errors accumulate
   - API capacity limit is exceeded (403 error)
3. **Resume Capability**: Always saves progress before pausing, allowing easy resumption
4. **Partial Results**: Even if paused due to errors, generates a graph from collected data

## Visualization

To visualize the generated networks:

1. **Gephi** (recommended):
   - Download from https://gephi.org/
   - Open the `.gexf` file
   - Apply layout algorithms (Force Atlas 2, etc.)
   - Color nodes by degree/centrality

2. **Python** (using the visualize_network.py script):
   ```bash
   python visualize_network.py nft_network_<contract_short>.gexf
   ```

## Troubleshooting

1. **API Key Issues**: Make sure your `.env` file contains `ALCHEMY_API_KEY=your_key`
2. **Rate Limits**: The tool handles rate limits automatically, but you can reduce `rate_limit` if needed
3. **Memory Issues**: For very large collections, consider reducing `max_nodes`
4. **Checkpoint Recovery**: The tool automatically finds and loads the latest checkpoint

## Advanced Usage

### Examples

See `example_usage.py` for interactive examples of both modes:

```bash
python example_usage.py
```

### Filtering Contracts

To exclude certain contracts from analysis, add them to the `SPAM_CONTRACTS` set in the code.

### Custom Parameters

You can modify these parameters in the code:
- `max_consecutive_errors`: Number of consecutive errors before pausing (default: 10)
- `max_total_errors`: Total errors before pausing (default: 50)
- `MIN_SHARED_CONTRACTS`: Minimum shared contracts to create an edge (default: 2)
- `MAX_OWNERS_PER_CONTRACT`: Skip contracts with more owners than this (default: 10,000)
- `MAX_NFTS_PER_WALLET`: Skip wallets with more NFTs than this (default: 1,000)

## License

This tool is provided as-is for educational and research purposes. 