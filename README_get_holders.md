# Get Contract Holders Script

A simple script to fetch and output all holders of an NFT contract using the Alchemy API.

## Prerequisites

1. Python 3.7+
2. An Alchemy API key (get one free at https://www.alchemy.com/)
3. Required packages (already in your requirements.txt):
   - aiohttp
   - python-dotenv
   - All dependencies from nft_network_builder.py

## Setup

1. Create a `.env` file in the project directory:
   ```
   ALCHEMY_API_KEY=your_actual_api_key_here
   ```

2. Make sure you have all dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage (Console Output)

```bash
python get_contract_holders.py <contract_address>
```

This will display the first 10 and last 10 holders in the console, along with statistics.

### Save to File

```bash
python get_contract_holders.py <contract_address> --output <filename>
```

The script supports multiple output formats based on the file extension:

- **JSON** (`.json`): Full data including metadata and statistics
- **Text** (`.txt`): Plain text list with header comments
- **CSV** (`.csv`): Simple CSV format with wallet addresses
- **Other**: Plain text list of addresses (one per line)

## Examples

### Example 1: CryptoPunks (Console Output)
```bash
python get_contract_holders.py 0xb47e3cd837ddf8e4c57f05d70ab865de6e193bbb
```

### Example 2: Save to JSON
```bash
python get_contract_holders.py 0xb47e3cd837ddf8e4c57f05d70ab865de6e193bbb --output cryptopunks_holders.json
```

### Example 3: Save to CSV
```bash
python get_contract_holders.py 0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d --output bayc_holders.csv
```

### Example 4: Save to Text
```bash
python get_contract_holders.py 0x23581767a106ae21c074b2276d25e5c3e136a68b --output moonbirds_holders.txt
```

## Output Formats

### JSON Format
```json
{
  "contract_address": "0xb47e3cd837ddf8e4c57f05d70ab865de6e193bbb",
  "total_holders": 3542,
  "timestamp": "2024-01-15T10:30:45.123456",
  "time_taken_seconds": 45.67,
  "api_calls": 36,
  "holders": [
    "0x0000000000000000000000000000000000000000",
    "0x0001234567890abcdef1234567890abcdef12345",
    ...
  ]
}
```

### Text Format
```
# Holders for contract 0xb47e3cd837ddf8e4c57f05d70ab865de6e193bbb
# Total: 3542
# Generated: 2024-01-15T10:30:45.123456

0x0000000000000000000000000000000000000000
0x0001234567890abcdef1234567890abcdef12345
...
```

### CSV Format
```csv
wallet_address
0x0000000000000000000000000000000000000000
0x0001234567890abcdef1234567890abcdef12345
...
```

## Performance Notes

- The script uses rate limiting to respect API limits
- Large collections may take several minutes to fetch all holders
- Progress is shown in real-time with logging
- The script handles pagination automatically
- API calls are optimized to minimize requests

## Error Handling

- The script validates contract addresses before making API calls
- Handles rate limiting gracefully with automatic retries
- Provides clear error messages for missing API keys or network issues
- Saves partial results if interrupted (when using file output)

## Tips

- For large collections (>10k holders), always use the `--output` option to save results
- JSON format is best for programmatic use or further analysis
- CSV format works well for importing into spreadsheets
- Text format is good for simple lists or grep operations 