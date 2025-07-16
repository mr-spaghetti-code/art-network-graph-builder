# Checkpoint to Graph Converter

This tool converts NFT network checkpoints (saved by `gen_network.py`) into NetworkX graph files that can be visualized and analyzed.

## Features

- Load checkpoint files from interrupted or completed network building sessions
- Rebuild the graph structure with proper edge weights based on shared NFT contracts
- Export to multiple graph formats (GEXF, GraphML, GML, JSON)
- Analyze graph statistics (nodes, edges, connectivity, etc.)
- Interactive checkpoint selection

## Usage

### Basic Command Line Usage

```bash
# Convert the most recent checkpoint to GEXF and GraphML formats
python checkpoint_to_graph.py

# Specify a particular checkpoint file
python checkpoint_to_graph.py --checkpoint-file ./checkpoints/checkpoint_1234567890.json

# Change output filename and formats
python checkpoint_to_graph.py --output my_nft_graph --formats gexf graphml json

# Include graph analysis
python checkpoint_to_graph.py --analyze

python checkpoint_to_graph.py --stream-large-files --filter-discovered-only --output "5000"
```

### Command Line Options

- `--checkpoint-dir`: Directory containing checkpoint files (default: `./checkpoints`)
- `--checkpoint-file`: Specific checkpoint file to use (bypasses selection menu)
- `--output`: Base filename for output files (default: `nft_network_from_checkpoint`)
- `--formats`: Output formats to generate (default: `gexf graphml`)
  - Available formats: `gexf`, `graphml`, `gml`, `json`
- `--analyze`: Print detailed graph statistics

### Python API Usage

```python
from checkpoint_to_graph import CheckpointToGraph

# Initialize converter
converter = CheckpointToGraph("./checkpoints")

# List available checkpoints
checkpoints = converter.list_checkpoints()

# Load most recent checkpoint
checkpoint_data = converter.load_checkpoint(checkpoints[0]['path'])

# Build graph
graph = converter.build_graph_from_checkpoint(checkpoint_data)

# Analyze graph (optional)
stats = converter.analyze_graph(graph)

# Save graph
converter.save_graph(graph, "output_filename", formats=["gexf", "graphml"])
```

## Output Formats

### GEXF (Graph Exchange XML Format)
- Best for Gephi visualization
- Preserves all node and edge attributes
- Supports dynamic graphs

### GraphML
- XML-based format
- Good compatibility with many tools
- Preserves attributes

### GML (Graph Modelling Language)
- Simple text format
- Compatible with older tools

### JSON (Node-Link Format)
- JavaScript-friendly
- Good for web visualizations (D3.js, etc.)
- Preserves all data

## Graph Structure

The generated graph contains:

### Nodes (Wallets)
- Node ID: Ethereum wallet address
- Attributes:
  - `num_contracts`: Number of NFT contracts owned

### Edges (Connections)
- Connect wallets that own NFTs from the same contracts
- Attributes:
  - `weight`: Number of shared contracts between the two wallets

## Examples

See `checkpoint_to_graph_example.py` for usage examples.

## Visualization

The generated graph files can be visualized using:
- **Gephi**: Best for GEXF files, powerful layout algorithms
- **Cytoscape**: Works well with GraphML
- **NetworkX + Matplotlib**: For programmatic visualization
- **D3.js**: For web-based interactive visualizations (use JSON format) 