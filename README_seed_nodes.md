# Seed Node Functionality

## Overview

The NFT network builder now supports marking "seed" nodes - these are the original wallet addresses that hold NFTs from the specified contract. This feature helps you identify and visualize the original holders versus wallets discovered through network exploration.

## How It Works

### In Contract Mode

When you run the network builder with a contract address:

```bash
python nft_network_builder.py --contract 0xb47e3cd837ddf8e4c57f05d70ab865de6e193bbb
```

1. All holders of the specified contract are fetched
2. These holders are marked as "seed" nodes with attributes:
   - `seed=True` (boolean flag)
   - `label="Seed"` (text label)
3. The graph explores connections from these seed nodes
4. Seed node information is preserved in checkpoints

### In Wallet Mode

Seed nodes are not used in wallet mode since you're starting from specific wallets rather than contract holders.

## Visualization

### Using the Built-in Visualizer

```bash
python visualize_seed_nodes.py nft_network_b47e3cd8.gexf -o network_viz.png
```

This creates a visualization where:
- **Red nodes** = Seed wallets (original contract holders)
- **Blue nodes** = Discovered wallets (found through exploration)
- Seed nodes are larger and labeled

### In Gephi or Other Tools

The seed node attributes are preserved in GEXF and GraphML formats. You can:
- Filter nodes by the `seed` attribute
- Apply different colors/sizes based on `seed=True/false`
- Use the `label="Seed"` attribute for text displays

### In React/Web Visualizations

The React format includes seed attributes in the node data:

```json
{
  "nodes": [
    {
      "id": "0x123...",
      "name": "0x123...",
      "val": 5,
      "seed": true,
      "label": "Seed"
    }
  ]
}
```

## Checkpoint Support

Seed wallet information is automatically saved to and restored from checkpoints, so you don't lose this information when resuming interrupted graph builds.

## Statistics

The network builder and checkpoint converter now report seed node statistics:
- Total number of seed nodes
- Percentage of nodes that are seeds
- Top seed nodes by degree (most connected)

## Example Usage

```bash
# Build network for Bored Apes with seed marking
python nft_network_builder.py --contract 0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D

# Convert checkpoint preserving seed info
python checkpoint_to_graph.py --analyze

# Visualize with seed nodes highlighted
python visualize_seed_nodes.py nft_network_bc4ca0ed.gexf -o bored_apes_network.png
```

## Benefits

1. **Origin Tracking**: Easily identify which wallets are original holders vs discovered through connections
2. **Community Analysis**: Understand how tightly connected the original holder community is
3. **Influence Mapping**: See which original holders are most connected to the broader network
4. **Filtering**: Focus analysis on just seed nodes or just discovered nodes 