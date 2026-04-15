Provide a low-cost structural screening tool for model weights before applying expensive interpretability methods.

# Bread Crumbs

A passive compression probe for structural mapping in large language model weight files.

📄 Paper: https://doi.org/10.5281/zenodo.19582324

## Overview

Bread Crumbs is a CPU-only method for probing structural regularity in LLM weight files using lossless compression.

By scanning model binaries in chunks and measuring compression ratios, it produces a coarse structural map without loading the model into memory or running inference.

## Key Idea

Compression ratio acts as a proxy for local statistical regularity.

- High ratio → dense / structured regions  
- Low ratio → less regular / normalization regions  

## Scripts

### Stage 1 — Structural Scan
`gguf_compression_probe.py`

- Fixed chunking  
- Fibonacci chunking  
- Per-tensor chunking (GGUF)  
- Outputs CSV summaries  

### Stage 2 — Coordinate Sensitivity
`coordinate_probe_phase2.py`

Tests how compression changes under:
- native
- reversed
- transposed
- random
- fibblock

### Simple Viewer
`breadcrumbs_v3.py`

- Easy entry point  
- Generates heatmap + JSON

# Install dependencies (once)
pip install numpy pandas pygguf zstandard matplotlib seaborn tqdm

# Run a quick scan
python gguf_compression_probe.py --model models/gemma-2-9b.gguf --chunk-size 1048576

## Usage

Run Stage 1:

```bash
python gguf_compression_probe.py --model path/to/model.gguf

Run Stage 2:

python coordinate_probe_phase2.py --model path/to/model.gguf
