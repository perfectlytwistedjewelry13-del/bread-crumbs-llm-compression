#!/usr/bin/env python3
"""
Bread Crumbs v3 — Super Simple Version for Asha
Just drag-and-drop or type the path to your GGUF file.
"""

import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from tqdm import tqdm
import zstandard as zstd
import brotli

def compress_bytes(data: bytes):
    # Try both compressors and average the ratio
    z_size = len(zstd.ZstdCompressor(level=3).compress(data))
    b_size = len(brotli.compress(data, quality=11))
    return (len(data) / z_size + len(data) / b_size) / 2

def run_breadcrumbs(model_path_str: str):
    path = Path(model_path_str).expanduser().resolve()
    if not path.exists():
        print(f"❌ File not found: {path}")
        return

    print(f"🍞 Starting Bread Crumbs on: {path.name}")
    print(f"   Size: {path.stat().st_size / 1_000_000_000:.2f} GB\n")

    chunk_size = 1 * 1024 * 1024   # 1 MB chunks (safe for your laptop)
    chunks = []
    total_size = path.stat().st_size

    with open(path, "rb") as f:
        idx = 0
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Scanning") as pbar:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                ratio = compress_bytes(data)
                chunks.append({"chunk": idx, "ratio": round(ratio, 4)})
                idx += 1
                pbar.update(len(data))

    # Save results
    out_base = path.with_suffix("")
    with open(f"{out_base}_breadcrumbs.json", "w") as f:
        json.dump(chunks, f, indent=2)

    # Quick plot
    ratios = [c["ratio"] for c in chunks]
    plt.figure(figsize=(12, 5))
    plt.plot(ratios, color="teal", linewidth=1.5)
    plt.title(f"Bread Crumbs — {path.name}")
    plt.xlabel("Chunk Index (1 MB each)")
    plt.ylabel("Compression Ratio (higher = more structure)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_base}_heatmap.png", dpi=150)
    plt.show()

    print(f"\n✅ DONE! Files saved:")
    print(f"   {out_base}_breadcrumbs.json")
    print(f"   {out_base}_heatmap.png")
    print("\nNow open the .json file and look for chunks with unusually high or low ratios.")
    print("Those are your 'needles' — the interesting parts of the model!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_breadcrumbs(sys.argv[1])
    else:
        path = input("\nPaste the full path to your GGUF file here and press Enter:\n> ")
        run_breadcrumbs(path)