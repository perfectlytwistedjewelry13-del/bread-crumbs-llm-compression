#!/usr/bin/env python3
"""
Coordinate-sensitivity probe for neural network weights (Phase 2).

Tests whether the measured compressibility of a weight tensor changes
when the tensor's data is re-traversed in a different order while
preserving local adjacency within blocks.

Theoretical framing: Paper K (Kolmogorov Resonance, Gardner 2026) argues
that apparent incompressibility is a coordinate-dependent property.
A string that compresses poorly in one traversal order may compress
differently in another. This experiment tests that claim narrowly, on
three tensors from gemma-2-9b, by dequantizing them to fp32 and then
compressing under five different traversal orders.

Experimental design (Team High Five: Susan + Claude + GPT):
- Model: gemma-2-9b (Q4_0)
- Tensors (3):
    1. token_embd.weight       - "ceiling" case, near-maximally dense
    2. blk.20.ffn_down.weight  - representative mid-network dense matrix
    3. blk.37.post_ffw_norm.weight - most compressible norm (outlier peak)
- Transforms (5):
    1. native   - the tensor as stored (post-dequantization)
    2. reversed - flat-array reversed; sanity check
    3. transposed - matrix transpose before flattening; wrong-axis case
    4. random   - uniform random permutation with fixed seed=0; negative control
    5. fibblock - Fibonacci column grouping + reverse group ordering;
                  the Paper K test (preserves local adjacency within blocks,
                  changes mid-range adjacency via non-uniform partitioning)
- Compressor: zstd level 3 (default)
- Output: one CSV with 15 rows

Usage:
    python coordinate_probe_phase2.py \\
      --model "C:\\Users\\izzyz\\.ollama\\models\\blobs\\sha256-ff1d1fc78170d787ee1201778e2dd65ea211654ca5fb7d69b5a2e7b123a50373" \\
      --outdir coordinate_probe_phase2_results

The script will print progress to the console and write a single CSV:
    coordinate_probe_phase2_results/phase2_results.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFReader, dequantize, GGMLQuantizationType
except ImportError:
    print("ERROR: gguf package not installed. Run: pip install gguf", file=sys.stderr)
    sys.exit(2)

try:
    import zstandard as zstd
except ImportError:
    print("ERROR: zstandard package not installed. Run: pip install zstandard", file=sys.stderr)
    sys.exit(2)


TARGET_TENSORS = [
    "token_embd.weight",
    "blk.20.ffn_down.weight",
    "blk.37.post_ffw_norm.weight",
]


def load_and_dequantize(model_path: Path, tensor_name: str) -> tuple[np.ndarray, str, tuple[int, ...]]:
    """
    Open the GGUF file, find the named tensor, dequantize it to fp32,
    and return the result as a numpy array along with metadata.
    """
    reader = GGUFReader(str(model_path))
    for tensor in reader.tensors:
        if str(tensor.name) == tensor_name:
            qtype = tensor.tensor_type
            qtype_name = GGMLQuantizationType(qtype).name if isinstance(qtype, int) else qtype.name
            raw_data = tensor.data
            shape = tuple(int(x) for x in tensor.shape)
            # Dequantize to fp32. If it's already a float type, this is a cast.
            try:
                dequantized = dequantize(raw_data, qtype)
            except Exception as e:
                raise RuntimeError(f"Failed to dequantize {tensor_name} (qtype={qtype_name}): {e}") from e
            # Ensure fp32 for consistent byte representation across tensors
            dequantized = np.asarray(dequantized, dtype=np.float32)
            # Reshape to the GGUF-reported shape.
            # GGUF shapes are in reverse order vs what dequantize outputs in some cases,
            # so we use the total element count as the source of truth.
            if dequantized.size != int(np.prod(shape)):
                # Fall back to native 1D
                print(f"  warning: dequantized size {dequantized.size} != shape product {np.prod(shape)}; "
                      f"using 1D flat array", file=sys.stderr)
            else:
                try:
                    dequantized = dequantized.reshape(shape)
                except ValueError:
                    pass
            return dequantized, qtype_name, shape
    raise KeyError(f"Tensor {tensor_name!r} not found in {model_path}")


# ----------------------------------------------------------------------
# Transforms
# Each transform accepts a 2D (or 1D) numpy array and returns a 1D array
# of the same dtype, representing the bytes that will be compressed.
# ----------------------------------------------------------------------

def transform_native(arr: np.ndarray) -> np.ndarray:
    """The tensor in its dequantized, row-major order."""
    return np.ascontiguousarray(arr).ravel(order="C")


def transform_reversed(arr: np.ndarray) -> np.ndarray:
    """Flat array reversed end-to-end. Sanity check."""
    flat = np.ascontiguousarray(arr).ravel(order="C")
    return flat[::-1].copy()


def transform_transposed(arr: np.ndarray) -> np.ndarray:
    """Matrix transposed before flattening. Tests 'wrong axis' serialization."""
    if arr.ndim < 2:
        # 1D tensor cannot be meaningfully transposed; return native
        return transform_native(arr)
    return np.ascontiguousarray(arr.T).ravel(order="C")


def transform_random(arr: np.ndarray, seed: int = 0) -> np.ndarray:
    """Uniform random permutation with fixed seed. Negative control."""
    flat = np.ascontiguousarray(arr).ravel(order="C")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(flat.size)
    return flat[perm].copy()


def fibonacci_widths_up_to(total: int) -> list[int]:
    """
    Build a list of Fibonacci widths that sum to exactly `total`.
    Uses [1, 2, 3, 5, 8, 13, 21, 34, 55, ...] with the final group
    taking whatever remains.
    """
    if total <= 0:
        return []
    fib = [1, 2]
    while fib[-1] < total:
        nxt = fib[-1] + fib[-2]
        if nxt <= total:
            fib.append(nxt)
        else:
            break
    widths = []
    remaining = total
    for w in fib:
        if remaining <= 0:
            break
        take = min(w, remaining)
        widths.append(take)
        remaining -= take
    if remaining > 0:
        widths.append(remaining)
    return widths


def transform_fibblock(arr: np.ndarray) -> np.ndarray:
    """
    Fibonacci column grouping + reverse group ordering.

    Partitions columns into Fibonacci-sized groups, then reverses the
    order of groups. Within each group, original column order is
    preserved. This preserves local adjacency inside groups while
    changing mid-range adjacency.

    For 1D tensors, treat the array as (1, N) and partition.
    """
    if arr.ndim == 1:
        # Treat 1D as a single row
        work = arr.reshape(1, -1)
    elif arr.ndim == 2:
        work = arr
    else:
        # For higher-dim tensors, flatten all but the last dim
        work = arr.reshape(-1, arr.shape[-1])

    n_cols = work.shape[1]
    widths = fibonacci_widths_up_to(n_cols)

    # Build column group boundaries
    boundaries = []
    start = 0
    for w in widths:
        boundaries.append((start, start + w))
        start += w

    # Reverse the group ordering
    reversed_groups = list(reversed(boundaries))

    # Assemble the reordered matrix by concatenating the groups in reverse order
    pieces = [work[:, s:e] for s, e in reversed_groups]
    reordered = np.concatenate(pieces, axis=1)

    # Flatten row-major
    return np.ascontiguousarray(reordered).ravel(order="C")


TRANSFORMS = {
    "native": transform_native,
    "reversed": transform_reversed,
    "transposed": transform_transposed,
    "random": transform_random,
    "fibblock": transform_fibblock,
}


# ----------------------------------------------------------------------
# Compression
# ----------------------------------------------------------------------

def zstd_compress(data_bytes: bytes, level: int = 3) -> tuple[int, float]:
    """Compress the given bytes with zstd and return (compressed_size, seconds)."""
    cctx = zstd.ZstdCompressor(level=level)
    t0 = time.perf_counter()
    compressed = cctx.compress(data_bytes)
    seconds = time.perf_counter() - t0
    return len(compressed), seconds


# ----------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------

def run_experiment(model_path: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    results_csv = outdir / "phase2_results.csv"

    print(f"Model: {model_path}")
    print(f"File size: {model_path.stat().st_size:,} bytes")
    print(f"Output: {results_csv}")
    print()

    all_rows: list[dict] = []
    run_ts = dt.datetime.now(dt.timezone.utc).isoformat()

    for tensor_name in TARGET_TENSORS:
        print(f"Loading and dequantizing: {tensor_name}")
        try:
            arr, qtype_name, shape = load_and_dequantize(model_path, tensor_name)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            continue

        # Establish the "reference" native bytes for sanity
        native_bytes = transform_native(arr).tobytes()
        native_size = len(native_bytes)
        print(f"  shape: {shape}")
        print(f"  dtype: {arr.dtype}")
        print(f"  qtype (original): {qtype_name}")
        print(f"  dequantized size: {native_size:,} bytes ({native_size/1024/1024:.2f} MB)")

        for transform_name, transform_fn in TRANSFORMS.items():
            try:
                reordered = transform_fn(arr)
            except Exception as e:
                print(f"  ! transform {transform_name} failed: {e}", file=sys.stderr)
                continue

            reordered_bytes = reordered.tobytes()
            if len(reordered_bytes) != native_size:
                print(f"  ! {transform_name}: size mismatch "
                      f"({len(reordered_bytes)} vs {native_size}) — "
                      f"skipping", file=sys.stderr)
                continue

            # Verify the transform is a permutation of the native bytes
            # (by sorting both and comparing). This protects against bugs
            # where a transform accidentally duplicates or drops data.
            # Note: we compare the float arrays, not bytes, because float
            # representations can vary for the same value in theory; here
            # we're just permuting existing floats so sort-equal is valid.
            native_sorted = np.sort(transform_native(arr))
            reordered_sorted = np.sort(reordered)
            if not np.array_equal(native_sorted, reordered_sorted):
                print(f"  ! {transform_name}: NOT a pure permutation of native "
                      f"(value set differs)", file=sys.stderr)

            compressed_size, seconds = zstd_compress(reordered_bytes, level=3)
            ratio = compressed_size / native_size

            # Hash of the reordered bytes so we can verify reproducibility
            sha_hex = hashlib.sha256(reordered_bytes).hexdigest()[:16]

            row = {
                "tensor_name": tensor_name,
                "tensor_shape": "x".join(str(s) for s in shape),
                "tensor_qtype_original": qtype_name,
                "dequantized_bytes": native_size,
                "transform": transform_name,
                "compressed_bytes": compressed_size,
                "ratio": round(ratio, 6),
                "zstd_level": 3,
                "seconds": round(seconds, 4),
                "sha256_prefix_16": sha_hex,
                "run_timestamp": run_ts,
            }
            all_rows.append(row)
            print(f"    {transform_name:<12} ratio={ratio:.4f}  "
                  f"size={compressed_size:>12,}  ({seconds:.3f}s)")

        # Compute and print native-relative deltas for this tensor
        tensor_rows = [r for r in all_rows if r["tensor_name"] == tensor_name]
        native_row = next((r for r in tensor_rows if r["transform"] == "native"), None)
        if native_row is not None:
            print(f"  deltas from native (for this tensor):")
            for r in tensor_rows:
                if r["transform"] != "native":
                    delta = r["ratio"] - native_row["ratio"]
                    print(f"    {r['transform']:<12} delta={delta:+.4f}")
        print()

    # Write CSV
    if all_rows:
        with results_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Wrote {len(all_rows)} rows to {results_csv}")
    else:
        print("No rows to write (all tensors failed to load).", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 coordinate-sensitivity probe.")
    p.add_argument("--model", required=True, help="Path to the .gguf (or Ollama blob) file")
    p.add_argument("--outdir", default="coordinate_probe_phase2_results",
                   help="Directory to write the results CSV")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        print(f"Model file not found: {model_path}", file=sys.stderr)
        return 2

    outdir = Path(args.outdir).expanduser().resolve()
    run_experiment(model_path, outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
