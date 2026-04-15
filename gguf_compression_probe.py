#!/usr/bin/env python3
"""
Compression probe for GGUF model files (Windows-friendly, Python-lib edition).

Based on GPT's starter script, modified for Windows by using Python compression
libraries instead of external CLI binaries. Same chunking schemes, same CSV schema,
same output structure.

Stage-1 friendly:
- fixed-size chunking
- Fibonacci chunking
- per-tensor GGUF chunking (requires Python package `gguf`)
- compressors via Python libraries: zstandard, gzip, lzma, brotli, bz2
- detailed CSV + summary CSV per run
- GGUF metadata extraction (architecture, quantization) from the file itself
- optional smoke test mode (--max-chunks N) to verify pipeline before full run

Intended use (smoke test first):
    python gguf_compression_probe.py ^
      --model "C:\\Users\\izzyz\\.ollama\\models\\blobs\\sha256-ff1d1fc78170d787ee1201778e2dd65ea211654ca5fb7d69b5a2e7b123a50373" ^
      --model-name gemma-2 ^
      --chunking fixed fibonacci per_tensor ^
      --compressors zstd ^
      --max-chunks 50 ^
      --outdir probe_results_smoke

Then full Stage 1 (same command without --max-chunks, different outdir):
    python gguf_compression_probe.py ^
      --model "C:\\Users\\izzyz\\.ollama\\models\\blobs\\sha256-ff1d1..." ^
      --model-name gemma-2 ^
      --chunking fixed fibonacci per_tensor ^
      --compressors zstd ^
      --outdir probe_results_stage1
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Chunk:
    chunk_id: int
    chunk_start: int
    chunk_end: int
    chunk_type: str
    chunk_label: str

    @property
    def raw_size(self) -> int:
        return self.chunk_end - self.chunk_start


@dataclass
class RunConfig:
    model_name: str
    file_format: str
    quantization: str
    compressor: str
    level: Optional[int]
    chunking_scheme: str
    chunk_param: str
    run_timestamp: str


# Compressor registry: name -> (callable(bytes, level) -> bytes, default_level)
def _compress_zstd(data: bytes, level: Optional[int]) -> bytes:
    import zstandard as zstd
    lvl = level if level is not None else 3
    cctx = zstd.ZstdCompressor(level=lvl)
    return cctx.compress(data)


def _compress_gzip(data: bytes, level: Optional[int]) -> bytes:
    import gzip
    lvl = level if level is not None else 6
    return gzip.compress(data, compresslevel=lvl)


def _compress_xz(data: bytes, level: Optional[int]) -> bytes:
    import lzma
    # lzma preset is 0-9; default 6
    preset = level if level is not None else 6
    return lzma.compress(data, preset=preset)


def _compress_brotli(data: bytes, level: Optional[int]) -> bytes:
    import brotli
    # brotli quality is 0-11; default 11
    q = level if level is not None else 11
    return brotli.compress(data, quality=q)


def _compress_bzip2(data: bytes, level: Optional[int]) -> bytes:
    import bz2
    # bz2 compresslevel is 1-9; default 9
    lvl = level if level is not None else 9
    return bz2.compress(data, compresslevel=lvl)


COMPRESSORS = {
    "zstd": _compress_zstd,
    "gzip": _compress_gzip,
    "xz": _compress_xz,
    "brotli": _compress_brotli,
    "bzip2": _compress_bzip2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe GGUF compressibility patterns by chunk.")
    parser.add_argument("--model", required=True, help="Path to the .gguf (or Ollama blob) file")
    parser.add_argument("--model-name", default="unknown_model")
    parser.add_argument("--file-format", default="gguf")
    parser.add_argument("--quantization", default="auto",
                        help="Quantization label. 'auto' reads from GGUF metadata.")
    parser.add_argument("--outdir", default="probe_results")
    parser.add_argument("--chunking", nargs="+", default=["fixed", "fibonacci", "per_tensor"],
                        choices=["fixed", "fibonacci", "per_tensor"])
    parser.add_argument("--compressors", nargs="+", default=["zstd"],
                        choices=list(COMPRESSORS.keys()))
    parser.add_argument("--fixed-size-mb", type=int, default=1)
    parser.add_argument("--fib-min-kb", type=int, default=256)
    parser.add_argument("--fib-max-mb", type=int, default=8)
    parser.add_argument("--min-chunk-bytes", type=int, default=4096)
    parser.add_argument("--max-chunks", type=int, default=0,
                        help="Safety cap / smoke test mode. 0 means unlimited.")
    parser.add_argument("--levels", nargs="*", default=[],
                        help="Optional compressor levels: compressor=level, e.g. zstd=3 xz=6")
    return parser.parse_args()


def load_levels(level_args: list[str]) -> dict[str, int]:
    levels: dict[str, int] = {}
    for item in level_args:
        if "=" not in item:
            raise ValueError(f"Bad --levels entry: {item!r}. Use compressor=level.")
        k, v = item.split("=", 1)
        levels[k.strip()] = int(v.strip())
    return levels


def file_sha256(path: Path, limit_mb: int = 16) -> str:
    h = hashlib.sha256()
    limit = limit_mb * 1024 * 1024
    with path.open("rb") as f:
        remaining = limit
        while remaining > 0:
            data = f.read(min(1024 * 1024, remaining))
            if not data:
                break
            h.update(data)
            remaining -= len(data)
    return h.hexdigest()


def read_gguf_metadata(model_path: Path) -> dict:
    """Extract useful metadata from the GGUF file itself."""
    try:
        from gguf import GGUFReader
    except Exception as e:
        return {"error": f"gguf package not available: {e}"}

    try:
        reader = GGUFReader(str(model_path))
    except Exception as e:
        return {"error": f"GGUFReader failed: {e}"}

    meta = {}
    # Extract common metadata fields if present.
    # In gguf-py, a ReaderField's .parts is a list of numpy arrays covering
    # the whole key+value+type region. The .data attribute is a list of
    # indices into .parts that point at the *value* portion only.
    interesting_keys = [
        "general.architecture",
        "general.name",
        "general.file_type",
        "general.quantization_version",
        "general.size_label",
        "general.basename",
        "general.finetune",
    ]

    def field_value(field) -> str:
        """Extract a human-readable value from a GGUFReader field."""
        try:
            data_indices = list(field.data) if hasattr(field, "data") else []
            if not data_indices:
                return "<no-value>"
            value_parts = [field.parts[i] for i in data_indices]
            # String fields: one or more byte arrays -> decode as utf-8
            # Numeric fields: one or more numpy scalar arrays -> stringify
            pieces = []
            for part in value_parts:
                try:
                    # Try bytes-decode first (string fields)
                    as_bytes = bytes(part)
                    # If it decodes cleanly and has printable chars, treat as string
                    decoded = as_bytes.decode("utf-8", errors="strict")
                    if decoded and all(c.isprintable() or c.isspace() for c in decoded):
                        pieces.append(decoded)
                        continue
                except Exception:
                    pass
                # Otherwise treat as numeric scalar
                try:
                    if hasattr(part, "tolist"):
                        val = part.tolist()
                        if isinstance(val, list) and len(val) == 1:
                            val = val[0]
                        pieces.append(str(val))
                    else:
                        pieces.append(str(part))
                except Exception:
                    pieces.append("<unreadable>")
            return " ".join(pieces).strip()[:200] or "<empty>"
        except Exception as e:
            return f"<error: {e}>"

    try:
        for field in reader.fields.values():
            key = field.name
            if key in interesting_keys:
                meta[key] = field_value(field)
    except Exception as e:
        meta["metadata_error"] = str(e)

    # GGUF file_type is an enum; map the number to a human-readable name
    # Source: https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/gguf/constants.py
    FILE_TYPE_NAMES = {
        0: "ALL_F32", 1: "MOSTLY_F16", 2: "MOSTLY_Q4_0", 3: "MOSTLY_Q4_1",
        4: "MOSTLY_Q4_1_SOME_F16", 7: "MOSTLY_Q8_0", 8: "MOSTLY_Q5_0",
        9: "MOSTLY_Q5_1", 10: "MOSTLY_Q2_K", 11: "MOSTLY_Q3_K_S",
        12: "MOSTLY_Q3_K_M", 13: "MOSTLY_Q3_K_L", 14: "MOSTLY_Q4_K_S",
        15: "MOSTLY_Q4_K_M", 16: "MOSTLY_Q5_K_S", 17: "MOSTLY_Q5_K_M",
        18: "MOSTLY_Q6_K", 19: "MOSTLY_IQ2_XXS", 20: "MOSTLY_IQ2_XS",
        21: "MOSTLY_Q2_K_S", 22: "MOSTLY_IQ3_XS", 23: "MOSTLY_IQ3_XXS",
        24: "MOSTLY_IQ1_S", 25: "MOSTLY_IQ4_NL", 26: "MOSTLY_IQ3_S",
        27: "MOSTLY_IQ3_M", 28: "MOSTLY_IQ2_S", 29: "MOSTLY_IQ2_M",
        30: "MOSTLY_IQ4_XS", 31: "MOSTLY_IQ1_M", 32: "MOSTLY_BF16",
    }
    ft_raw = meta.get("general.file_type", "")
    try:
        ft_num = int(ft_raw)
        meta["file_type_name"] = FILE_TYPE_NAMES.get(ft_num, f"unknown({ft_num})")
    except (ValueError, TypeError):
        meta["file_type_name"] = "unknown"

    try:
        meta["tensor_count"] = len(reader.tensors)
    except Exception:
        meta["tensor_count"] = -1

    return meta


def fixed_chunks(file_size: int, size_bytes: int, max_chunks: int = 0) -> list[Chunk]:
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < file_size:
        end = min(start + size_bytes, file_size)
        chunks.append(Chunk(idx, start, end, "fixed", f"fixed_{size_bytes}"))
        idx += 1
        start = end
        if max_chunks and idx >= max_chunks:
            break
    return chunks


def fibonacci_sizes(min_bytes: int, max_bytes: int) -> list[int]:
    seq = [1, 1]
    while seq[-1] < max_bytes:
        seq.append(seq[-1] + seq[-2])
    sizes = sorted({n for n in seq if min_bytes <= n <= max_bytes})
    if not sizes:
        sizes = [min_bytes]
    return sizes


def fibonacci_chunks(file_size: int, min_bytes: int, max_bytes: int, max_chunks: int = 0) -> list[Chunk]:
    sizes = fibonacci_sizes(min_bytes, max_bytes)
    chunks: list[Chunk] = []
    idx = 0
    start = 0
    sidx = 0
    while start < file_size:
        size = sizes[sidx % len(sizes)]
        end = min(start + size, file_size)
        chunks.append(Chunk(idx, start, end, "fibonacci", f"fib_{size}"))
        idx += 1
        start = end
        sidx += 1
        if max_chunks and idx >= max_chunks:
            break
    return chunks


def per_tensor_chunks(model_path: Path, min_chunk_bytes: int, max_chunks: int = 0) -> list[Chunk]:
    try:
        from gguf import GGUFReader
    except Exception as e:
        raise RuntimeError(
            "Per-tensor chunking requires the Python package 'gguf'. "
            "Install with: pip install gguf"
        ) from e

    reader = GGUFReader(str(model_path))
    chunks: list[Chunk] = []
    idx = 0
    for tensor in reader.tensors:
        offset = int(tensor.data_offset)
        n_bytes = int(tensor.n_bytes)
        if n_bytes < min_chunk_bytes:
            continue
        chunks.append(
            Chunk(
                chunk_id=idx,
                chunk_start=offset,
                chunk_end=offset + n_bytes,
                chunk_type="per_tensor",
                chunk_label=str(tensor.name),
            )
        )
        idx += 1
        if max_chunks and idx >= max_chunks:
            break
    if not chunks:
        raise RuntimeError("No per-tensor chunks were found. Check GGUF reader compatibility.")
    return chunks


def build_chunks(args: argparse.Namespace, model_path: Path, file_size: int) -> dict[str, list[Chunk]]:
    built: dict[str, list[Chunk]] = {}
    if "fixed" in args.chunking:
        sz = args.fixed_size_mb * 1024 * 1024
        built["fixed"] = fixed_chunks(file_size, sz, args.max_chunks)
    if "fibonacci" in args.chunking:
        built["fibonacci"] = fibonacci_chunks(
            file_size,
            args.fib_min_kb * 1024,
            args.fib_max_mb * 1024 * 1024,
            args.max_chunks,
        )
    if "per_tensor" in args.chunking:
        built["per_tensor"] = per_tensor_chunks(model_path, args.min_chunk_bytes, args.max_chunks)
    return built


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict], config: RunConfig, model_path: Path) -> dict:
    ratios = [r["ratio"] for r in rows]
    raw_total = sum(r["raw_size"] for r in rows)
    compressed_total = sum(r["compressed_size"] for r in rows)
    seconds_total = sum(r["seconds"] for r in rows)
    return {
        "model_name": config.model_name,
        "model_path": str(model_path),
        "file_format": config.file_format,
        "quantization": config.quantization,
        "chunking_scheme": config.chunking_scheme,
        "chunk_param": config.chunk_param,
        "compressor": config.compressor,
        "level": config.level if config.level is not None else "default",
        "run_timestamp": config.run_timestamp,
        "file_size": model_path.stat().st_size,
        "num_chunks": len(rows),
        "total_raw_size": raw_total,
        "total_compressed_size": compressed_total,
        "overall_ratio": (compressed_total / raw_total) if raw_total else 0.0,
        "avg_chunk_ratio": (sum(ratios) / len(ratios)) if ratios else 0.0,
        "min_chunk_ratio": min(ratios) if ratios else 0.0,
        "max_chunk_ratio": max(ratios) if ratios else 0.0,
        "total_seconds": round(seconds_total, 4),
    }


def probe_run(
    model_path: Path,
    chunks: list[Chunk],
    config: RunConfig,
    outdir: Path,
) -> tuple[Path, Path]:
    detailed_rows: list[dict] = []
    compress_fn = COMPRESSORS[config.compressor]

    run_slug = f"{config.chunking_scheme}__{config.compressor}"
    if config.level is not None:
        run_slug += f"_lvl{config.level}"
    detailed_csv = outdir / f"{run_slug}_detailed.csv"
    summary_csv = outdir / f"{run_slug}_summary.csv"

    total = len(chunks)
    progress_every = max(1, total // 20)  # ~20 progress ticks per run

    with model_path.open("rb") as f:
        for i, chunk in enumerate(chunks):
            f.seek(chunk.chunk_start)
            data = f.read(chunk.raw_size)

            start_t = time.perf_counter()
            try:
                compressed = compress_fn(data, config.level)
                compressed_size = len(compressed)
            except Exception as e:
                print(f"  ! compression error on chunk {chunk.chunk_id}: {e}", file=sys.stderr)
                compressed_size = -1
            seconds = time.perf_counter() - start_t

            row = {
                "model_name": config.model_name,
                "model_path": str(model_path),
                "file_format": config.file_format,
                "quantization": config.quantization,
                "chunk_id": chunk.chunk_id,
                "chunk_start": chunk.chunk_start,
                "chunk_end": chunk.chunk_end,
                "chunk_type": chunk.chunk_type,
                "chunk_label": chunk.chunk_label,
                "raw_size": chunk.raw_size,
                "compressed_size": compressed_size,
                "ratio": (compressed_size / chunk.raw_size) if chunk.raw_size and compressed_size >= 0 else 0.0,
                "compressor": config.compressor,
                "level": config.level if config.level is not None else "default",
                "seconds": round(seconds, 6),
                "chunking_scheme": config.chunking_scheme,
                "chunk_param": config.chunk_param,
                "run_timestamp": config.run_timestamp,
            }
            detailed_rows.append(row)

            if (i + 1) % progress_every == 0 or (i + 1) == total:
                pct = (i + 1) * 100 // total
                print(f"    {config.chunking_scheme}/{config.compressor}: "
                      f"{i+1}/{total} chunks ({pct}%)", flush=True)

    write_csv(detailed_csv, detailed_rows)
    write_csv(summary_csv, [summarize_rows(detailed_rows, config, model_path)])
    return detailed_csv, summary_csv


def main() -> int:
    args = parse_args()
    levels = load_levels(args.levels)

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        print(f"Model file not found: {model_path}", file=sys.stderr)
        return 2

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    file_size = model_path.stat().st_size
    print(f"Model file: {model_path}")
    print(f"File size:  {file_size:,} bytes ({file_size / (1024**3):.2f} GB)")

    # Read GGUF metadata
    print("Reading GGUF metadata...")
    gguf_meta = read_gguf_metadata(model_path)
    for k, v in gguf_meta.items():
        print(f"  {k}: {v}")

    # Resolve quantization from metadata if requested
    quantization = args.quantization
    if quantization == "auto":
        quantization = (
            gguf_meta.get("file_type_name")
            or gguf_meta.get("general.size_label")
            or "unknown"
        )
    print(f"Using quantization label: {quantization}")

    # Build chunk sets
    print("Building chunk sets...")
    chunk_sets = build_chunks(args, model_path, file_size)
    for name, chunks in chunk_sets.items():
        coverage = sum(c.raw_size for c in chunks)
        print(f"  {name}: {len(chunks)} chunks, covering {coverage:,} bytes "
              f"({100 * coverage / file_size:.1f}% of file)")

    # Write manifest
    manifest = {
        "model_name": args.model_name,
        "model_path": str(model_path),
        "file_format": args.file_format,
        "quantization": quantization,
        "file_size": file_size,
        "sha256_prefix_16mb": file_sha256(model_path),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "chunking_requested": "|".join(args.chunking),
        "compressors_requested": "|".join(args.compressors),
        "max_chunks": args.max_chunks,
        "gguf_architecture": gguf_meta.get("general.architecture", "unknown"),
        "gguf_name": gguf_meta.get("general.name", "unknown"),
        "gguf_tensor_count": gguf_meta.get("tensor_count", -1),
    }
    write_csv(outdir / "manifest.csv", [manifest])

    # Run the probe grid
    summary_index_rows: list[dict] = []
    total_runs = len(chunk_sets) * len(args.compressors)
    run_num = 0

    for chunking_name, chunks in chunk_sets.items():
        if chunking_name == "fixed":
            chunk_param = f"{args.fixed_size_mb}MB"
        elif chunking_name == "fibonacci":
            chunk_param = f"min={args.fib_min_kb}KB|max={args.fib_max_mb}MB"
        else:
            chunk_param = "gguf_tensor_boundaries"

        for compressor in args.compressors:
            run_num += 1
            level = levels.get(compressor)
            config = RunConfig(
                model_name=args.model_name,
                file_format=args.file_format,
                quantization=quantization,
                compressor=compressor,
                level=level,
                chunking_scheme=chunking_name,
                chunk_param=chunk_param,
                run_timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
            )
            print(f"\n[{run_num}/{total_runs}] {compressor} on {chunking_name} "
                  f"({len(chunks)} chunks)...")
            t0 = time.perf_counter()
            detailed_csv, summary_csv = probe_run(
                model_path=model_path,
                chunks=chunks,
                config=config,
                outdir=outdir,
            )
            elapsed = time.perf_counter() - t0
            print(f"  done in {elapsed:.1f}s -> {summary_csv.name}")
            summary_index_rows.append(
                {
                    "chunking_scheme": chunking_name,
                    "compressor": compressor,
                    "level": level if level is not None else "default",
                    "detailed_csv": str(detailed_csv),
                    "summary_csv": str(summary_csv),
                    "num_chunks": len(chunks),
                    "wall_seconds": round(elapsed, 2),
                }
            )

    write_csv(outdir / "run_index.csv", summary_index_rows)
    print(f"\nDone. Results written to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
