#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def generate_progress(manifest: Path, cache_dir: Path, logs_dir: Path, output: Path) -> None:
    total = sum(1 for _ in manifest.open()) - 1
    cached = len(list(cache_dir.glob("prot_emb_*.pt")))

    progress = {
        "manifest": str(manifest),
        "cache_dir": str(cache_dir),
        "total_genomes": total,
        "cached_files": cached,
        "coverage_pct": round(100 * cached / total, 2) if total else 0.0,
        "shards": {},
    }

    for log_path in sorted(logs_dir.glob("esm2_cache_*.out")):
        m = re.search(r"esm2_cache_(\d+)", log_path.name)
        shard = m.group(1) if m else "unknown"
        processed = set()
        hits = misses = 0
        for line in log_path.open():
            if "Processed" in line:
                hits += int("cache=hit" in line)
                misses += int("cache=miss" in line)
                m2 = re.search(r"Processed (\S+)", line)
                if m2:
                    processed.add(m2.group(1))
        progress["shards"][shard] = {
            "completed": len(processed),
            "hits": hits,
            "misses": misses,
        }

    output.write_text(json.dumps(progress, indent=2))
    print(json.dumps(progress, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate cache population progress JSON")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--logs-dir", type=Path, default=Path("logs"))
    p.add_argument("--output", type=Path, default=Path("cache_population_progress.json"))
    return p


def main() -> None:
    args = build_parser().parse_args()
    generate_progress(args.manifest, args.cache_dir, args.logs_dir, args.output)


if __name__ == "__main__":
    main()

