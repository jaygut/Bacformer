"""Benchmark pipeline latency and cache behavior for the pilot manifest."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from bacformer.pp.embed_prot_seqs import load_plm, compute_genome_protein_embeddings
from bacformer.pp.preprocess import preprocess_genome_assembly
from foodguard.config import PipelineConfig
from foodguard.pipeline import FoodGuardPipeline


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "gbff_path" not in df.columns:
        raise ValueError("Manifest must contain 'gbff_path' column.")
    return df


def preprocess_sequences(gbff_path: Path) -> List[List[str]]:
    pre = preprocess_genome_assembly(str(gbff_path))
    proteins = pre["protein_sequence"]
    if not proteins:
        raise ValueError(f"No proteins extracted from {gbff_path}")
    return proteins


def benchmark_full_pipeline(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    config = PipelineConfig(
        model_path=str(args.model_path),
        device=args.device,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        cache_max_prot_seq_len=args.max_prot_seq_len,
    )
    pipeline = FoodGuardPipeline(config=config)

    records: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        if args.limit and idx >= args.limit:
            break
        gbff_path = Path(row["gbff_path"])
        if not gbff_path.exists():
            logging.error("Skipping missing GBFF: %s", gbff_path)
            continue

        start = time.perf_counter()
        result = pipeline.process_genome(str(gbff_path))
        elapsed = time.perf_counter() - start
        cache_info = result.get("cache") or pipeline.last_cache_info or {}

        record = {
            "genome_id": row.get("genome_id"),
            "species": row.get("species"),
            "path": str(gbff_path),
            "timing_total_sec": elapsed,
            "timing_pipeline_sec": result.get("timing_sec"),
            "cache": cache_info,
        }
        records.append(record)
        logging.info(
            "Genome %s total=%.2fs pipeline=%.2fs cache_hit=%s",
            record["genome_id"],
            elapsed,
            record["timing_pipeline_sec"],
            cache_info.get("hit"),
        )
    return records


def benchmark_embeddings_only(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    config = PipelineConfig(
        use_stub=True,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        cache_max_prot_seq_len=args.max_prot_seq_len,
    )
    pipeline = FoodGuardPipeline(config=config)
    if pipeline._plm is None or pipeline._tok is None:  # pylint: disable=protected-access
        pipeline._plm, pipeline._tok = load_plm(pipeline._plm_model_id, model_type="esm2")  # type: ignore[attr-defined]

    records: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        if args.limit and idx >= args.limit:
            break
        gbff_path = Path(row["gbff_path"])
        if not gbff_path.exists():
            logging.error("Skipping missing GBFF: %s", gbff_path)
            continue

        sequences = preprocess_sequences(gbff_path)
        start = time.perf_counter()
        _, cache_info = pipeline._get_protein_embeddings(  # pylint: disable=protected-access
            protein_sequences=sequences,
            compute_fn=compute_genome_protein_embeddings,
            tokenizer=pipeline._tok,  # pylint: disable=protected-access
            batch_size=args.batch_size,
            max_prot_seq_len=args.max_prot_seq_len,
        )
        elapsed = time.perf_counter() - start

        record = {
            "genome_id": row.get("genome_id"),
            "species": row.get("species"),
            "path": str(gbff_path),
            "timing_total_sec": elapsed,
            "cache": cache_info,
        }
        records.append(record)
        logging.info(
            "Genome %s embedding_time=%.2fs cache_hit=%s",
            record["genome_id"],
            elapsed,
            cache_info.get("hit"),
        )
    return records


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}
    total = len(records)
    total_time = sum(r["timing_total_sec"] for r in records)
    hits = sum(1 for r in records if r.get("cache", {}).get("hit"))
    misses = sum(1 for r in records if r.get("cache", {}).get("hit") is False)
    return {
        "genomes": total,
        "avg_total_sec": total_time / total,
        "cache_hits": hits,
        "cache_misses": misses,
    }


def write_report(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    logging.info("Benchmark report written to %s", path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark FoodGuard pipeline/cache on pilot manifest.")
    parser.add_argument("--manifest", type=Path, required=True, help="Pilot manifest TSV.")
    parser.add_argument("--cache-dir", type=Path, help="Cache directory used by the pipeline.")
    parser.add_argument("--model-path", type=Path, help="Optional Bacformer classification checkpoint.")
    parser.add_argument("--device", type=str, default=None, help="Device override passed to pipeline (e.g., cuda:0).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for embedding-only mode.")
    parser.add_argument("--max-prot-seq-len", type=int, default=1024, help="Max protein length for embeddings.")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N genomes (0 = all).")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSONL report output path.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    manifest_df = load_manifest(args.manifest)

    if args.model_path:
        logging.info("Running full pipeline benchmark with model %s", args.model_path)
        records = benchmark_full_pipeline(manifest_df, args)
    else:
        logging.info("Running embedding-only benchmark (no classifier).")
        records = benchmark_embeddings_only(manifest_df, args)

    summary = summarize(records)
    logging.info("Summary: %s", summary)

    if args.report:
        write_report(records, args.report)


if __name__ == "__main__":
    main()
