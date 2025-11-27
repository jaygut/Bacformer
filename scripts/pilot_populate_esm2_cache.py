"""Populate the ESM-2 protein embedding cache for the pilot manifest."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

from bacformer.pp.preprocess import preprocess_genome_assembly
from bacformer.pp.embed_prot_seqs import load_plm, add_protein_embeddings
from foodguard.utils import cache as cache_utils

DEFAULT_MODEL_ID = "facebook/esm2_t12_35M_UR50D"


def iter_shard(df: pd.DataFrame, shard_index: int, shard_count: int) -> Iterable[Tuple[int, pd.Series]]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive.")
    if not (0 <= shard_index < shard_count):
        raise ValueError("shard_index must satisfy 0 <= index < shard_count.")

    for idx, row in df.reset_index(drop=True).iterrows():
        if idx % shard_count == shard_index:
            yield idx, row


def count_proteins(sequences) -> int:
    if not sequences:
        return 0
    if isinstance(sequences[0], list):
        return sum(len(contig) for contig in sequences)
    return len(sequences)


def populate_cache(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest, sep="\t")
    total_rows = len(manifest)
    logging.info("Loaded manifest with %d genomes", total_rows)

    model, tokenizer = load_plm(args.model_id, model_type="esm2")
    processed = 0
    cache_hits = 0
    cache_misses = 0
    total_seconds = 0.0
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in iter_shard(manifest, args.shard_index, args.shard_count):
        genome_id = row["genome_id"]
        gbff_path = Path(row["gbff_path"])
        if not gbff_path.exists():
            logging.error("GBFF file not found for genome %s: %s", genome_id, gbff_path)
            continue

        pre = preprocess_genome_assembly(str(gbff_path))
        sequences = pre["protein_sequence"]
        contig_names = pre.get("contig_name")

        cache_key = cache_utils.make_key(
            protein_sequences=sequences,
            model_id=args.model_id,
            model_type="esm2",
            max_prot_seq_len=args.max_prot_seq_len,
            genome_pooling_method=None,
        )
        cache_path = cache_dir / f"prot_emb_{cache_key}.pt"
        existed_before = cache_path.exists()

        start = time.perf_counter()
        add_protein_embeddings(
            row={"protein_sequence": sequences, "contig_name": contig_names},
            prot_seq_col="protein_sequence",
            output_col="embeddings",
            model=model,
            tokenizer=tokenizer,
            model_type="esm2",
            batch_size=args.batch_size,
            max_prot_seq_len=args.max_prot_seq_len,
            genome_pooling_method=None,
            cache_dir=str(cache_dir),
            cache_overwrite=args.overwrite,
            model_id=args.model_id,
        )
        elapsed = time.perf_counter() - start
        total_seconds += elapsed

        cache_hit = existed_before and not args.overwrite
        if cache_hit:
            cache_hits += 1
        else:
            cache_misses += 1

        processed += 1
        proteins = count_proteins(sequences)
        logging.info(
            "Processed %s (%d proteins) shard_idx=%d elapsed=%.2fs cache=%s",
            genome_id,
            proteins,
            idx,
            elapsed,
            "hit" if cache_hit else "miss",
        )

        if args.limit and processed >= args.limit:
            logging.info("Limit of %d genomes reached; stopping early.", args.limit)
            break

    if processed == 0:
        logging.warning("No genomes processed for shard %d/%d.", args.shard_index, args.shard_count)
        return

    logging.info(
        "Summary: %d genomes processed (hits=%d, misses=%d) total_time=%.2fs avg_time=%.2fs",
        processed,
        cache_hits,
        cache_misses,
        total_seconds,
        total_seconds / processed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Populate the ESM-2 cache for the pilot manifest.")
    parser.add_argument("--manifest", type=Path, required=True, help="Pilot manifest TSV path.")
    parser.add_argument("--cache-dir", type=Path, required=True, help="Directory to store prot_emb_*.pt files.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for ESM-2 inference.")
    parser.add_argument("--max-prot-seq-len", type=int, default=1024, help="Maximum protein sequence length.")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="ESM-2 model identifier.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute embeddings even if cache exists.")
    parser.add_argument("--shard-index", type=int, default=0, help="Shard index for distributed runs.")
    parser.add_argument("--shard-count", type=int, default=1, help="Total number of shards (workers).")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on genomes processed (0 = no limit).")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    populate_cache(args)


if __name__ == "__main__":
    main()
