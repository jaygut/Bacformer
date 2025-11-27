"""Generate a pilot-scale manifest for FoodGuard MVP experiments.

This script samples a small, stratified subset of genomes from the
production manifest so engineers can iterate quickly on cache
infrastructure, training loops, and calibration.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

DEFAULT_PATHOGENIC: List[Tuple[str, int]] = [
    ("Salmonella_enterica", 20),
    ("E_coli_O157H7", 20),
    ("L_monocytogenes", 20),
]

DEFAULT_NON_PATHOGENIC: List[Tuple[str, int]] = [
    ("E_coli_nonpathogenic", 20),
    ("B_subtilis", 20),
    ("C_koseri", 20),
]


def parse_species_arg(entries: Iterable[str]) -> List[Tuple[str, int]]:
    result: List[Tuple[str, int]] = []
    for raw in entries:
        try:
            name, count_str = raw.split(":", maxsplit=1)
            count = int(count_str)
            if count <= 0:
                raise ValueError
        except ValueError as exc:  # pragma: no cover - argparse enforces format, but guard anyway
            raise argparse.ArgumentTypeError(f"Invalid species spec '{raw}'. Use format Name:Count.") from exc
        result.append((name, count))
    return result


def sample_species(
    df: pd.DataFrame,
    targets: List[Tuple[str, int]],
    *,
    seed: int,
    label: str,
) -> pd.DataFrame:
    samples = []
    for idx, (species, count) in enumerate(targets):
        subset = df[df["species"] == species]
        if subset.empty:
            logging.warning("Species '%s' not found in manifest for %s class.", species, label)
            continue

        actual_count = min(count, len(subset))
        if actual_count < count:
            logging.warning(
                "Requested %d genomes for %s (%s), but only %d available. Using all.",
                count,
                species,
                label,
                actual_count,
            )
        sampled = subset.sample(n=actual_count, random_state=seed + idx, replace=False)
        samples.append(sampled)
    if not samples:
        raise RuntimeError(f"No genomes selected for {label}; aborting.")
    return pd.concat(samples, ignore_index=True)


def build_pilot_manifest(
    manifest_path: Path,
    output_path: Path,
    *,
    seed: int,
    pathogenic_targets: List[Tuple[str, int]],
    non_pathogenic_targets: List[Tuple[str, int]],
) -> None:
    df = pd.read_csv(manifest_path, sep="\t")
    required_cols = {"genome_id", "gbff_path", "is_pathogenic", "species"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Manifest {manifest_path} missing required columns: {sorted(missing)}")

    pathogenic_df = sample_species(df[df["is_pathogenic"] == 1], pathogenic_targets, seed=seed, label="pathogenic")
    non_pathogenic_df = sample_species(
        df[df["is_pathogenic"] == 0], non_pathogenic_targets, seed=seed + 1000, label="non-pathogenic"
    )

    pilot_df = (
        pd.concat([pathogenic_df, non_pathogenic_df], ignore_index=True)
        .drop_duplicates(subset="genome_id")
        .sort_values(by=["is_pathogenic", "species", "genome_id"])
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pilot_df.to_csv(output_path, sep="\t", index=False)

    logging.info("Pilot manifest written to %s", output_path)
    logging.info(
        "Breakdown:\n%s",
        pilot_df.groupby(["is_pathogenic", "species"])["genome_id"].count(),
    )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a pilot-scale manifest for FoodGuard experiments.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to full manifest TSV.")
    parser.add_argument("--output", type=Path, required=True, help="Output path for pilot manifest TSV.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic sampling.")
    parser.add_argument(
        "--pathogenic-species",
        type=str,
        nargs="+",
        default=[f"{name}:{count}" for name, count in DEFAULT_PATHOGENIC],
        help="Species:count entries for pathogenic genomes (default: %(default)s).",
    )
    parser.add_argument(
        "--non-pathogenic-species",
        type=str,
        nargs="+",
        default=[f"{name}:{count}" for name, count in DEFAULT_NON_PATHOGENIC],
        help="Species:count entries for non-pathogenic genomes (default: %(default)s).",
    )
    return parser


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    pathogenic_cfg = parse_species_arg(args.pathogenic_species)
    non_pathogenic_cfg = parse_species_arg(args.non_pathogenic_species)

    build_pilot_manifest(
        args.manifest,
        args.output,
        seed=args.seed,
        pathogenic_targets=pathogenic_cfg,
        non_pathogenic_targets=non_pathogenic_cfg,
    )


if __name__ == "__main__":
    main()
