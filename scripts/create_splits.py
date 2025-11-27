#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def create_splits(manifest_path: Path, output_dir: Path, random_seed: int) -> None:
    df = pd.read_csv(manifest_path, sep="\t")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        stratify=df["species"],
        random_state=random_seed,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["species"],
        random_state=random_seed,
    )

    train_df.to_csv(output_dir / "train_split.tsv", sep="\t", index=False)
    val_df.to_csv(output_dir / "val_split.tsv", sep="\t", index=False)
    test_df.to_csv(output_dir / "test_split.tsv", sep="\t", index=False)

    print("Splits created successfully")
    print(f"Train: {len(train_df)}")
    print(f"Val: {len(val_df)}")
    print(f"Test: {len(test_df)}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create stratified train/val/test splits from manifest")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--random-seed", type=int, default=42)
    return p


def main() -> None:
    args = build_parser().parse_args()
    create_splits(args.manifest, args.output_dir, args.random_seed)


if __name__ == "__main__":
    main()

