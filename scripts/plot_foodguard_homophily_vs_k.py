#!/usr/bin/env python3
r"""
Plot multi-scale kNN homophily for FoodGuard genome embeddings.

This script recomputes pathogenicity-prior neighborhood agreement across multiple k values using cached genome embeddings
(`foodguard/logs/genome_embeddings.npz`) and metadata (`foodguard/analysis/genome_embeddings_enriched.parquet`).

Why this exists:
- Mean homophily is often near-ceiling in taxonomically structured datasets; plotting only mean homophily on a 0–1 axis
  can hide the informative differences.
- We therefore visualize (i) mean homophily with a zoomed y-range and (ii) the fraction of genomes with mixed
  neighborhoods (homophily below a configurable threshold).

Example:
    python scripts/plot_foodguard_homophily_vs_k.py \\
        --output foodguard/analysis/homophily_vs_k.png \\
        --summary-csv foodguard/analysis/homophily_vs_k_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def _parse_int_list(csv: str) -> list[int]:
    values: list[int] = []
    for item in csv.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        values.append(int(stripped))
    return values


def _label_for_species(species_name: str) -> str:
    mapping = {
        "Salmonella_enterica": "Salmonella enterica",
        "L_monocytogenes": "Listeria monocytogenes",
        "L_innocua": "Listeria innocua",
        "E_coli_O157H7": "E. coli O157:H7",
        "E_coli_nonpathogenic": "E. coli (non-pathogenic)",
        "E_fergusonii": "E. fergusonii",
    }
    return mapping.get(species_name, species_name.replace("_", " "))


def compute_multiscale_homophily(
    embeddings: np.ndarray,
    pathogenicity_labels: np.ndarray,
    species: np.ndarray,
    k_values: list[int],
    mixed_threshold: float,
    tracked_species: list[str],
) -> pd.DataFrame:
    """Compute mean homophily and mixed-neighborhood fraction for multiple k values."""
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")

    if len(embeddings) != len(pathogenicity_labels) or len(embeddings) != len(species):
        raise ValueError("Embedding rows and label vectors must have the same length.")

    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("k_values must be a non-empty list of positive integers.")

    max_k = max(k_values)
    if max_k >= len(embeddings):
        raise ValueError(f"max(k) must be < n_samples; got max_k={max_k}, n_samples={len(embeddings)}")

    neighbors = NearestNeighbors(n_neighbors=max_k + 1, metric="euclidean", n_jobs=-1)
    neighbors.fit(embeddings)

    neighbor_indices = neighbors.kneighbors(embeddings, return_distance=False)[:, 1:]  # drop self
    same_label = pathogenicity_labels[neighbor_indices] == pathogenicity_labels[:, None]
    cumulative_same = np.cumsum(same_label.astype(np.int32), axis=1)

    def summarize(mask: np.ndarray, group: str) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for k in k_values:
            homophily = cumulative_same[mask, k - 1] / float(k)
            rows.append(
                {
                    "group": group,
                    "k": k,
                    "mean_homophily": float(homophily.mean()),
                    "mixed_fraction": float((homophily < mixed_threshold).mean()),
                    "n": int(mask.sum()),
                }
            )
        return rows

    all_mask = np.ones(len(embeddings), dtype=bool)
    out_rows: list[dict[str, object]] = summarize(all_mask, "All genomes")

    for species_name in tracked_species:
        species_mask = species == species_name
        if species_mask.any():
            out_rows.extend(summarize(species_mask, _label_for_species(species_name)))

    return pd.DataFrame(out_rows)


def plot_multiscale_homophily(
    summary: pd.DataFrame,
    mixed_threshold: float,
    output_path: Path,
) -> None:
    """Render a two-panel figure (mean homophily; mixed-neighborhood fraction)."""
    groups = summary["group"].unique().tolist()
    k_values = sorted(summary["k"].unique().tolist())

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 7), sharex=True, constrained_layout=True)
    ax_mean, ax_mixed = axes

    colors = dict(zip(groups, plt.cm.tab10.colors, strict=False))

    for group in groups:
        sub = summary[summary["group"] == group].sort_values("k")
        ax_mean.plot(
            sub["k"],
            sub["mean_homophily"],
            marker="o",
            linewidth=2,
            label=group,
            color=colors.get(group),
        )
        ax_mixed.plot(
            sub["k"],
            sub["mixed_fraction"],
            marker="o",
            linewidth=2,
            label=group,
            color=colors.get(group),
        )

    min_mean = float(summary["mean_homophily"].min())
    ax_mean.set_ylim(max(0.0, min_mean - 0.015), 1.001)
    ax_mean.set_ylabel("Mean homophily (pathogenicity prior)")
    ax_mean.set_title("Multi-scale neighborhood agreement (pathogenicity prior)")
    ax_mean.grid(True, alpha=0.3)
    ax_mean.legend(loc="lower left", ncol=2, frameon=True)

    ax_mixed.set_xlabel("k (nearest neighbors)")
    ax_mixed.set_ylabel(f"Fraction with homophily < {mixed_threshold:g}")
    ax_mixed.set_xticks(k_values)
    ax_mixed.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Parse CLI args, compute homophily, and write outputs."""
    parser = argparse.ArgumentParser(description="Plot multi-scale kNN homophily for FoodGuard embeddings.")
    parser.add_argument(
        "--embeddings-npz",
        default="foodguard/logs/genome_embeddings.npz",
        help="NPZ containing pooled genome embeddings (key: embeddings).",
    )
    parser.add_argument(
        "--metadata-parquet",
        default="foodguard/analysis/genome_embeddings_enriched.parquet",
        help="Parquet containing genome metadata with columns: species, pathogenicity_label.",
    )
    parser.add_argument(
        "--k-values",
        default="1,5,10,20,50,100",
        help="Comma-separated k values to evaluate.",
    )
    parser.add_argument(
        "--mixed-threshold",
        type=float,
        default=0.9,
        help="Homophily threshold below which a genome is considered to have a mixed neighborhood.",
    )
    parser.add_argument(
        "--tracked-species",
        default="Salmonella_enterica,L_monocytogenes,E_coli_nonpathogenic,E_coli_O157H7,L_innocua,E_fergusonii",
        help="Comma-separated species names to plot as separate curves.",
    )
    parser.add_argument(
        "--output",
        default="foodguard/analysis/homophily_vs_k.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Optional CSV path for the computed summary table.",
    )
    args = parser.parse_args()

    embeddings_path = Path(args.embeddings_npz)
    metadata_path = Path(args.metadata_parquet)
    output_path = Path(args.output)

    k_values = _parse_int_list(args.k_values)
    tracked_species = [s.strip() for s in args.tracked_species.split(",") if s.strip()]
    mixed_threshold = float(args.mixed_threshold)

    npz = np.load(embeddings_path, allow_pickle=True)
    embeddings = npz["embeddings"]

    df = pd.read_parquet(metadata_path)
    if "genome_id" in df.columns:
        # Enforce consistent ordering between embeddings NPZ and the enriched metadata bundle.
        npz_ids = npz["genome_id"] if "genome_id" in npz.files else None
        if npz_ids is not None and len(npz_ids) == len(df):
            df_ids = df["genome_id"].to_numpy()
            if not np.array_equal(npz_ids, df_ids):
                raise ValueError("Mismatch between genome order in embeddings NPZ and metadata parquet; re-align by genome_id.")

    pathogenicity_labels = df["pathogenicity_label"].astype(str).to_numpy()
    species = df["species"].astype(str).to_numpy()

    embeddings_scaled = StandardScaler().fit_transform(embeddings)

    summary = compute_multiscale_homophily(
        embeddings=embeddings_scaled,
        pathogenicity_labels=pathogenicity_labels,
        species=species,
        k_values=k_values,
        mixed_threshold=mixed_threshold,
        tracked_species=tracked_species,
    )

    if args.summary_csv:
        summary_path = Path(args.summary_csv)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.sort_values(["group", "k"]).to_csv(summary_path, index=False)

    plot_multiscale_homophily(summary=summary, mixed_threshold=mixed_threshold, output_path=output_path)


if __name__ == "__main__":
    main()
