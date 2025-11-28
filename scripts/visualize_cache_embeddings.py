#!/usr/bin/env python3
"""
Create an interactive two-panel dashboard (PCA vs UMAP) of cached genome embeddings.

The script:
- Samples genomes from the manifest
- Reconstructs the cache key for each genome (using the same logic as population)
- Loads cached per-protein embeddings, mean-pools to a genome embedding
- Projects to 2D via PCA and UMAP
- Renders an interactive Plotly HTML with hover metadata (genome_id, species, pathogenicity)

Requirements (install into your env):
    pip install plotly scikit-learn umap-learn pandas numpy tqdm
    # optional: joblib for faster cache of projections (not required)

Usage example:
    python scripts/visualize_cache_embeddings.py \
        --manifest /path/to/gbff_manifest_full_20251020_123050_h100.tsv \
        --cache-dir /path/to/.cache/esm2_h100 \
        --output viz_cache_pca_umap.html \
        --sample 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from tqdm import tqdm

from bacformer.pp.embed_prot_seqs import _make_cache_key
from bacformer.pp.preprocess import preprocess_genome_assembly


def flatten_embeddings(embs: List[List[np.ndarray]] | List[np.ndarray]) -> np.ndarray:
    """Flatten nested contig/protein embeddings to [N_proteins, dim]."""
    flat: list[np.ndarray] = []
    if not embs:
        return np.zeros((0, 0), dtype=np.float32)
    if isinstance(embs[0], list):
        for contig in embs:  # type: ignore[index]
            flat.extend(contig)
    else:
        flat = list(embs)  # type: ignore[list-item]
    return np.stack(flat) if flat else np.zeros((0, 0), dtype=np.float32)


def load_genome_embedding(
    gbff_path: str,
    cache_dir: Path,
    model_id: str,
    max_prot_seq_len: int,
) -> Tuple[np.ndarray | None, int]:
    """Load cached per-protein embeddings and return genome-level mean + protein count."""
    pre = preprocess_genome_assembly(gbff_path)
    sequences = pre["protein_sequence"]
    key = _make_cache_key(
        protein_sequences=sequences,
        model_id=model_id,
        model_type="esm2",
        max_prot_seq_len=max_prot_seq_len,
        genome_pooling_method=None,
    )
    cache_path = cache_dir / f"prot_emb_{key}.pt"
    if not cache_path.exists():
        return None, 0

    import torch  # defer import

    obj = torch.load(cache_path, map_location="cpu")
    arr = flatten_embeddings(obj)
    if arr.shape[0] == 0:
        return None, 0
    genome_emb = arr.mean(axis=0)
    return genome_emb, arr.shape[0]


def build_projections(embs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute PCA and UMAP projections (fallback to PCA if UMAP missing)."""
    pca_coords = PCA(n_components=2).fit_transform(embs)
    try:
        from umap import UMAP  # type: ignore

        umap_coords = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1).fit_transform(embs)
    except Exception:
        umap_coords = pca_coords.copy()
    return pca_coords, umap_coords


def make_figure(df: pd.DataFrame) -> go.Figure:
    """Create side-by-side PCA/UMAP Plotly scatter plots."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("PCA", "UMAP"))
    # df["pathogenicity"] is normalized to string labels upstream
    labels = df["pathogenicity"].fillna("unknown").astype(str)
    palette = {"pathogenic": "#d62728", "non_pathogenic": "#1f77b4", "unknown": "#7f7f7f"}
    colors = labels.map(palette).fillna("#7f7f7f")
    hover = (
        "genome_id: " + df["genome_id"].astype(str)
        + "<br>species: " + df["species"].astype(str)
        + "<br>pathogenicity: " + labels
        + "<br>proteins: " + df["proteins"].astype(str)
    )

    fig.add_trace(
        go.Scattergl(
            x=df["pca_x"],
            y=df["pca_y"],
            mode="markers",
            marker=dict(color=colors, size=6, opacity=0.8),
            text=hover,
            hoverinfo="text",
            name="PCA",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=df["umap_x"],
            y=df["umap_y"],
            mode="markers",
            marker=dict(color=colors, size=6, opacity=0.8),
            text=hover,
            hoverinfo="text",
            name="UMAP",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        title="FoodGuardAI Cache Embeddings: PCA vs UMAP",
        template="plotly_white",
        height=600,
        width=1200,
        showlegend=False,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize cached genome embeddings (PCA vs UMAP).")
    parser.add_argument("--manifest", required=True, help="Manifest TSV with genome_id, species, is_pathogenic, gbff_path.")
    parser.add_argument("--cache-dir", required=True, help="Directory containing prot_emb_*.pt files.")
    parser.add_argument("--output", default="viz_cache_pca_umap.html", help="Output HTML file.")
    parser.add_argument("--sample", type=int, default=0, help="Number of genomes to sample (0=all).")
    parser.add_argument("--embeddings-cache", default=None, help="Optional NPZ to store/load genome embeddings (mean pooled).")
    parser.add_argument("--model-id", default="facebook/esm2_t12_35M_UR50D", help="Model ID used for cache keys.")
    parser.add_argument("--max-prot-seq-len", type=int, default=1024, help="Max protein seq length used for cache keys.")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    df = pd.read_csv(args.manifest, sep="\t")
    if args.sample and args.sample > 0 and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=42)

    embeddings_cache = Path(args.embeddings_cache) if args.embeddings_cache else None
    records = []
    embeddings = []

    # Load cached genome embeddings if provided and exists
    if embeddings_cache and embeddings_cache.exists():
        data = np.load(embeddings_cache, allow_pickle=True)
        embs_arr = data["embeddings"]
        meta = data["meta"].tolist()
        records = meta
    else:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading cache embeddings"):
            genome_id = row.get("genome_id")
            species = row.get("species")
            path = row.get("gbff_path")
            is_path = row.get("is_pathogenic", None)
            emb, n_prots = load_genome_embedding(
                gbff_path=str(path),
                cache_dir=cache_dir,
                model_id=args.model_id,
                max_prot_seq_len=args.max_prot_seq_len,
            )
            if emb is None:
                continue
            embeddings.append(emb)
            records.append(
                {
                    "genome_id": genome_id,
                    "species": species,
                    "pathogenicity": is_path,
                    "proteins": n_prots,
                }
            )
        if not embeddings:
            raise SystemExit("No embeddings loaded; check manifest paths and cache_dir.")
        embs_arr = np.vstack(embeddings)
        # Save cache if requested
        if embeddings_cache:
            embeddings_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(embeddings_cache, embeddings=embs_arr, meta=np.array(records, dtype=object))

    pca_coords, umap_coords = build_projections(embs_arr)
    out_df = pd.DataFrame(records)
    out_df["pca_x"], out_df["pca_y"] = pca_coords[:, 0], pca_coords[:, 1]
    out_df["umap_x"], out_df["umap_y"] = umap_coords[:, 0], umap_coords[:, 1]

    # Normalize pathogenicity labels (handle ints/strings)
    def _label(x):
        if pd.isna(x):
            return "unknown"
        if isinstance(x, str):
            xs = x.strip().lower()
            if xs in {"1", "pathogenic"}:
                return "pathogenic"
            if xs in {"0", "non_pathogenic", "nonpathogenic", "non-pathogenic"}:
                return "non_pathogenic"
            return xs
        try:
            return "pathogenic" if int(x) == 1 else "non_pathogenic"
        except Exception:
            return "unknown"

    out_df["pathogenicity"] = out_df["pathogenicity"].apply(_label)

    fig = make_figure(out_df)
    fig.write_html(args.output, include_plotlyjs="cdn")
    print(f"Wrote {len(out_df)} points to {args.output}")
    print(out_df["pathogenicity"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
