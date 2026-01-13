"""
Generate demonstration data for the FoodGuard Dashboard.

This script creates realistic synthetic genome embedding data that mirrors
the structure of the actual GenomeTrakr dataset for development and
demonstration purposes.

Usage:
    python foodguard/generate_demo_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Configuration
REPO_ROOT = Path(__file__).parent.parent
ANALYSIS_DIR = REPO_ROOT / "foodguard" / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset configuration matching manuscript
TAXA_CONFIG = {
    "Salmonella_enterica": {"n": 7000, "pathogenic": True, "center": (25, 5)},
    "Listeria_monocytogenes": {"n": 4500, "pathogenic": True, "center": (-30, -15)},
    "E_coli_nonpathogenic": {"n": 4312, "pathogenic": False, "center": (15, -8)},
    "Bacillus_subtilis": {"n": 2361, "pathogenic": False, "center": (-25, 10)},
    "E_coli_O157H7": {"n": 1500, "pathogenic": True, "center": (18, -5)},
    "Citrobacter_koseri": {"n": 897, "pathogenic": False, "center": (8, 12)},
    "Listeria_innocua": {"n": 449, "pathogenic": False, "center": (-28, -12)},
    "Escherichia_fergusonii": {"n": 438, "pathogenic": False, "center": (12, -3)},
    "Citrobacter_freundii": {"n": 200, "pathogenic": False, "center": (5, 15)},
}

# Total should be ~21,657 to match manuscript
SCALE_FACTOR = 1.0  # Set to 0.1 for faster demo generation


def generate_species_embeddings(
    species: str,
    n_samples: int,
    center: tuple[float, float],
    spread: float = 3.0,
    rng: np.random.Generator = None
) -> tuple[np.ndarray, np.ndarray]:
    """Generate clustered embeddings for a species in PCA space."""
    if rng is None:
        rng = np.random.default_rng(42)

    # Primary cluster
    n_main = int(n_samples * 0.85)
    n_outlier = n_samples - n_main

    # Main cluster with species-specific spread
    main_pca = np.column_stack([
        rng.normal(center[0], spread, n_main),
        rng.normal(center[1], spread * 0.6, n_main)
    ])

    # Some outliers with larger spread
    outlier_pca = np.column_stack([
        rng.normal(center[0], spread * 3, n_outlier),
        rng.normal(center[1], spread * 2, n_outlier)
    ])

    pca_coords = np.vstack([main_pca, outlier_pca])

    # Mark outliers
    is_outlier = np.concatenate([
        np.zeros(n_main, dtype=bool),
        rng.random(n_outlier) < 0.3  # 30% of spread points are true outliers
    ])

    return pca_coords, is_outlier


def compute_homophily(
    pca_coords: np.ndarray,
    labels: np.ndarray,
    k: int = 20
) -> np.ndarray:
    """Compute k-NN homophily for each point."""
    from scipy.spatial.distance import cdist

    n = len(pca_coords)
    homophily = np.zeros(n)

    # Process in chunks for memory efficiency
    chunk_size = 1000
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        chunk = pca_coords[i:end]

        # Compute distances to all points
        dists = cdist(chunk, pca_coords, metric='euclidean')

        for j, d in enumerate(dists):
            # Get k nearest neighbors (excluding self)
            neighbor_idx = np.argsort(d)[1:k+1]
            same_label = (labels[neighbor_idx] == labels[i + j]).mean()
            homophily[i + j] = same_label

    return homophily


def assign_clusters(
    pca_coords: np.ndarray,
    species_labels: np.ndarray,
    n_clusters: int = 34,
    noise_fraction: float = 0.356
) -> np.ndarray:
    """Assign cluster labels similar to HDBSCAN output."""
    rng = np.random.default_rng(42)
    n = len(pca_coords)

    # Start with species-based clustering
    unique_species = np.unique(species_labels)
    species_to_base_cluster = {sp: i * 3 for i, sp in enumerate(unique_species)}

    clusters = np.array([species_to_base_cluster[sp] for sp in species_labels])

    # Add sub-clustering based on PCA position
    for sp in unique_species:
        mask = species_labels == sp
        sp_coords = pca_coords[mask]

        # K-means-like subclustering (simplified)
        if mask.sum() > 100:
            # Split into 2-4 subclusters based on PC1
            pc1 = sp_coords[:, 0]
            thresholds = np.percentile(pc1, [33, 66])
            subcluster = np.digitize(pc1, thresholds)
            clusters[mask] = clusters[mask] + subcluster

    # Mark some as noise
    noise_mask = rng.random(n) < noise_fraction
    clusters[noise_mask] = -1

    # Renumber clusters to be contiguous
    unique_clusters = np.unique(clusters[clusters >= 0])
    cluster_map = {old: new for new, old in enumerate(unique_clusters)}
    cluster_map[-1] = -1
    clusters = np.array([cluster_map[c] for c in clusters])

    return clusters


def generate_demo_dataset():
    """Generate the complete demonstration dataset."""
    print("Generating FoodGuard demonstration dataset...")
    print(f"Scale factor: {SCALE_FACTOR}")

    rng = np.random.default_rng(42)

    all_data = []
    all_pca = []

    for species, config in TAXA_CONFIG.items():
        n = int(config["n"] * SCALE_FACTOR)
        if n < 10:
            n = 10  # Minimum samples

        print(f"  Generating {n:,} {species} genomes...")

        # Generate PCA embeddings
        pca, is_outlier = generate_species_embeddings(
            species, n, config["center"],
            spread=3.0 if "coli" in species.lower() else 2.5,
            rng=rng
        )

        # Generate metadata
        genome_ids = [f"GCA_{species[:3].upper()}_{i:06d}" for i in range(n)]
        proteins = rng.integers(3000, 6000, n)

        species_data = pd.DataFrame({
            "genome_id": genome_ids,
            "species": species,
            "pathogenicity_label": "pathogenic" if config["pathogenic"] else "non-pathogenic",
            "proteins": proteins,
            "is_outlier": is_outlier,
            "is_outlier_strict": is_outlier & (rng.random(n) < 0.5),  # Stricter criterion
        })

        all_data.append(species_data)
        all_pca.append(pca)

    # Combine all data
    df = pd.concat(all_data, ignore_index=True)
    pca_coords = np.vstack(all_pca)

    print(f"\nTotal genomes: {len(df):,}")

    # Compute homophily scores
    print("Computing species homophily...")
    df["species_homophily"] = compute_homophily(
        pca_coords, df["species"].values, k=20
    )

    print("Computing pathogenicity homophily...")
    df["pathogenicity_homophily"] = compute_homophily(
        pca_coords, df["pathogenicity_label"].values, k=20
    )

    # Assign clusters
    print("Assigning clusters...")
    df["cluster"] = assign_clusters(pca_coords, df["species"].values)

    # Generate UMAP-like coordinates (simplified transformation)
    print("Generating UMAP coordinates...")
    umap_coords = pca_coords * 0.3 + rng.normal(0, 0.5, pca_coords.shape)

    # Save outputs
    print("\nSaving outputs...")

    # Save enriched parquet
    output_path = ANALYSIS_DIR / "genome_embeddings_enriched.parquet"
    df.to_parquet(output_path, index=False)
    print(f"  Saved: {output_path}")

    # Save dimension reduction cache
    cache_path = ANALYSIS_DIR / "dim_reduction_cache.npz"
    np.savez_compressed(
        cache_path,
        pca_result=pca_coords,
        umap_result=umap_coords
    )
    print(f"  Saved: {cache_path}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total genomes: {len(df):,}")
    print(f"Species: {df['species'].nunique()}")
    print(f"Pathogenic: {(df['pathogenicity_label'] == 'pathogenic').sum():,}")
    print(f"Non-pathogenic: {(df['pathogenicity_label'] == 'non-pathogenic').sum():,}")
    print(f"Mean species homophily: {df['species_homophily'].mean():.4f}")
    print(f"Mean pathogenicity homophily: {df['pathogenicity_homophily'].mean():.4f}")
    print(f"Clusters: {df[df['cluster'] >= 0]['cluster'].nunique()}")
    print(f"Noise points: {(df['cluster'] == -1).sum():,} ({(df['cluster'] == -1).mean()*100:.1f}%)")
    print(f"Outliers (strict): {df['is_outlier_strict'].sum():,} ({df['is_outlier_strict'].mean()*100:.1f}%)")

    print("\n" + "=" * 60)
    print("SPECIES DISTRIBUTION")
    print("=" * 60)
    for species in df["species"].value_counts().index:
        count = (df["species"] == species).sum()
        homophily = df[df["species"] == species]["species_homophily"].mean()
        print(f"  {species}: {count:,} (homophily: {homophily:.4f})")

    print("\nDemo data generation complete!")
    print(f"\nTo launch dashboard:")
    print(f"  streamlit run foodguard/dashboard.py")

    return df, pca_coords, umap_coords


if __name__ == "__main__":
    generate_demo_dataset()
