#!/usr/bin/env python3
r"""
Local-only validation analyses for the FoodGuard manuscript.

These analyses are designed to be runnable on a laptop using existing cached artifacts:
- `foodguard/logs/genome_embeddings.npz` (pooled genome embeddings; N×480)
- `foodguard/analysis/genome_embeddings_enriched.parquet` (labels + metadata)

Outputs are written to `foodguard/analysis/` by default and can be referenced from the manuscript.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample


@dataclass(frozen=True)
class BinaryCVResult:
    """Summary statistics for a binary within-genus cross-validation experiment."""

    subset: str
    n: int
    accuracy: float
    accuracy_ci_low: float
    accuracy_ci_high: float
    balanced_accuracy: float
    balanced_accuracy_ci_low: float
    balanced_accuracy_ci_high: float


def _ensure_aligned(npz: np.lib.npyio.NpzFile, df: pd.DataFrame) -> None:
    if "genome_id" not in df.columns:
        return
    if "genome_id" not in npz.files:
        return
    npz_ids = npz["genome_id"]
    if len(npz_ids) != len(df):
        raise ValueError(f"Embeddings NPZ and metadata parquet have different lengths: {len(npz_ids)} vs {len(df)}")
    df_ids = df["genome_id"].to_numpy()
    if not np.array_equal(npz_ids, df_ids):
        raise ValueError("Mismatch between genome order in embeddings NPZ and metadata parquet; re-align by genome_id.")


def _bootstrap_ci(values: np.ndarray, n_bootstrap: int, random_state: int) -> tuple[float, float]:
    rng = np.random.default_rng(random_state)
    n = len(values)
    stats = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = rng.integers(0, n, size=n)
        stats[i] = values[sample].mean()
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return float(lo), float(hi)


def binary_5fold_cv(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    species: list[str],
    subset_name: str,
    n_bootstrap: int,
    random_state: int,
    k_neighbors: int = 20,
) -> tuple[BinaryCVResult, pd.DataFrame, pd.DataFrame]:
    """Run 5-fold stratified CV for a binary within-genus comparison."""
    mask = df["species"].isin(species).to_numpy()
    sub = df.loc[mask].copy()
    X = embeddings[mask]

    y = sub["pathogenicity_label"].astype(str).to_numpy()
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k_neighbors, metric="euclidean")),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    pred = cross_val_predict(model, X, y_enc, cv=cv, method="predict")
    proba = cross_val_predict(model, X, y_enc, cv=cv, method="predict_proba")

    acc = float(accuracy_score(y_enc, pred))
    bacc = float(balanced_accuracy_score(y_enc, pred))

    correct = (pred == y_enc).astype(np.float64)
    acc_ci = _bootstrap_ci(correct, n_bootstrap=n_bootstrap, random_state=random_state)

    # Balanced accuracy bootstrapped at the sample level (label-aware metric).
    rng = np.random.default_rng(random_state)
    n = len(y_enc)
    bacc_samples = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = rng.integers(0, n, size=n)
        bacc_samples[i] = balanced_accuracy_score(y_enc[sample], pred[sample])
    bacc_ci = tuple(map(float, np.quantile(bacc_samples, [0.025, 0.975])))

    result = BinaryCVResult(
        subset=subset_name,
        n=int(mask.sum()),
        accuracy=acc,
        accuracy_ci_low=acc_ci[0],
        accuracy_ci_high=acc_ci[1],
        balanced_accuracy=bacc,
        balanced_accuracy_ci_low=bacc_ci[0],
        balanced_accuracy_ci_high=bacc_ci[1],
    )

    per_sample = pd.DataFrame(
        {
            "subset": subset_name,
            "genome_id": sub["genome_id"].astype(str).to_numpy(),
            "species": sub["species"].astype(str).to_numpy(),
            "true_label": y,
            "pred_label": encoder.inverse_transform(pred),
            "confidence": proba.max(axis=1),
            "correct": pred == y_enc,
        }
    )

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
    total_errors = int((~per_sample["correct"]).sum())
    rows: list[dict[str, object]] = []
    for t in thresholds:
        kept = per_sample["confidence"] >= t
        coverage = float(kept.mean())
        if kept.any():
            acc_kept = float(per_sample.loc[kept, "correct"].mean())
        else:
            acc_kept = float("nan")
        deferred = ~kept
        if total_errors > 0:
            errors_deferred = int((deferred & (~per_sample["correct"])).sum())
            error_capture = errors_deferred / total_errors
        else:
            error_capture = float("nan")
        rows.append(
            {
                "threshold": t,
                "coverage": coverage,
                "accuracy_on_covered": acc_kept,
                "defer_rate": float(deferred.mean()),
                "error_capture_rate": float(error_capture),
                "n": int(len(per_sample)),
                "n_covered": int(kept.sum()),
            }
        )
    calibration = pd.DataFrame(rows)

    return result, per_sample, calibration


def outlier_assembly_quality_summary(
    df: pd.DataFrame,
    genome_stats_csv: Path,
) -> pd.DataFrame:
    """Summarize assembly/genome statistics for strict outliers vs non-strict genomes."""
    if "genome_id" not in df.columns:
        raise ValueError("Expected `genome_id` column in enriched metadata.")
    if not genome_stats_csv.exists():
        raise FileNotFoundError(f"Genome stats CSV not found: {genome_stats_csv}")

    stats_cols = [
        "genome_id",
        "genome_size_bp",
        "gc_content_percent",
        "cds_count",
        "gene_count",
        "contig_count",
        "n50_length",
    ]
    stats = pd.read_csv(genome_stats_csv, usecols=stats_cols)
    merged = df.merge(stats, on="genome_id", how="left", validate="one_to_one")

    missing = merged[stats_cols[1:]].isna().any(axis=1)
    if missing.any():
        raise ValueError("Missing assembly statistics for some genomes; check genome_id alignment and input CSV.")

    def q(series: pd.Series, p: float) -> float:
        return float(series.quantile(p))

    def summarize(mask: np.ndarray, label: str) -> dict[str, object]:
        sub = merged.loc[mask]
        return {
            "group": label,
            "n": int(len(sub)),
            "proteins_median": float(sub["proteins"].median()),
            "proteins_q25": q(sub["proteins"], 0.25),
            "proteins_q75": q(sub["proteins"], 0.75),
            "genome_size_bp_median": float(sub["genome_size_bp"].median()),
            "genome_size_bp_q25": q(sub["genome_size_bp"], 0.25),
            "genome_size_bp_q75": q(sub["genome_size_bp"], 0.75),
            "contig_count_median": float(sub["contig_count"].median()),
            "contig_count_q25": q(sub["contig_count"], 0.25),
            "contig_count_q75": q(sub["contig_count"], 0.75),
            "n50_length_median": float(sub["n50_length"].median()),
            "n50_length_q25": q(sub["n50_length"], 0.25),
            "n50_length_q75": q(sub["n50_length"], 0.75),
        }

    strict_mask = merged["is_outlier_strict"].to_numpy(dtype=bool)
    rows = [
        summarize(~strict_mask, "non_strict"),
        summarize(strict_mask, "strict_outliers"),
    ]
    return pd.DataFrame(rows)


def leave_one_taxon_out_knn(
    embeddings_scaled: np.ndarray,
    df: pd.DataFrame,
    k_neighbors: int = 20,
) -> pd.DataFrame:
    """Train kNN on 8 taxa, test on the held-out taxon (binary pathogenicity prior)."""
    y = df["pathogenicity_label"].astype(str).to_numpy()
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)

    species = df["species"].astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    for sp in sorted(np.unique(species)):
        test_mask = species == sp
        train_mask = ~test_mask

        X_train = embeddings_scaled[train_mask]
        y_train = y_enc[train_mask]
        X_test = embeddings_scaled[test_mask]
        y_test = y_enc[test_mask]

        clf = KNeighborsClassifier(n_neighbors=k_neighbors, metric="euclidean")
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        proba = clf.predict_proba(X_test)

        # Test sets are single-taxon and thus often single-class; accuracy is still meaningful as correctness rate.
        acc = float(accuracy_score(y_test, pred))
        conf = float(np.max(proba, axis=1).mean())
        pred_path_rate = float((encoder.inverse_transform(pred) == "pathogenic").mean())
        rows.append(
            {
                "held_out_taxon": sp,
                "n_test": int(test_mask.sum()),
                "true_label": str(encoder.inverse_transform([y_test[0]])[0]),
                "accuracy": acc,
                "mean_confidence": conf,
                "predicted_pathogenic_rate": pred_path_rate,
            }
        )
    return pd.DataFrame(rows)


def centroid_distance_vs_neighbor_mix(
    embeddings_scaled: np.ndarray,
    df: pd.DataFrame,
    output_dir: Path,
    k_neighbors: int = 20,
) -> tuple[pd.DataFrame, Path]:
    """Validate centroid distances as confusability priors via cross-species neighbor mixing."""
    species = df["species"].astype(str).to_numpy()
    species_list = sorted(np.unique(species))
    index = {sp: i for i, sp in enumerate(species_list)}

    # Centroid distances
    centroids = np.zeros((len(species_list), embeddings_scaled.shape[1]), dtype=np.float64)
    for sp in species_list:
        mask = species == sp
        centroids[index[sp]] = embeddings_scaled[mask].mean(axis=0)
    dists = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)

    # kNN mixing
    neighbors = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="euclidean", n_jobs=-1).fit(embeddings_scaled)
    nn_idx = neighbors.kneighbors(embeddings_scaled, return_distance=False)[:, 1:]
    nn_species = species[nn_idx]

    mix = np.zeros((len(species_list), len(species_list)), dtype=np.float64)
    for sp in species_list:
        mask = species == sp
        # For genomes of sp, count neighbor species frequencies.
        counts = pd.Series(nn_species[mask].ravel()).value_counts()
        total = float(mask.sum() * k_neighbors)
        for other, c in counts.items():
            mix[index[sp], index[str(other)]] = float(c) / total

    # Build pairwise table (symmetric average mixing).
    rows: list[dict[str, object]] = []
    for i, sp_i in enumerate(species_list):
        for j, sp_j in enumerate(species_list):
            if i >= j:
                continue
            mix_sym = 0.5 * (mix[i, j] + mix[j, i])
            rows.append(
                {
                    "species_i": sp_i,
                    "species_j": sp_j,
                    "centroid_distance": float(dists[i, j]),
                    "neighbor_mix_sym_k20": float(mix_sym),
                    "neighbor_mix_i_to_j_k20": float(mix[i, j]),
                    "neighbor_mix_j_to_i_k20": float(mix[j, i]),
                }
            )
    table = pd.DataFrame(rows).sort_values("centroid_distance")

    rho, p = spearmanr(table["centroid_distance"], table["neighbor_mix_sym_k20"])
    fig_path = output_dir / "centroid_distance_vs_neighbor_mix.png"
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.scatter(table["centroid_distance"], table["neighbor_mix_sym_k20"], alpha=0.8)
    ax.set_xlabel("Centroid distance (Euclidean, z-scored embedding space)")
    ax.set_ylabel("Symmetric neighbor mixing (k=20)")
    ax.set_title(f"Centroid distance vs cross-species neighbor mixing (Spearman ρ={rho:.2f}, p={p:.1e})")
    ax.grid(True, alpha=0.3)
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    return table, fig_path


def cluster_stability_bootstrap_jaccard(
    clustering_data: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    n_bootstrap: int,
    sample_frac: float,
    random_state: int,
) -> pd.DataFrame:
    """Assess HDBSCAN cluster stability via bootstrap Jaccard similarity (Hennig-style matching)."""
    try:
        import hdbscan  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "hdbscan is required for cluster stability analysis; install with `pip install hdbscan`."
        ) from e

    n_samples = len(clustering_data)
    clusterer_params = {"min_cluster_size": min_cluster_size, "min_samples": min_samples, "metric": "euclidean"}

    original_clusterer = hdbscan.HDBSCAN(**clusterer_params)
    original_labels = original_clusterer.fit_predict(clustering_data)
    unique_clusters = [c for c in np.unique(original_labels) if c != -1]

    stability_scores: dict[int, list[float]] = {int(c): [] for c in unique_clusters}

    for b in range(n_bootstrap):
        boot_idx = resample(
            range(n_samples),
            n_samples=int(n_samples * sample_frac),
            random_state=random_state + b,
        )
        X_boot = clustering_data[boot_idx]
        boot_labels = hdbscan.HDBSCAN(**clusterer_params).fit_predict(X_boot)

        boot_idx_list = list(boot_idx)
        boot_idx_set = set(boot_idx_list)
        for orig_cluster in unique_clusters:
            orig_members = set(np.where(original_labels == orig_cluster)[0])
            boot_members_orig = orig_members.intersection(boot_idx_set)
            if len(boot_members_orig) < 5:
                continue

            best_jaccard = 0.0
            for boot_cluster in np.unique(boot_labels):
                if boot_cluster == -1:
                    continue
                boot_cluster_members = {
                    boot_idx_list[i] for i in range(len(boot_idx_list)) if boot_labels[i] == boot_cluster
                }
                union = len(boot_members_orig | boot_cluster_members)
                if union == 0:
                    continue
                jaccard = len(boot_members_orig & boot_cluster_members) / union
                if jaccard > best_jaccard:
                    best_jaccard = jaccard

            stability_scores[int(orig_cluster)].append(float(best_jaccard))

    rows: list[dict[str, object]] = []
    for cluster_id, scores in stability_scores.items():
        if not scores:
            continue
        scores_arr = np.asarray(scores, dtype=np.float64)
        mean_stab = float(scores_arr.mean())
        std_stab = float(scores_arr.std())
        ci_lo, ci_hi = (float(np.quantile(scores_arr, 0.025)), float(np.quantile(scores_arr, 0.975)))

        if mean_stab >= 0.85:
            interpretation = "Highly stable"
        elif mean_stab >= 0.75:
            interpretation = "Stable"
        elif mean_stab >= 0.60:
            interpretation = "Moderate"
        else:
            interpretation = "Unstable"

        rows.append(
            {
                "cluster": int(cluster_id),
                "mean_jaccard": mean_stab,
                "std": std_stab,
                "ci_95_lo": ci_lo,
                "ci_95_hi": ci_hi,
                "interpretation": interpretation,
                "n_bootstrap_used": int(len(scores_arr)),
            }
        )

    return pd.DataFrame(rows).sort_values("cluster")


def centroid_distance_bootstrap_ci(
    embeddings_scaled: np.ndarray,
    df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    n_bootstrap: int,
    random_state: int,
) -> pd.DataFrame:
    """Bootstrap 95% CIs for selected centroid distances."""
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []
    for a, b in pairs:
        Xa = embeddings_scaled[df["species"] == a]
        Xb = embeddings_scaled[df["species"] == b]
        na, nb = len(Xa), len(Xb)
        if na == 0 or nb == 0:
            continue

        dists = np.empty(n_bootstrap, dtype=np.float64)
        for i in range(n_bootstrap):
            ca = Xa[rng.integers(0, na, size=na)].mean(axis=0)
            cb = Xb[rng.integers(0, nb, size=nb)].mean(axis=0)
            dists[i] = np.linalg.norm(ca - cb)
        lo, hi = np.quantile(dists, [0.025, 0.975])
        base = float(np.linalg.norm(Xa.mean(axis=0) - Xb.mean(axis=0)))
        rows.append(
            {
                "species_a": a,
                "species_b": b,
                "centroid_distance": base,
                "ci_95_lo": float(lo),
                "ci_95_hi": float(hi),
                "n_bootstrap": int(n_bootstrap),
            }
        )
    return pd.DataFrame(rows)


def proteins_outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Quantify association between protein count and outlier/edge-case indicators."""
    out = []
    for col in ["is_outlier_strict", "is_outlier", "is_distance_outlier"]:
        if col not in df.columns:
            continue
        for val in [True, False]:
            subset = df[df[col] == val]
            if subset.empty:
                continue
            proteins = subset["proteins"].to_numpy()
            out.append(
                {
                    "group": col,
                    "value": bool(val),
                    "n": int(len(subset)),
                    "proteins_median": float(np.median(proteins)),
                    "proteins_q25": float(np.quantile(proteins, 0.25)),
                    "proteins_q75": float(np.quantile(proteins, 0.75)),
                }
            )
    return pd.DataFrame(out)


def main() -> None:
    """Run local validation analyses and write summary artifacts."""
    parser = argparse.ArgumentParser(description="Run local-only FoodGuard validation analyses (no HPC required).")
    parser.add_argument(
        "--embeddings-npz",
        default="foodguard/logs/genome_embeddings.npz",
        help="NPZ containing pooled genome embeddings (key: embeddings).",
    )
    parser.add_argument(
        "--metadata-parquet",
        default="foodguard/analysis/genome_embeddings_enriched.parquet",
        help="Parquet containing enriched metadata (labels + analysis columns).",
    )
    parser.add_argument(
        "--output-dir",
        default="foodguard/analysis",
        help="Directory to write CSV/PNG outputs.",
    )
    parser.add_argument(
        "--genome-stats-csv",
        default="data/genome_statistics/genome_detailed_statistics_20251020_123050.csv",
        help="CSV with per-genome assembly statistics (used for outlier characterization).",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Bootstrap iterations for confidence intervals.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for CV/bootstrap reproducibility.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(args.embeddings_npz, allow_pickle=True)
    embeddings = npz["embeddings"]
    df = pd.read_parquet(args.metadata_parquet)
    _ensure_aligned(npz, df)

    # 1) E. coli within-genus CV + confidence/deferral calibration
    ecoli_result, ecoli_per_sample, ecoli_calibration = binary_5fold_cv(
        embeddings=embeddings,
        df=df,
        species=["E_coli_O157H7", "E_coli_nonpathogenic"],
        subset_name="E_coli_O157H7_vs_E_coli_nonpathogenic",
        n_bootstrap=args.n_bootstrap,
        random_state=args.random_state,
        k_neighbors=20,
    )
    pd.DataFrame([ecoli_result.__dict__]).to_csv(output_dir / "ecoli_o157_vs_nonpath_5fold_cv.csv", index=False)
    ecoli_per_sample.to_csv(output_dir / "ecoli_o157_vs_nonpath_cv_predictions.csv", index=False)
    ecoli_calibration.to_csv(output_dir / "ecoli_o157_risk_coverage.csv", index=False)

    # 1b) Listeria within-genus CV (for manuscript Table 1 reproducibility)
    listeria_result, _, _ = binary_5fold_cv(
        embeddings=embeddings,
        df=df,
        species=["L_monocytogenes", "L_innocua"],
        subset_name="L_monocytogenes_vs_L_innocua",
        n_bootstrap=args.n_bootstrap,
        random_state=args.random_state,
        k_neighbors=20,
    )
    pd.DataFrame([listeria_result.__dict__]).to_csv(output_dir / "listeria_mono_vs_innocua_5fold_cv.csv", index=False)

    # Risk-coverage plot
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    ax.plot(ecoli_calibration["coverage"], 1 - ecoli_calibration["accuracy_on_covered"], marker="o")
    for _, row in ecoli_calibration.iterrows():
        ax.annotate(
            f"t={row['threshold']}",
            (row["coverage"], 1 - row["accuracy_on_covered"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("Coverage (fraction not deferred)")
    ax.set_ylabel("Error rate on covered set")
    ax.set_title("E. coli O157:H7 vs non-pathogenic: risk–coverage tradeoff (5-fold CV)")
    ax.grid(True, alpha=0.3)
    fig_path = output_dir / "ecoli_risk_coverage.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    # 2) Leave-one-taxon-out pathogenicity prior generalization (kNN)
    embeddings_scaled = StandardScaler().fit_transform(embeddings)
    lotot = leave_one_taxon_out_knn(embeddings_scaled=embeddings_scaled, df=df, k_neighbors=20)
    lotot.to_csv(output_dir / "leave_one_taxon_out_pathogenicity_knn.csv", index=False)

    # 3) Centroid distance vs neighbor mixing (confusability validation)
    pair_table, mix_fig = centroid_distance_vs_neighbor_mix(
        embeddings_scaled=embeddings_scaled,
        df=df,
        output_dir=output_dir,
        k_neighbors=20,
    )
    pair_table.to_csv(output_dir / "centroid_distance_vs_neighbor_mix.csv", index=False)

    # 3b) Bootstrap CIs for key centroid distances
    centroid_ci = centroid_distance_bootstrap_ci(
        embeddings_scaled=embeddings_scaled,
        df=df,
        pairs=[
            ("L_monocytogenes", "L_innocua"),
            ("E_coli_O157H7", "E_coli_nonpathogenic"),
            ("E_coli_nonpathogenic", "E_fergusonii"),
            ("Salmonella_enterica", "E_coli_nonpathogenic"),
        ],
        n_bootstrap=500,
        random_state=args.random_state,
    )
    centroid_ci.to_csv(output_dir / "centroid_distance_bootstrap_ci.csv", index=False)

    # 4) Cluster stability (bootstrap Jaccard)
    clustering_data = np.load("foodguard/analysis/dim_reduction_cache.npz")["pca_result"][:, :20]
    stability_df = cluster_stability_bootstrap_jaccard(
        clustering_data=clustering_data,
        min_cluster_size=50,
        min_samples=10,
        n_bootstrap=50,
        sample_frac=0.8,
        random_state=args.random_state,
    )
    stability_df.to_csv(output_dir / "cluster_stability_jaccard.csv", index=False)

    # 5) Silhouette baseline: real labels vs permuted labels (excluding noise)
    try:
        labels = df["cluster"].to_numpy()
    except KeyError:
        labels = None
    if labels is not None:
        try:
            mask = labels != -1
            sil = float(
                silhouette_score(
                    clustering_data[mask],
                    labels[mask],
                    metric="euclidean",
                    sample_size=5000,
                    random_state=args.random_state,
                )
            )
            rng = np.random.default_rng(args.random_state)
            labels_perm = labels[mask].copy()
            rng.shuffle(labels_perm)
            sil_perm = float(
                silhouette_score(
                    clustering_data[mask],
                    labels_perm,
                    metric="euclidean",
                    sample_size=5000,
                    random_state=args.random_state,
                )
            )
            pd.DataFrame(
                [
                    {
                        "silhouette_sampled": sil,
                        "silhouette_permuted_sampled": sil_perm,
                        "sample_size": 5000,
                        "excluded_noise": True,
                    }
                ]
            ).to_csv(output_dir / "silhouette_baseline.csv", index=False)
        except ValueError:
            pass

    # 4) Protein count vs outlier indicators (annotation/assembly proxy)
    proteins_summary = proteins_outlier_summary(df)
    proteins_summary.to_csv(output_dir / "protein_count_outlier_summary.csv", index=False)

    # 4b) Assembly/genome statistics for strict outliers (laptop-feasible QC proxy)
    assembly_summary = outlier_assembly_quality_summary(df=df, genome_stats_csv=Path(args.genome_stats_csv))
    assembly_summary.to_csv(output_dir / "outlier_assembly_quality_summary.csv", index=False)


if __name__ == "__main__":
    main()
