# PRD: FoodGuardAI Genome Embedding Analysis Notebook

**Author:** Jay Gutierrezer (PhD) — FoodGuardAI / Bacformer  
**Target Notebook Path:** `/Users/jaygut/Documents/Side_Projects/Bacformer/notebooks/foodguard_cache_embedding_analysis.ipynb`  
**Input Artifacts:** `/Users/jaygut/Documents/Side_Projects/Bacformer/foodguard/logs/genome_embeddings.npz`, manifest `.../data/genome_statistics/gbff_manifest_full_20251020_123050_h100.tsv`  
**Goal:** Build a publication-grade, interactive analysis that characterizes the cached ESM-2 genome embeddings, validates data quality, and uncovers biological/operational insights aligned with FoodGuardAI’s pathogen detection mission.

## Narrative Arc (scientific storyline)
1) **Context & Motivation:** ESM-2 protein language models capture proteome-wide signals. Pooled genome embeddings are the backbone of FoodGuardAI’s fast, cache-first pathogen risk scoring. We probe these embeddings to assess separability (pathogenic vs non-pathogenic), species structure, and potential label noise.  
2) **Data & Methods:** Describe ESM-2 (facebook/esm2_t12_35M_UR50D), per-protein embeddings, mean pooling to genome vectors, cached in `prot_emb_*.pt`, aggregated into `genome_embeddings.npz` (embeddings + metadata).  
3) **Results:** Dimensionality reductions (PCA/UMAP/t-SNE), clustering, neighborhood homophily, distance diagnostics, species/pathogenicity enrichment. Highlight clear class separation and any ambiguities.  
4) **Implications for FoodGuardAI:** Strong upstream separability reduces fine-tuning burden, informs calibration, and guides evidence/novelty modules. Mixed clusters/outliers flag QC or biological edge cases.  
5) **Next Steps:** Feed insights into fine-tuning (class balance, hard negatives), calibration, and API posture; propose follow-up experiments or data curation.

## Notebook Structure & Sections
1) **Setup & Inputs**  
   - Load NPZ (`genome_embeddings.npz`) and manifest. Validate shapes, row alignment, and metadata (genome_id, species, pathogenicity, proteins, taxon_id).  
   - Config: random seeds, sample controls (full vs sampled), output dirs.  
   - Dependencies: numpy, pandas, plotly, sklearn, umap-learn, hdbscan (optional), tqdm, scipy.

2) **Data Overview & QC**  
   - Counts: total genomes, class balance (pathogenic vs non-pathogenic vs unknown), species distribution.  
   - Protein count stats; flag outliers (very low/high).  
   - Missing/unknown labels; duplicates check.

3) **Dimensionality Reduction & Visualization**  
   - Recompute PCA/UMAP/t-SNE on pooled embeddings (reuse cached if desired).  
   - Plots colored by pathogenicity and by species; interactive hover (genome_id, species, pathogenicity, proteins).  
   - Narrative: comment on separability, cluster morphology, and visible class structure.

4) **Clustering & Enrichment**  
   - Run HDBSCAN/DBSCAN on embeddings.  
   - For each cluster: species composition, pathogenicity fraction, entropy; identify pure vs mixed clusters.  
   - Narrative: mixed clusters may indicate label noise or convergent biology; pure clusters can be exemplars/reference anchors.

5) **Nearest-Neighbor Homophily**  
   - KNN in embedding space; for each genome, compute fraction of neighbors sharing pathogenicity/species.  
   - Aggregate homophily by class/species; highlight genomes with low homophily (potential mislabels or edge cases).  
   - Narrative: strong homophily supports downstream PS reliability; low homophily flags review targets.

6) **Distance Diagnostics**  
   - Intra-/inter-class distance distributions; silhouette/separability metrics.  
   - Species centroid distances; heatmap of inter-species embedding distances.  
   - Narrative: quantify how distinct pathogenic vs non-pathogenic proteomes are at the embedding level.

7) **Outlier Analysis**  
   - Identify outliers via distance to k-th neighbor or cluster noise points.  
   - List genome_ids/species/pathogenicity; suggest QC or biological follow-up (novel strains, assembly issues).

8) **Artifacts & Outputs**  
   - Save enriched DataFrame (embeddings + metadata + cluster IDs + homophily) to Parquet/CSV in `logs/` or `analysis/`.  
   - Save interactive plots (HTML) and static PNGs for reports.  
   - Summarize key metrics (homophily, silhouette, cluster purity) in a concise table.

9) **Interpretation & Recommendations**  
   - Discuss how observed separability supports PS/CRS.  
   - Recommend data curation or re-labeling for mixed clusters/outliers.  
   - Tie to downstream: calibration targets, evidence integration (VFDB/CARD), novelty scoring priorities.

## Acceptance Criteria
- Notebook runs end-to-end on H100 cache outputs (`genome_embeddings.npz` + manifest).  
- Produces interactive PCA/UMAP/t-SNE colored by pathogenicity and species.  
- Delivers clustering, homophily, and distance diagnostics with clear narrative text cells.  
- Exports enriched metadata table and plots to disk.  
- Highlights actionable findings (mixed clusters, outliers) and links to FoodGuardAI goals.

## Implementation Notes
- Input defaults in the notebook:  
  - `EMB_PATH = "/Users/jaygut/Documents/Side_Projects/Bacformer/foodguard/logs/genome_embeddings.npz"`  
  - `MANIFEST = "/Users/jaygut/Documents/Side_Projects/Bacformer/data/genome_statistics/gbff_manifest_full_20251020_123050_h100.tsv"`  
- Sampling: support full and subsampled modes; ensure reproducibility (seed).  
- Performance: PCA/UMAP/t-SNE can reuse cached reductions if desired; allow toggles.  
- Plots: use Plotly for interactivity; add hover fields: genome_id, species, pathogenicity, proteins, cluster, homophily score (if computed).
