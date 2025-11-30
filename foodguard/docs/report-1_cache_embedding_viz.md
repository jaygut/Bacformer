# FoodGuardAI Cache Embedding Landscape — Interim Report (Dec 2025)

## Purpose
Summarize the current state of the ESM-2 cache population on Apolo-3 (H100), present the pooled genome embedding landscape (PCA/UMAP/t-SNE), and connect findings to the FoodGuardAI objective: calibrated pathogen detection with fast, cache-first inference.

## Status Snapshot
- **Cache population:** In progress on Apolo-3 `a3-accel-0` (H100 NVL). Cache dir: `/home/rsg-jcorre38/Jay_Proyects/FoodGuardAI/Bacformer/.cache/esm2_h100`. Manifest (rewritten paths): `.../gbff_manifest_full_20251020_123050_h100.tsv`.
- **Preflight:** GPU run validated; cache files being generated (target 21,657 genomes).
- **Splits:** Train/val/test created (70/15/15) with H100 manifest: Train 15,159; Val 3,249; Test 3,249.
- **Viz job:** Full-sample pooling + PCA/UMAP/t-SNE completed; HTML dashboard generated (`logs/viz_cache_pca_umap.html`) using pooled genome embeddings cached to `logs/genome_embeddings.npz`.

## Figure (placeholder)
![PCA/UMAP/t-SNE of pooled genome embeddings colored by pathogenicity](path/to/viz_cache_pca_umap.html_or_png)

**Figure legend:** Pooled ESM-2 genome embeddings (mean of per-protein embeddings) for cached genomes. Each point = one genome. Colors encode manifest pathogenicity (`pathogenic` red, `non_pathogenic` blue, `unknown` gray). Panels show 2D projections via PCA (left), UMAP (middle), and t-SNE (right). Hover (in HTML) reveals genome_id, species, pathogenicity, and protein count.

## Rationale & Interpretation
- **Goal alignment:** Fast, cache-first inference requires precomputed embeddings. Visualizing pooled embeddings assesses class separability and data quality before fine-tuning.
- **Embedding separability:** t-SNE and UMAP show clear clustering between pathogenic vs non-pathogenic genomes, indicating the pooled ESM-2 representations already capture pathogen-specific signal; PCA shows weaker separation (expected for linear projection).
- **Quality checks:** Uniform cluster shapes with minimal mixing suggest consistent preprocessing and cache keying. Isolated points may flag rare genomes or QC issues (worth spot-checking).
- **Throughput readiness:** The H100-backed cache run is viable; visualization confirms embeddings load and pool correctly at scale.

## What’s next (prioritized)
1) **Complete cache population** to 21,657 files; run `generate_cache_progress.py` and integrity spot-checks (`torch.load` on random files).
2) **Benchmark** (embedding-only) on 50 genomes from the H100 manifest: `scripts/pilot_benchmark_pipeline.py ... --cache-dir .cache/esm2_h100 --limit 50 --report logs/pipeline_benchmark.jsonl`; confirm hit-rate/latency targets.
3) **Full viz refresh** once cache is ≥80% complete (reuse `--embeddings-cache` to avoid reloading).
4) **Fine-tuning prep**: use the generated splits with the cached embeddings; then train the Bacformer classifier and proceed to calibration.
5) **Docs & reporting**: update P0 doc and `apolo_hpc_cache_report.md` with final cache counts, benchmarks, and a link to the viz dashboard.

## Notes on artifacts
- Cache dir: `/home/rsg-jcorre38/Jay_Proyects/FoodGuardAI/Bacformer/.cache/esm2_h100`
- Viz outputs: `logs/viz_cache_pca_umap.html`; pooled embeddings cache: `logs/genome_embeddings.npz`
- Scripts: `scripts/visualize_cache_embeddings.py`, `scripts/submit_visualize_cache.sh`, `scripts/pilot_populate_esm2_cache.py`, `scripts/generate_cache_progress.py`
