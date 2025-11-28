# FoodGuard AI — Execution TODOs (Apolo-3 H100) – Dec 2025

## Environment & Infrastructure
- [x] Identify GPU node with modern driver (Apolo-3 `a3-accel-0`, H100 NVL, driver 575/CUDA 12.9)
- [x] Python env: `FoodGuard310` (Python 3.10), torch 2.2.2+cu121, numpy 1.26.4
- [x] Data copied to new cluster: manifest + `data/genometrakr` paths rewritten to `/home/rsg-jcorre38/Jay_Proyects/FoodGuardAI/...`
- [ ] Install optional faESM (speedup) if compatible; otherwise proceed with HF ESM-2

## Cache Population (P0.1)
- [x] Preflight on H100 (limit=2) with cache hits confirmed
- [x] Launch sharded population on `a3-accel-0` (2 GPUs, shard-count=2) via `scripts/submit_cache_population_accel.sh`
- [ ] Monitor counts → target 21,657 cache files (`.cache/esm2_h100`)
- [ ] Run `generate_cache_progress.py` regularly; capture coverage JSON
- [ ] Integrity spot-check: torch.load on random 10 files
- [ ] Document final stats in `foodguard/docs/apolo_hpc_cache_report.md`

## Benchmark & Validation (P0.2)
- [ ] Run embedding-only benchmark on 50 genomes:
  - `python scripts/pilot_benchmark_pipeline.py --manifest ..._h100.tsv --cache-dir .cache/esm2_h100 --limit 50 --report logs/pipeline_benchmark.jsonl`
- [ ] Run full pipeline benchmark once classifier is available (`--model-path <ckpt>`)
- [ ] Check hit-rate ≥90%, cache lookup <5s, p95 latency <30s

## Splits & Fine-Tuning (P0.3)
- [ ] Generate train/val/test splits with rewritten manifest paths
  - `python scripts/create_splits.py --manifest ..._h100.tsv --output-dir data/splits`
- [ ] Prepare training data using cached embeddings
- [ ] Fine-tune Bacformer classifier (target: high balanced accuracy); save checkpoint
- [ ] Calibration: fit/save calibrators; integrate thresholds in pipeline config

## Visualization & Analysis
- [x] Add PCA/UMAP dashboard script (`scripts/visualize_cache_embeddings.py`) with embedding cache option
- [x] SBATCH helper (`scripts/submit_visualize_cache.sh`) for CPU-run viz
- [ ] Run full-viz job with `SAMPLE=0` and `EMBEDDINGS_CACHE=.../genome_embeddings.npz`; review HTML

## API & Integration
- [ ] Wire cache dir into pipeline config for production
- [ ] Update FastAPI scaffold (load fine-tuned model + cache paths)
- [ ] Add monitoring/logging for cache hits/misses in service

## Documentation & Reporting
- [x] Apolo H100 report (`foodguard/docs/apolo_hpc_cache_report.md`) updated with GPU reality and plan
- [ ] Update P0 doc once cache completes and benchmarks are in
- [ ] Add benchmark results and cache coverage summary to docs

## Actionable Next Steps (now)
- [ ] Let shard jobs complete; track cache count to 21,657
- [ ] Run progress script + integrity checks
- [ ] Execute embedding-only benchmark on 50 genomes from cached set
- [ ] Generate splits and prep for fine-tuning
- [ ] Run full viz with all genomes once cache is near-complete
