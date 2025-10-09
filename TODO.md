# FoodGuard AI – MVP TODO

This living checklist tracks the remaining work to bring the FoodGuard MVP from partial scaffold to a production-ready prototype. Use it alongside GitHub Issues/Milestones:
- Create one issue per checkbox, label by area (pipeline, calibration, novelty, evidence, retrieval, api, tooling, data, perf, ops) and priority (P0/P1/P2).
- Assign to owners, link PRs, and update status. Keep this file concise and high-level; the issue holds details.

## P0 – Now (Critical Path)
- [ ] Pipeline: add `cache_dir` to `PipelineConfig` and wire ESM-2 caching via `add_protein_embeddings` (content-hash keys, no recompute on hits).
- [ ] Calibration: implement `foodguard/calibration.py` with Platt/Isotonic, fit/save/load per-pathogen tables; add ECE reporting and unit tests.
- [ ] CRS Fusion: finalize weighting and thresholds in `foodguard/risk.py`; load posture presets from config; add audit log entries.
- [ ] Evidence: add `foodguard/evidence.py` to extract attention overlays from Bacformer outputs and map VFDB/CARD matches (local CSV/JSON adapters). Populate `evidence` block.
- [ ] Novelty (NS): use genome embedding + threat library to compute deviation; define thresholding and “Beta” gating. Stub API for MLM/perplexity for later.
- [ ] Retrieval: implement `foodguard/retrieval.py` (sklearn NearestNeighbors fallback; optional FAISS). Return `similar_genomes` with ids/distances.
- [ ] API: switch default to real pipeline (disable stub) behind env var; add `MODEL_PATH`, `CACHE_DIR`, and posture envs; input validation and error handling.
- [ ] Tests: add `tests/foodguard/test_calibration.py`, `test_risk.py`, `test_api_contract.py`; extend smoke test to validate evidence/retrieval fields when enabled.

## P1 – Next (Enable Scale & Usability)
- [ ] Scripts: `fg_cache_rebuild.py` (warm cache from manifest), `fg_benchmark_latency.py`, `fg_calibrate.py`, `fg_build_retrieval_index.py`, `fg_manifest_generate.py`.
- [ ] Manifests: `foodguard/io/manifests.py` to parse FG-Data manifests; enforce caps/QC gates; log rejects; helper to derive train/val/test splits without leakage.
- [ ] Performance docs: add latency budget notes to `ai_docs/Bacformer_MVP_Implementation_Guide.md` with measured numbers on dev slice.
- [ ] Docker: containerize API with model bundle; document envs and health checks.
- [ ] Docs cross-links: reference new FoodGuard modules from README and `ai_docs/*` (Data Curation, Strategy, Implementation Guide).

## P2 – Later (Optimization & Integrations)
- [ ] Optimization: ONNX/TensorRT export for Bacformer head/backbone; post-training INT8 for linears; evaluate speed/accuracy deltas.
- [ ] Distillation: smaller Bacformer variant for edge; benchmark vs teacher.
- [ ] Novelty (MLM): implement masked-token perplexity proxy when feasible; calibrate NS jointly with embedding deviation.
- [ ] Retrieval: switch default backend to FAISS for large libraries; add index persistence/versioning.
- [ ] Dashboard: minimal UI for uploads, scores, evidence visualization.
- [ ] LIMS: define webhook/CSV connectors; small adapters for LabWare/STARLIMS.
- [ ] Security/Compliance: audit log hardening (input/model hashes, signatures), SOP docs, and AOAC report templates.

## Current Status (Snapshot)
- Added: `foodguard/` scaffold (`pipeline`, `config`, `risk`, `novelty`, `utils/cache`), `api/app.py` skeleton, and smoke test.
- Pending: calibration module, evidence overlays, retrieval integration, cache wiring, API env/config, and broader tests.

## References
- MVP Strategy & Implementation Guides under `ai_docs/`
- Data Curation Guide: `ai_docs/FoodGuard_Public_Data_Curation_Guide.md`
- Bacformer preprocessing/embedding utilities: `bacformer/pp/`
