# FoodGuard AI – MVP TODO (Revised based on Strategic Review)

**Last Updated:** October 8, 2025
**Status:** Post-Strategic Review - Aligned with 4-month realistic roadmap

This living checklist tracks the remaining work to bring the FoodGuard MVP from partial scaffold to a production-ready prototype. All tasks have been re-prioritized based on the comprehensive strategic review findings.

**Key Changes from Original TODO:**
- ESM-2 caching elevated from "optimization" to **P0 MANDATORY** (blocking)
- Fine-tuned pathogen classifier added as **P0 BLOCKING** (not previously listed)
- Novelty detection downgraded to "Beta" status with validation requirements
- Timeline extended from 3 to 4 months with realistic effort estimates

Use this alongside GitHub Issues/Milestones:
- Create one issue per checkbox, label by area (pipeline, calibration, novelty, evidence, retrieval, api, tooling, data, perf, ops) and priority (P0/P1/P2)
- Assign to owners, link PRs, and update status
- Keep this file concise and high-level; the issue holds details

---

## P0 – Now (Critical Path - BLOCKING for MVP Launch)

**Timeline: 12-14 weeks**
**Goal: Functional pathogen detection with calibrated PS, basic ES, Beta NS**

### 1. ESM-2 Cache Infrastructure (MANDATORY) - **Week 1-2**
- [ ] **Cache Pre-population**: Download FDA GenomeTrakr dataset (~5K genomes)
  - Effort: 3-5 days
  - Output: ~29 GB cached protein embeddings
  - Script: `scripts/fg_prepopulate_cache.py`
- [ ] **Pipeline Integration**: Add `cache_dir` to `PipelineConfig` and wire to `add_protein_embeddings`
  - Effort: 2-3 days
  - Files: `foodguard/config.py`, `foodguard/pipeline.py`
  - Verify content-hash keying, no recompute on hits
- [ ] **Cache Monitoring**: Implement cache hit/miss tracking and logging
  - Effort: 1-2 days
  - Metrics: hit rate (target: 80%+), cache size, avg lookup time
  - Files: `foodguard/utils/cache.py`, `foodguard/pipeline.py`
- [ ] **Validation**: Verify <30s latency on cache hit, log warning on cache miss
  - Effort: 1 day
  - Test with 10+ genomes, measure timing breakdown
- [ ] **Documentation**: Update README with cache setup instructions
  - Reference: `ai_docs/Bacformer_MVP_Implementation_Guide_REVISED.md` Section 5A

**Owner:** DevOps + ML Engineer
**Blocking:** All downstream tasks (can't test without cache)
**Success Criteria:**
- GenomeTrakr cache pre-populated (5K genomes, 29 GB)
- Cache hit latency <5s, total pipeline <30s
- Cache hit rate >80% on test set

---

### 2. Fine-Tuned Pathogen Classifier - **Week 3-9** (5-7 weeks)
- [ ] **Dataset Curation**: Download and label FDA GenomeTrakr food pathogens
  - Effort: 1 week
  - Species: Salmonella, E. coli, Listeria, Clostridium, Bacillus, Staphylococcus
  - Target: 5,000+ labeled genomes with train/val/test splits
  - Output: `data/genometrakr_labeled.csv`
- [ ] **Data Preprocessing**: Preprocess all genomes and cache embeddings
  - Effort: 1-2 weeks
  - Use ESM-2 cache from Task #1
  - Script: `scripts/fg_prepare_training_data.py`
- [ ] **Fine-Tuning**: Train `BacformerForGenomeClassification` on food pathogens
  - Effort: 2-3 weeks
  - Config: 1e-5 LR, 10 epochs, A100 GPU
  - Target: 95% balanced accuracy on held-out test set
  - Script: `scripts/fg_finetune_pathogen_classifier.py`
- [ ] **Validation**: Test on held-out set, generate confusion matrix and per-pathogen metrics
  - Effort: 3-5 days
  - Report: F1, precision, recall per pathogen class
  - Output: `models/bacformer-foodguard-v1.0/` + validation report
- [ ] **Model Checkpoint**: Save fine-tuned model to HuggingFace Hub or local registry
  - Output: `macwiatrak/bacformer-foodguard-v1.0`

**Owner:** ML Engineer (primary), Bioinformatician (data curation)
**Blocking:** Calibration (Task #3), API deployment (Task #7)
**Dependencies:** Task #1 (cache must be ready)
**Success Criteria:**
- 95% balanced accuracy on held-out test (95% CI: [90%, 98%])
- Per-pathogen F1 > 0.90 for all target species
- Model checkpoint saved and loadable via `from_pretrained()`

---

### 3. Per-Pathogen Calibration - **Week 10-12** (3 weeks)
- [ ] **Calibration Module**: Implement `foodguard/calibration.py` with Platt/Isotonic scaling
  - Effort: 1 week
  - Functions: `fit_calibrator()`, `save_calibrator()`, `load_calibrator()`, `calibrate_score()`
  - Support: Per-pathogen calibration curves
- [ ] **Threshold Selection**: Define decision thresholds for 3 postures (precision/balanced/recall)
  - Effort: 3-5 days
  - Use validation set to select thresholds maximizing F1 under constraints
  - Output: `models/calibration/thresholds_fg-food-v1.json`
- [ ] **ECE Reporting**: Calculate Expected Calibration Error on calibrated scores
  - Effort: 2-3 days
  - Ensure calibrated probabilities are well-calibrated (ECE < 0.05)
- [ ] **Integration**: Wire calibration into `foodguard/pipeline.py`
  - Effort: 2-3 days
  - Load calibrators on initialization
  - Apply to PS before CRS computation
- [ ] **Unit Tests**: Add `tests/foodguard/test_calibration.py`
  - Effort: 2-3 days
  - Test: save/load, score calibration, threshold application

**Owner:** ML Engineer
**Blocking:** Production API deployment (Task #7)
**Dependencies:** Task #2 (fine-tuned model must exist)
**Success Criteria:**
- ECE < 0.05 on validation set for all pathogens
- Thresholds defined for 3 postures with documented precision/recall trade-offs
- Calibrators saved and loadable

---

### 4. VFDB/CARD Integration for Evidence Extraction - **Week 10-13** (3-4 weeks, parallel with Task #3)
- [ ] **Database Download**: Download VFDB and CARD databases
  - Effort: 1 day
  - Files: `data/vfdb.fasta`, `data/card.fasta`
  - Script: `scripts/fg_download_vfdb_card.py`
- [ ] **Protein Matching**: Implement BLAST or HMM-based alignment for protein matching
  - Effort: 1-2 weeks
  - Tool: BLAST+ or HMMER
  - Module: `foodguard/evidence.py` function `match_proteins_to_vfdb()`
- [ ] **Attention Extraction**: Extract attention weights from Bacformer outputs
  - Effort: 3-5 days
  - Aggregate across layers/heads, map to protein indices
  - Function: `extract_attention_scores()`
- [ ] **Evidence Scoring**: Combine attention weights with VFDB matches
  - Effort: 3-5 days
  - ES = Σ (attention_weight × match_score) for top-k proteins
  - Function: `compute_evidence_score()`
- [ ] **Pipeline Integration**: Wire evidence extraction into `foodguard/pipeline.py`
  - Effort: 2-3 days
  - Replace empty evidence list with VFDB-matched proteins
  - Include: protein_id, feature name, match score, attention weight
- [ ] **Unit Tests**: Add `tests/foodguard/test_evidence.py`
  - Effort: 2-3 days

**Owner:** Bioinformatician (primary), ML Engineer (attention extraction)
**Blocking:** Production API with full response schema
**Dependencies:** None (can run in parallel with Tasks #2-3)
**Success Criteria:**
- VFDB/CARD databases indexed and searchable
- Evidence list populated for 90%+ of true positive pathogens
- Attention weights correctly extracted and normalized

---

### 5. Threat Embedding Library (Baseline NS) - **Week 10-12** (2-3 weeks, parallel)
- [ ] **Reference Genome Curation**: Select 100 known food-related pathogens
  - Effort: 3-5 days
  - Include: Salmonella, E. coli, Listeria variants
  - Output: `data/threat_library_genomes.csv`
- [ ] **Embedding Generation**: Preprocess and embed all reference genomes
  - Effort: 1 week
  - Use cache from Task #1
  - Output: `data/threat_library_embeddings.npy` (100 × 480)
- [ ] **NS Scorer Implementation**: Update `foodguard/novelty.py` with cosine similarity baseline
  - Effort: 2-3 days
  - Load threat library on pipeline init
  - Compute NS as embedding deviation (already implemented, just needs library)
- [ ] **Beta Gating**: Add "Beta" status to NS outputs
  - Effort: 1 day
  - Always set `"status": "beta"` in response
  - Add `analyst_review_required: true` if NS > 0.5
- [ ] **Documentation**: Mark NS as "Research Preview" in API docs

**Owner:** Bioinformatician (curation), ML Engineer (implementation)
**Blocking:** None (NS is Beta, not critical for MVP launch)
**Dependencies:** Task #1 (cache), Task #2 (embeddings)
**Success Criteria:**
- Threat library of 100 reference genomes embedded
- NS computed as embedding deviation from library
- All NS outputs marked as "beta" with analyst review flags

---

### 6. CRS Fusion & Risk Scoring - **Week 13** (1 week)
- [ ] **Finalize Weights**: Review and document CRS fusion weights in `foodguard/risk.py`
  - Effort: 2-3 days
  - Current: (0.6, 0.1, 0.3) for recall_high
  - Validate on labeled risk dataset if available
- [ ] **Posture Presets**: Load posture configs from `foodguard/config.py`
  - Effort: 1 day
  - Ensure precision/balanced/recall presets are well-defined
- [ ] **Audit Logging**: Add CRS computation logging
  - Effort: 2-3 days
  - Log: input scores (PS, NS, ES), weights, output CRS, posture
- [ ] **Unit Tests**: Add `tests/foodguard/test_risk.py`
  - Effort: 2 days
  - Test: weight application, clamping, posture switching

**Owner:** ML Engineer
**Blocking:** None (CRS already implemented, just polish)
**Dependencies:** Tasks #2, #3, #4 (PS, ES must be calibrated)
**Success Criteria:**
- CRS weights documented with rationale
- Posture presets validated on test data
- Audit logs include full CRS computation trace

---

### 7. API & Deployment - **Week 14** (1 week)
- [ ] **Switch to Real Pipeline**: Disable stub mode by default in `api/app.py`
  - Effort: 1 day
  - Add env var `FOODGUARD_USE_STUB` (default: false)
- [ ] **Environment Variables**: Add `MODEL_PATH`, `CACHE_DIR`, `POSTURE` env configs
  - Effort: 1 day
  - Document in README and `.env.example`
- [ ] **Input Validation**: Add file type validation, size limits, error handling
  - Effort: 2-3 days
  - Max file size: 50 MB
  - Supported: .gbff, .gbff.gz, .gff, .gff.gz
- [ ] **Response Schema**: Update to match revised strategy (full schema with metadata)
  - Effort: 1 day
  - Include: classification, novelty (Beta), evidence, risk, calibration, metadata, timing
- [ ] **API Tests**: Add `tests/foodguard/test_api_contract.py`
  - Effort: 2-3 days
  - Test: schema validation, error handling, latency requirements
- [ ] **Docker**: Containerize API with model bundle
  - Effort: 2-3 days
  - Include: models, calibrators, cache mount points
  - Document health checks and startup sequence

**Owner:** Backend Engineer, DevOps
**Blocking:** Beta deployment
**Dependencies:** All P0 tasks (1-6) must be complete
**Success Criteria:**
- API returns full response schema matching revised strategy
- Latency <30s on cache hit (95th percentile)
- Docker image builds and runs with env configs
- API tests pass with 100% coverage on contract

---

### 8. End-to-End Testing & Validation - **Week 14** (parallel with Task #7)
- [ ] **Smoke Tests**: Extend `tests/foodguard/test_pipeline_smoke.py` to validate all response fields
  - Effort: 2-3 days
  - Verify: classification, novelty, evidence, risk, calibration, metadata
- [ ] **Latency Benchmarking**: Measure end-to-end latency on 50 test genomes
  - Effort: 1-2 days
  - Report: mean, median, 95th percentile
  - Validate: 95th percentile <30s on cache hit
- [ ] **Accuracy Validation**: Run classifier on held-out test set
  - Effort: 2-3 days
  - Report: confusion matrix, per-pathogen F1, calibration curve
- [ ] **Beta Customer Testing**: Deploy to 1-2 beta partners for real-world validation
  - Effort: 1 week (overlaps with Month 2)
  - Collect: 50+ real samples, feedback on accuracy/latency

**Owner:** QA Engineer, ML Engineer
**Blocking:** Public beta launch
**Dependencies:** Task #7 (API must be deployed)
**Success Criteria:**
- All smoke tests pass
- 95th percentile latency <30s on 50 test genomes
- 95% balanced accuracy validated on held-out test
- Beta customer feedback collected and documented

---

## P1 – Next (Enable Scale & Usability)

**Timeline: Months 2-3**
**Goal: Production-ready system with validated novelty detection and retrieval**

### 9. MLM Perplexity Scorer for NS (2-3 weeks)
- [ ] Implement masked-token perplexity computation in `foodguard/novelty.py`
- [ ] Compute baseline perplexity on training corpus
- [ ] Add perplexity component to NS (weighted fusion with embedding deviation)
- [ ] Validate on synthetic constructs (iGEM registry)
- [ ] Unit tests: `tests/foodguard/test_novelty.py`

**Owner:** ML Engineer
**Dependencies:** Task #2 (fine-tuned model)
**Success Criteria:** Perplexity computed, validated on 50+ synthetic sequences

---

### 10. Attention Anomaly Detector (3-4 weeks)
- [ ] Extract baseline attention patterns from training corpus
- [ ] Implement outlier detection (z-score or IQR-based)
- [ ] Add attention anomaly component to NS
- [ ] Validate on engineered sequences

**Owner:** ML Engineer
**Dependencies:** Task #4 (attention extraction)
**Success Criteria:** Attention anomalies detected, validated on 50+ engineered sequences

---

### 11. FAISS Retrieval System (3-5 weeks)
- [ ] Build FAISS index for 1.3M genome embeddings
- [ ] Implement `foodguard/retrieval.py` with top-k search API
- [ ] Link genome embeddings to metadata (accession, species, outbreak ID)
- [ ] Return `similar_genomes` list with distances and metadata
- [ ] Index persistence and versioning

**Owner:** ML Engineer
**Dependencies:** Task #2 (genome embeddings available)
**Success Criteria:** FAISS index built, top-5 retrieval 95%+ genus-level accuracy

---

### 12. Novelty Detection Validation (4-6 weeks)
- [ ] Curate 100+ synthetic constructs (iGEM registry, adversarial examples)
- [ ] Label as engineered/novel vs. natural
- [ ] Validate multi-component NS (embedding + perplexity + attention)
- [ ] Measure: precision, recall, F1 on synthetic dataset
- [ ] Target: 85%+ detection rate

**Owner:** ML Engineer, Bioinformatician
**Dependencies:** Tasks #9, #10 (NS components implemented)
**Success Criteria:** NS validated on 100+ synthetic sequences, 85%+ F1

---

### 13. Utility Scripts (2-3 weeks total)
- [ ] `scripts/fg_cache_rebuild.py`: Warm cache from genome manifest
- [ ] `scripts/fg_benchmark_latency.py`: Measure and report pipeline latency
- [ ] `scripts/fg_calibrate.py`: Fit calibrators on new datasets
- [ ] `scripts/fg_build_retrieval_index.py`: Rebuild FAISS index
- [ ] `scripts/fg_manifest_generate.py`: Generate genome manifests for datasets

**Owner:** ML Engineer, DevOps
**Success Criteria:** All scripts documented and tested

---

### 14. Learned CRS Combiner (2 weeks)
- [ ] Collect labeled risk dataset (pathogen vs. non-pathogen)
- [ ] Train logistic regression on (PS, NS, ES) → risk label
- [ ] Replace hand-tuned weights with learned weights
- [ ] Validate on held-out risk data
- [ ] Add confidence intervals via bootstrap

**Owner:** ML Engineer
**Dependencies:** Tasks #3, #12 (calibration + NS validation)
**Success Criteria:** Learned combiner outperforms hand-tuned weights on test set

---

### 15. Performance Documentation (1 week)
- [ ] Add latency budget to Implementation Guide (measured on dev hardware)
- [ ] Document cache hit rate impact on SLA
- [ ] Benchmark on different hardware (T4, A100, CPU)
- [ ] Create performance tuning guide

**Owner:** ML Engineer, DevOps
**Success Criteria:** Documentation published with measured benchmarks

---

### 16. Docker & Deployment (2-3 weeks)
- [ ] Containerize API with model bundle (Task #7 completion)
- [ ] Document environment variables and health checks
- [ ] Add docker-compose for local development
- [ ] Deploy to staging environment
- [ ] Load testing (100+ concurrent requests)

**Owner:** DevOps
**Dependencies:** Task #7 (API ready)
**Success Criteria:** Docker image < 5 GB, API handles 100+ concurrent requests

---

### 17. Documentation Cross-Links (1 week)
- [ ] Update README with FoodGuard modules overview
- [ ] Link revised strategy and implementation guides
- [ ] Reference data curation guide from README
- [ ] Add API documentation (OpenAPI spec)

**Owner:** Technical Writer, ML Engineer
**Success Criteria:** All docs linked, OpenAPI spec published

---

## P2 – Later (Optimization & Integrations)

**Timeline: Months 4-6+**
**Goal: Edge deployment, multi-task models, regulatory compliance**

### 18. Model Quantization (INT8) - 2-3 weeks
- [ ] Quantize Bacformer to INT8 using PyTorch dynamic quantization
- [ ] Validate accuracy retention (>98% on test set)
- [ ] Measure speedup (target: 2x inference, 4x memory reduction)
- [ ] ONNX export for TensorRT optimization

**Success Criteria:** Quantized model <500 MB, 2x speedup, >98% accuracy retention

---

### 19. Knowledge Distillation (6-8 weeks)
- [ ] Define student architecture (240 hidden, 3 layers, 4 heads)
- [ ] Train student on teacher outputs (distillation loss)
- [ ] Validate accuracy retention (target: 90-95%)
- [ ] Measure speedup (target: 5-10x)

**Success Criteria:** Student model 50M params, 90%+ accuracy retention, 5x+ speedup

---

### 20. Multi-Task Fine-Tuning (4-6 weeks)
- [ ] Add AMR prediction head (CARD database)
- [ ] Add serotype classification head (MLST/typing schemes)
- [ ] Joint training with shared encoder
- [ ] Validate on held-out AMR/serotype datasets

**Success Criteria:** AMR prediction >90% accuracy, serotype >85% accuracy

---

### 21. Jetson Edge Deployment (3-4 weeks)
- [ ] Export quantized model to ONNX + TensorRT
- [ ] Deploy to NVIDIA Jetson AGX Orin
- [ ] Cloud-backed cache for ESM-2 embeddings
- [ ] Fallback to cloud on cache miss
- [ ] Measure on-device latency (target: 20-30s with cache)

**Success Criteria:** Jetson deployment <50W power, 20-30s latency with cache

---

### 22. Dashboard UI (4-6 weeks)
- [ ] Minimal web UI for genome uploads
- [ ] Risk score visualization (gauges, charts)
- [ ] Evidence heatmap (attention overlays)
- [ ] Historical trending and batch results

**Success Criteria:** Functional dashboard deployed, 95%+ user satisfaction

---

### 23. LIMS Integration (3-4 weeks)
- [ ] Define webhook/CSV connectors for LabWare, STARLIMS
- [ ] Implement adapters for common LIMS formats
- [ ] Test with 2-3 beta customers using LIMS

**Success Criteria:** LIMS adapters deployed, 2+ customers integrated

---

### 24. Security & Compliance (4-6 weeks)
- [ ] Audit log hardening (input/model hashes, signatures)
- [ ] ISO 17025 quality management setup
- [ ] AOAC validation dossier preparation
- [ ] FDA pre-submission (Q-Sub) materials

**Success Criteria:** AOAC submission ready, FDA Q-Sub scheduled

---

## Current Status (Post-Strategic Review)

### ✅ Implemented Components
- Pipeline orchestration (`foodguard/pipeline.py`)
- ESM-2 protein embedding with content-hash caching (`bacformer/pp/embed_prot_seqs.py`)
- Bacformer input conversion with special tokens
- Generic genome classification head (requires fine-tuning)
- Basic risk scoring (weighted PS/NS/ES → CRS)
- Novelty baseline (embedding deviation, needs threat library)
- Configuration system with posture presets
- API skeleton (`api/app.py`)
- Smoke tests (`tests/foodguard/test_pipeline_smoke.py`)

### 🔴 Missing P0 Components (Blocking MVP Launch)
1. **ESM-2 Cache Pre-population** (Week 1-2, ~29 GB, MANDATORY)
2. **Fine-tuned Pathogen Classifier** (Week 3-9, 5-7 weeks, BLOCKING)
3. **Per-Pathogen Calibration** (Week 10-12, 3 weeks, BLOCKING)
4. **VFDB/CARD Integration** (Week 10-13, 3-4 weeks, BLOCKING)
5. **Threat Embedding Library** (Week 10-12, 2-3 weeks)
6. **Production API Deployment** (Week 14)

### Timeline to MVP: 14 weeks (3.5 months)
- **Month 1 (Weeks 1-4):** Cache infrastructure + dataset curation
- **Month 2 (Weeks 5-8):** Fine-tune classifier + begin calibration/VFDB
- **Month 3 (Weeks 9-12):** Complete calibration/VFDB + threat library
- **Month 4 (Weeks 13-14):** Integration, testing, beta deployment

---

## References

**Strategy & Implementation:**
- Revised MVP Strategy: `ai_docs/Bacformer_MVP_Strategy_REVISED.md`
- Revised Implementation Guide: `ai_docs/Bacformer_MVP_Implementation_Guide_REVISED.md`
- Strategic Review Findings: `ai_docs/FoodGuard_Strategic_Review_Findings.md`

**Data & Preprocessing:**
- Data Curation Guide: `ai_docs/FoodGuard_Public_Data_Curation_Guide.md`
- Bacformer preprocessing: `bacformer/pp/preprocess.py`
- Embedding utilities: `bacformer/pp/embed_prot_seqs.py`

**Current Implementation:**
- Pipeline: `foodguard/pipeline.py`
- Configuration: `foodguard/config.py`
- Risk scoring: `foodguard/risk.py`
- Novelty detection: `foodguard/novelty.py`
- API: `api/app.py`

---

## Next Immediate Actions (Week 1)

1. **[DevOps]** Download FDA GenomeTrakr dataset (~5K genomes)
2. **[ML Engineer]** Run cache pre-population script (create if needed)
3. **[Bioinformatician]** Begin labeling GenomeTrakr genomes by pathogen class
4. **[ML Engineer]** Validate cache hit <5s latency on 10 test genomes
5. **[Backend Engineer]** Add cache_dir to PipelineConfig and wire to pipeline
6. **[Team Lead]** Schedule kickoff meeting to review revised roadmap

---

**End of TODO - Revised based on Strategic Review Findings**
