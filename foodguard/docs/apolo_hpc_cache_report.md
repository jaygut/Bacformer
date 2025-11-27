# Apolo HPC ESM-2 Cache Experiments (Nov 2025)

## Purpose and Scope
This report documents the P0 ESM-2 cache preflight on Apolo, executed to:
- Validate that the FoodGuard ESM-2 embedding pipeline runs end-to-end in the Apolo environment.
- Identify hardware/software blockers to full cache population (21,657 genomes).
- Capture reproducible commands, environment pinning, and observed artifacts for auditability.
- Provide a go/no-go recommendation and migration plan to suitable GPU resources.

The preflight is part of the broader P0 “ESM-2 Cache Infrastructure — Warm, Integrate, Validate” plan: pre-populate embeddings, wire the cache into the pipeline, benchmark hit-rate/latency, and unlock fine-tuning. Apolo was evaluated as a potential execution site.

## Hardware and Cluster Context (Apolo)
- GPU partition: `accel` only (Tesla K80, 11 GB), driver 470.42.01 (CUDA 11.4).
- No V100/A100/T4 partitions visible (`sinfo` shows only `accel` for GPUs; AVAIL_FEATURES is null).
- Kepler K80 + driver 470 cannot run PyTorch ≥2.x CUDA wheels that Bacformer requires.

## Environment Used
- Conda env: `FoodGuard310` (Python 3.10).
- For CPU smoke:
  - `torch==2.2.2` (CPU wheel)  
  - `numpy==1.26.4` (avoids NumPy 2.x ABI issues)  
  - `pip install -e . --no-deps`
- GPU attempts:
  - `torch==2.9.1+cu128` (initial) → driver too old, CUDA unavailable.
  - `torch==1.12.1+cu113` (driver-compatible) → lacks `scaled_dot_product_attention`; Bacformer import fails.

## Actions and Commands (Chronological)
1) Verified GPU partition: `sinfo -o "%P %G %N %f %T"` → only `accel` (K80).
2) CPU-only smoke (code sanity check; very slow):
   ```bash
   pip uninstall -y torch torchvision torchaudio
   pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
   pip install "numpy==1.26.4"
   pip install -e . --no-deps

   CUDA_VISIBLE_DEVICES="" python scripts/pilot_populate_esm2_cache.py \
     --manifest /home/jcorre38/Jay_Proyects/FoodGuardAI/data/genome_statistics/gbff_manifest_full_20251020_123050.tsv \
     --cache-dir "$PWD/.cache/esm2_cpu" \
     --batch-size 1 \
     --max-prot-seq-len 512 \
     --limit 1
   ```
3) Result: CPU run completed (warnings about faESM/pooler expected) and wrote 1 cache file.
   ```bash
   ls -1 $PWD/.cache/esm2_cpu/prot_emb_*.pt | wc -l  # => 1
   ```
4) Cache artifact inspection (CPU run):
   - Location (example): `$PWD/.cache/esm2_cpu/prot_emb_<key>.pt`
   - Size check:
     ```bash
     ls -lh $PWD/.cache/esm2_cpu/prot_emb_*.pt
     ```
   - Load and summarize shape:
     ```bash
     python - <<'PY'
     import torch, glob
     paths = glob.glob(".cache/esm2_cpu/prot_emb_*.pt")
     print("files:", len(paths))
     if paths:
         obj = torch.load(paths[0], map_location="cpu")
         if isinstance(obj, list):
             print("example contig count:", len(obj))
             if obj and obj[0]:
                 print("example contig[0] protein count:", len(obj[0]))
                 print("example embedding dim:", len(obj[0][0]))
             else:
                 print("empty contig or embedding?")
         else:
             print("unexpected type:", type(obj))
     PY
     ```
   - Observed (Nov 26, 2025, Apolo CPU run):
     ```
     files: 1
     example contig count: 2
     example contig[0] protein count: 4511
     example embedding dim: 480
     ```

## Findings
- Apolo K80 nodes cannot run Bacformer with GPU acceleration: torch ≥2.x CUDA wheels drop Kepler support; older torch builds miss required ops.
- CPU-only execution works for a single-genome smoke but is far too slow for population-scale caching.

## Assessment and Decision
- **Go/No-Go:** No-go for production cache population on Apolo GPUs. Hardware/driver are incompatible with required torch 2.x CUDA builds and Bacformer ops.
- **Viable use:** Limited to CPU sanity checks (single-genome smoke) to validate code paths; not viable for throughput.

## Migration Plan for Production Cache
1) **Target hardware:** Access V100/T4/A100 GPUs with driver ≥525 (cluster or cloud).
2) **Environment on new hardware:** Python 3.10; `pip install torch==2.2.x` (cu118/cu121 matching driver); `pip install -e ".[dev]"`.
3) **Preflight:** Repeat 2-genome preflight with `--batch-size 8`, `--max-prot-seq-len 1024`, verify cache write + timing + cache hit on rerun.
4) **Sharded run:** Use `scripts/submit_cache_population.sh` with consistent `cache_dir`, `max-prot-seq-len=1024`, shard across available GPUs; monitor with `generate_cache_progress.py`.
5) **Validation:** Run `scripts/pilot_benchmark_pipeline.py --limit 50 --report ...` to confirm hit-rate/latency; run integrity spot-checks on random cache files; maintain progress JSON.

## Due Diligence Notes
- Recorded all dependency pins and failure modes (torch/driver mismatch, Kepler support drop, NumPy ABI warning).
- Ensured cache key compatibility guidance (max sequence length, model ID, single cache_dir) is retained for future runs.
- Captured the single CPU artifact for audit (contigs, protein count, embedding dim).
