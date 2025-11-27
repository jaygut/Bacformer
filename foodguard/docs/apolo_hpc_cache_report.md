# Apolo HPC ESM-2 Cache Experiments (Nov 2025)

## Goal
Execute the P0 ESM-2 cache preflight on Apolo to confirm end-to-end viability and identify blockers for full cache population.

## Hardware and Cluster Context
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

## Actions and Commands
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

## Conclusion
- Full ESM-2 cache population cannot be performed on Apolo’s K80/470 stack.
- Apolo is usable only for CPU sanity checks (small limits) to validate code paths.

## Next Steps (for production cache)
1) Acquire access to newer GPUs (V100/T4/A100) with driver ≥525 on another partition/cluster/cloud.
2) On that hardware: install a matching torch 2.x CUDA wheel (cu118/cu121), rerun the 2-genome preflight, then launch sharded cache population.
3) Keep `max-prot-seq-len=1024` and consistent `cache_dir` to preserve cache key compatibility; run `generate_cache_progress.py` for tracking.
