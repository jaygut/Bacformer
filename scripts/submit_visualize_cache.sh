#!/bin/bash
# SBATCH script to generate PCA/UMAP dashboard from cached embeddings.
# Runs on CPU (no GPU needed). Adjust partition/time/memory as needed.

#SBATCH --job-name=viz_cache
#SBATCH --partition=longjobs
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/viz_cache.out
#SBATCH --error=logs/viz_cache.err

set -euo pipefail

MANIFEST="${MANIFEST:-/home/rsg-jcorre38/Jay_Proyects/FoodGuardAI/data/genome_statistics/gbff_manifest_full_20251020_123050_h100.tsv}"
CACHE_DIR="${CACHE_DIR:-/home/rsg-jcorre38/Jay_Proyects/FoodGuardAI/Bacformer/.cache/esm2_h100}"
OUTPUT="${OUTPUT:-logs/viz_cache_pca_umap.html}"
SAMPLE="${SAMPLE:-1000}"  # set 0 to process all genomes (may be large)
MODEL_ID="${MODEL_ID:-facebook/esm2_t12_35M_UR50D}"
MAX_PROT_SEQ_LEN="${MAX_PROT_SEQ_LEN:-1024}"

# Activate environment (adjust if using a different conda module/env)
module load miniconda3/25.5.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate FoodGuard310

# Ensure logs directory exists
mkdir -p logs

# If deps are missing, uncomment:
# pip install plotly scikit-learn umap-learn pandas numpy tqdm

python scripts/visualize_cache_embeddings.py \
  --manifest "$MANIFEST" \
  --cache-dir "$CACHE_DIR" \
  --output "$OUTPUT" \
  --sample "$SAMPLE" \
  --model-id "$MODEL_ID" \
  --max-prot-seq-len "$MAX_PROT_SEQ_LEN"
