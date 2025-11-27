#!/bin/bash
# SBATCH script to populate the ESM-2 cache on Apolo-3 accel node (H100 x2)

#SBATCH --job-name=esm2_cache
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/esm2_cache_%a.out
#SBATCH --error=logs/esm2_cache_%a.err

set -euo pipefail

MANIFEST="${MANIFEST:-/home/rsg-jcorre38/Jay_Proyects/FoodGuardAI/data/genome_statistics/gbff_manifest_full_20251020_123050_h100.tsv}"
CACHE_DIR="${CACHE_DIR:-/home/rsg-jcorre38/Jay_Proyects/FoodGuardAI/Bacformer/.cache/esm2_h100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_PROT_SEQ_LEN="${MAX_PROT_SEQ_LEN:-1024}"
MODEL_ID="${MODEL_ID:-facebook/esm2_t12_35M_UR50D}"

# Activate environment (adjust if using a different conda module/env)
module load miniconda3/25.5.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate FoodGuard310

mkdir -p "$(dirname "$CACHE_DIR")" "$CACHE_DIR" logs

echo "Starting shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_MAX} on $(hostname)"
echo "Manifest: $MANIFEST"
echo "Cache dir: $CACHE_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Max prot seq len: $MAX_PROT_SEQ_LEN"

python scripts/pilot_populate_esm2_cache.py \
  --manifest "$MANIFEST" \
  --cache-dir "$CACHE_DIR" \
  --batch-size "$BATCH_SIZE" \
  --max-prot-seq-len "$MAX_PROT_SEQ_LEN" \
  --model-id "$MODEL_ID" \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --shard-count "$((SLURM_ARRAY_TASK_MAX+1))"
