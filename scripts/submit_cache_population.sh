#!/bin/bash
#SBATCH --job-name=esm2_cache_%a
#SBATCH --output=logs/esm2_cache_%a.out
#SBATCH --error=logs/esm2_cache_%a.err
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00

set -euo pipefail

FG_HOME=${FG_HOME:-$PWD}
REPO=${REPO:-"$FG_HOME"}
MANIFEST=${MANIFEST:-"$FG_HOME/data/genome_statistics/gbff_manifest_full_20251020_123050.tsv"}
CACHE_DIR=${CACHE_DIR:-"$FG_HOME/data/esm2_cache"}
MODEL_ID=${MODEL_ID:-"facebook/esm2_t12_35M_UR50D"}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-1024}
BATCH_SIZE=${BATCH_SIZE:-64}
SHARD_COUNT=${SHARD_COUNT:-8}

mkdir -p "$CACHE_DIR" logs

cd "$REPO"

python scripts/pilot_populate_esm2_cache.py \
  --manifest "$MANIFEST" \
  --cache-dir "$CACHE_DIR" \
  --batch-size "$BATCH_SIZE" \
  --max-prot-seq-len "$MAX_SEQ_LEN" \
  --model-id "$MODEL_ID" \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --shard-count "$SHARD_COUNT"
