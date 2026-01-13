#!/bin/bash
# SBATCH script to run the proteome dropout validation notebook on Apolo-3 (CPU).

#SBATCH --job-name=fg_dropout_validation
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/fg_dropout_validation.out
#SBATCH --error=logs/fg_dropout_validation.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-}"
if [ -z "$REPO_ROOT" ]; then
  if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    if [ "$(basename "$SLURM_SUBMIT_DIR")" = "scripts" ]; then
      REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
    else
      REPO_ROOT="${SLURM_SUBMIT_DIR}"
    fi
  else
    REPO_ROOT="/home/rsg-jcorre38/Jay_Proyects/FoodGuardAI/Bacformer"
  fi
fi

PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
NOTEBOOK="${REPO_ROOT}/notebooks/foodguard_simulated_drift_study.ipynb"

MANIFEST="${MANIFEST:-${PROJECT_ROOT}/data/genome_statistics/gbff_manifest_full_20251020_123050_h100.tsv}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/.cache/esm2_h100}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/foodguard/analysis}"
MODEL_ID="${MODEL_ID:-facebook/esm2_t12_35M_UR50D}"
MAX_PROT_SEQ_LEN="${MAX_PROT_SEQ_LEN:-1024}"

# Activate environment (adjust if using a different conda module/env)
module load miniconda3/25.5.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate FoodGuard310

cd "$REPO_ROOT"
### mkdir -p logs

export FOODGUARD_MANIFEST="$MANIFEST"
export FOODGUARD_CACHE_DIR="$CACHE_DIR"
export FOODGUARD_OUTPUT_DIR="$OUTPUT_DIR"
export FOODGUARD_MODEL_ID="$MODEL_ID"
export FOODGUARD_MAX_PROT_SEQ_LEN="$MAX_PROT_SEQ_LEN"
export FOODGUARD_NOTEBOOK="$NOTEBOOK"

# Normalize notebook cell indentation to avoid stray tab/whitespace errors.
python - <<'PY'
import json
import os
import uuid
from pathlib import Path

nb_path = Path(os.environ["FOODGUARD_NOTEBOOK"])
nb = json.loads(nb_path.read_text())
for cell in nb.get("cells", []):
    if "id" not in cell:
        cell["id"] = uuid.uuid4().hex
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    src = src.expandtabs(4)
    lines = [line.rstrip() for line in src.splitlines()]
    cell["source"] = [line + "\n" for line in lines]
nb_path.write_text(json.dumps(nb, indent=2))
print(f"Normalized notebook: {nb_path}")
PY

python -m jupyter nbconvert --execute --to notebook --inplace \
  "$NOTEBOOK"
