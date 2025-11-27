#!/bin/bash
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --job-name=FoodGuard_Python_Processor
#SBATCH -o genome_processor_%N_%j.out
#SBATCH -e genome_processor_%N_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jcorre38@eafit.edu.co

echo "=========================================="
echo "FoodGuardAI Python Genome Processor"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "=========================================="

## ACTIVATE CONDA ENVIRONMENT
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate FoodGuard

echo "Conda environment activated: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo

## CHECK PYTHON DEPENDENCIES
echo "Checking Python dependencies..."
python -c "
import sys
try:
    import pandas as pd
    print(f'✓ pandas: {pd.__version__}')
except ImportError:
    print('✗ pandas not found')
    sys.exit(1)

try:
    import numpy as np
    print(f'✓ numpy: {np.__version__}')
except ImportError:
    print('✗ numpy not found')
    sys.exit(1)

try:
    from Bio import SeqIO
    import Bio
    print(f'✓ biopython: {Bio.__version__}')
except ImportError:
    print('✗ biopython not found - installing...')
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'biopython'])

try:
    from tqdm import tqdm
    print('✓ tqdm available')
except ImportError:
    print('✗ tqdm not found - installing...')
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'tqdm'])
"

if [ $? -ne 0 ]; then
    echo "ERROR: Python dependencies check failed"
    exit 1
fi

echo
echo "=========================================="
echo "Running FoodGuardAI Genome Processor"
echo "=========================================="

## DEFINE PATHS
SCRIPT_PATH="/home/jcorre38/Jay_Proyects/FoodGuardAI/scripts/foodguard_genome_processor.py"
DATA_DIR="/home/jcorre38/Jay_Proyects/FoodGuardAI/data/genometrakr"
OUTPUT_DIR="/home/jcorre38/Jay_Proyects/FoodGuardAI/data/genome_statistics"

echo "Script path: $SCRIPT_PATH"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Workers: $SLURM_CPUS_PER_TASK"
echo

## VERIFY PATHS EXIST
if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "ERROR: Python script not found: $SCRIPT_PATH"
    echo "Available files in scripts directory:"
    ls -la /home/jcorre38/Jay_Proyects/FoodGuardAI/scripts/
    exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    exit 1
fi

## CREATE OUTPUT DIRECTORY
mkdir -p "$OUTPUT_DIR"

## RUN THE PYTHON PROCESSOR
echo "Starting genome processing with $SLURM_CPUS_PER_TASK parallel workers..."
echo

python "$SCRIPT_PATH" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --workers "$SLURM_CPUS_PER_TASK"

PYTHON_EXIT_CODE=$?

echo
echo "=========================================="
echo "JOB COMPLETION SUMMARY"
echo "=========================================="
echo "End time: $(date)"
echo "Job duration: $SECONDS seconds"
echo "Python exit code: $PYTHON_EXIT_CODE"

if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "✓ Genome processing completed successfully!"
    echo
    echo "Output files generated in: $OUTPUT_DIR"
    echo "Generated files:"
    ls -la "$OUTPUT_DIR"/*.csv "$OUTPUT_DIR"/*.tsv "$OUTPUT_DIR"/*.json 2>/dev/null || echo "No output files found"
    
    echo
    echo "Next steps:"
    echo "1. Review detailed statistics and manifest files"
    echo "2. Validate dataset balance and quality metrics"
    echo "3. Begin ESM-2 cache population"
    echo "4. Create train/validation/test splits"
    echo "5. Start Bacformer fine-tuning"
    echo
    echo "FoodGuardAI genome processing pipeline: COMPLETE! 🎉"
else
    echo "✗ Genome processing failed with exit code: $PYTHON_EXIT_CODE"
    echo "Check the error log for details: genome_processor_*.err"
    exit $PYTHON_EXIT_CODE
fi
