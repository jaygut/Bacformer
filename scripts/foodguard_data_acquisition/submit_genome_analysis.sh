#!/bin/bash
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --job-name=FoodGuard_GenomeAnalysis
#SBATCH -o genome_analysis_%N_%j.out
#SBATCH -e genome_analysis_%N_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jcorre38@eafit.edu.co

echo "=========================================="
echo "FoodGuardAI Genome Analysis & Manifest Creation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

## ACTIVATE CONDA ENVIRONMENT
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate FoodGuard

echo "Conda environment activated: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo

## NAVIGATE TO SCRIPT DIRECTORY  
SCRIPT_DIR="/home/jcorre38/Jay_Proyects/FoodGuardAI/scripts"
if [[ -d "$SCRIPT_DIR" ]]; then
    cd "$SCRIPT_DIR"
    echo "Successfully navigated to: $(pwd)"
    echo "Available scripts in current directory:"
    ls -la fg_*.sh
else
    echo "ERROR: Script directory not found: $SCRIPT_DIR"
    echo "Available directories:"
    ls -la /home/jcorre38/Jay_Proyects/FoodGuardAI/scripts/
    exit 1
fi

echo "=========================================="
echo "STEP 1: Genome Statistics Analysis"
echo "=========================================="
echo "Starting comprehensive genome statistics analysis..."
echo "This will analyze ~21,000 GBFF files across all collections"
echo

# Run genome statistics analysis
if bash ./fg_analyze_genome_stats.sh; then
    echo "✓ Genome statistics analysis completed successfully"
    echo "Statistics files generated in: /home/jcorre38/Jay_Proyects/FoodGuardAI/data/genome_statistics/"
else
    echo "✗ Genome statistics analysis failed"
    exit 1
fi

echo
echo "=========================================="
echo "STEP 2: Unified Manifest Creation"
echo "=========================================="
echo "Creating unified labeled manifest for model training..."
echo "Combining pathogenic + non-pathogenic genomes with labels"
echo

# Run unified manifest creation
if bash ./fg_create_unified_manifest.sh; then
    echo "✓ Unified manifest created successfully"
    echo "Manifest file: /home/jcorre38/Jay_Proyects/FoodGuardAI/data/manifests/gbff_manifest_full.tsv"
else
    echo "✗ Unified manifest creation failed"
    exit 1
fi

echo
echo "=========================================="
echo "STEP 3: Results Validation"
echo "=========================================="

# Validate manifest file
MANIFEST_FILE="/home/jcorre38/Jay_Proyects/FoodGuardAI/data/manifests/gbff_manifest_full.tsv"

if [[ -f "$MANIFEST_FILE" ]]; then
    echo "Manifest file validation:"
    echo "- File exists: ✓"
    echo "- Total lines: $(wc -l < "$MANIFEST_FILE")"
    echo "- Header preview:"
    head -3 "$MANIFEST_FILE"
    echo
    echo "- Dataset balance:"
    echo "  Pathogenic genomes: $(awk -F'\t' 'NR>1 && $3==1' "$MANIFEST_FILE" | wc -l)"
    echo "  Non-pathogenic genomes: $(awk -F'\t' 'NR>1 && $3==0' "$MANIFEST_FILE" | wc -l)"
    echo
    echo "- Species distribution:"
    awk -F'\t' 'NR>1 {print $4}' "$MANIFEST_FILE" | sort | uniq -c | sort -nr
else
    echo "✗ Manifest file not found: $MANIFEST_FILE"
    exit 1
fi

# Check statistics files
STATS_DIR="/home/jcorre38/Jay_Proyects/FoodGuardAI/data/genome_statistics"
echo
echo "Statistics files generated:"
for file in "$STATS_DIR"/*.csv; do
    if [[ -f "$file" ]]; then
        echo "  ✓ $(basename "$file"): $(wc -l < "$file") lines"
    fi
done

echo
echo "=========================================="
echo "JOB COMPLETION SUMMARY"
echo "=========================================="
echo "End time: $(date)"
echo "Job duration: $SECONDS seconds"
echo
echo "✓ Genome statistics analysis: COMPLETED"
echo "✓ Unified manifest creation: COMPLETED"
echo "✓ Results validation: COMPLETED"
echo
echo "Next steps:"
echo "1. Review statistics files in: $STATS_DIR"
echo "2. Validate manifest: $MANIFEST_FILE"
echo "3. Begin ESM-2 cache population"
echo "4. Create train/validation/test splits"
echo "5. Start Bacformer fine-tuning"
echo
echo "FoodGuardAI data acquisition pipeline: COMPLETE! 🎉"
