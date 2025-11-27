# Blueprint: Unified Labeled Manifest for Bacformer Fine-Tuning

**Version:** 2.0
**Date:** 2025-10-20
**Status:** ✅ COMPLETED - Manifest Generated and Validated

## 1. Overview

This document provides the blueprint for the unified, labeled manifest file that serves as the source of truth for all subsequent data processing steps, including ESM-2 cache population, dataset splitting (train/validation/test), and model fine-tuning. The manifest provides a reproducible, comprehensive, and correctly labeled dataset of pathogenic and non-pathogenic ("hard negative") bacterial genomes for the FoodGuard AI system.

## 2. Manifest File Specification

### 2.1. Current Manifest (PRODUCTION)

-   **Filename**: `gbff_manifest_full_20251020_123050.tsv`
-   **Location**: `/Users/jaygut/Documents/Side_Projects/Bacformer/data/genome_statistics/`
-   **Format**: Tab-Separated Values (TSV) with a header row
-   **Encoding**: UTF-8
-   **Total Genomes**: 21,657 bacterial genomes
-   **Pathogenic**: 13,000 genomes (60.0%)
-   **Non-Pathogenic**: 8,657 genomes (40.0%)
-   **Species Coverage**: 9 distinct bacterial species
-   **File Size**: ~2.4 MB
-   **Validation Status**: ✅ Verified and validated

### 2.2. Column Structure

The manifest contains the following columns:

| Column Name        | Data Type | Description                                                                                                                               | Example                                                              |
| ------------------ | --------- | ----------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `genome_id`        | `string`  | A unique identifier for the genome, typically the assembly accession (e.g., GCF_...). Extracted from the directory path.                 | `GCF_001652385.2`                                                    |
| `gbff_path`        | `string`  | The absolute file path to the corresponding GenBank Flat File (`.gbff` or `.gbff.gz`).                                                    | `/home/jcorre38/Jay_Proyects/FoodGuardAI/data/genometrakr/salmonella_gbff_selected/ncbi_dataset/data/GCF_001652385.2/genomic.gbff` |
| `is_pathogenic`    | `integer` | A binary label indicating the class. **1** for pathogenic, **0** for non-pathogenic (hard negative). This is the primary training target. | `1`                                                                  |
| `species`          | `string`  | A human-readable species identifier. For pathogens, this is the species name. For negatives, this identifies the source group.            | `Salmonella_enterica`                                                |
| `taxon_id`         | `integer` | The NCBI taxon ID from which the genome was originally sourced. Used for traceability and phylogenetic analysis.                          | `590`                                                                |
| `pathogenicity`    | `string`  | Human-readable pathogenicity label. Either `pathogenic` or `non_pathogenic`. Redundant with `is_pathogenic` but useful for filtering.     | `pathogenic`                                                         |

## 3. Data Sources and Labeling Strategy

This section maps the source directories to their corresponding labels in the manifest.

### 3.1. Pathogenic Genomes (is_pathogenic = 1)

**Total:** 13,000 genomes

| Species                  | Collection Directory                        | Taxon ID | Count  | Mean Size (Mb) | Mean GC (%) | Mean CDS |
| ------------------------ | ------------------------------------------- | -------- | ------ | -------------- | ----------- | -------- |
| `Salmonella_enterica`    | `salmonella_gbff_selected/`                 | `590`    | 7,000  | 4.87           | 52.05       | 4,697    |
| `E_coli_O157H7`          | `ecoli_o157h7_gbff_selected/`               | `83334`  | 1,500  | 5.43           | 50.33       | 5,367    |
| `L_monocytogenes`        | `listeria_monocytogenes_gbff_selected/`     | `1639`   | 4,500  | 2.97           | 37.90       | 2,940    |

**Combined Statistics:**
- **Mean Genome Size:** 4.28 Mb
- **Mean GC Content:** 46.95%
- **Mean CDS Count:** 4,166 genes
- **Total Base Pairs:** 55.6 Gb

### 3.2. Non-Pathogenic / Hard Negative Genomes (is_pathogenic = 0)

**Total:** 8,657 genomes

| Species                   | Collection Directory                    | Taxon ID | Count  | Mean Size (Mb) | Mean GC (%) | Mean CDS |
| ------------------------- | --------------------------------------- | -------- | ------ | -------------- | ----------- | -------- |
| `E_coli_nonpathogenic`    | `ecoli_nonpathogenic_gbff_filtered/`    | `562`    | 4,312  | 5.13           | 50.56       | 4,922    |
| `B_subtilis`              | `bacillus_subtilis_gbff_extracted/`     | `1423`   | 2,361  | 4.14           | 43.52       | 4,272    |
| `C_koseri`                | `citrobacter_koseri_gbff_extracted/`    | `547`    | 897    | 5.02           | 54.96       | 4,740    |
| `L_innocua`               | `listeria_innocua_gbff_extracted/`      | `1640`   | 449    | 2.89           | 37.22       | 2,832    |
| `E_fergusonii`            | `escherichia_fergusonii_gbff_extracted/`| `564`    | 438    | 4.78           | 49.84       | 4,564    |
| `C_freundii`              | `citrobacter_freundii_gbff_extracted/`  | `546`    | 200    | 5.33           | 51.73       | 5,098    |

**Combined Statistics:**
- **Mean Genome Size:** 4.72 Mb
- **Mean GC Content:** 48.40%
- **Mean CDS Count:** 4,603 genes
- **Total Base Pairs:** 40.8 Gb

### 3.3. Overall Dataset Statistics

- **Total Genomes:** 21,657
- **Total Base Pairs:** 96.4 Gb
- **Mean Genome Size:** 4.45 Mb (range: 2.51 - 7.62 Mb)
- **Mean GC Content:** 47.53% (range: 37.2% - 55.0%)
- **Mean CDS Count:** 4,341 genes (range: 2,832 - 5,367)
- **Mean Contig Count:** 44.2 contigs per genome
- **Mean N50 Length:** 2.39 Mb
- **Pathogenic:Non-pathogenic Ratio:** 1.5:1 (60:40)

## 4. Generation Process

### 4.1. Script Used

The manifest was generated using the Python script:
`scripts/foodguard_data_acquisition/foodguard_genome_processor.py`

**Key Features:**
- Fast GBFF parsing using BioPython
- Comprehensive genome statistics extraction
- Parallel processing with 4 workers
- Automatic label assignment based on directory structure
- Validation and integrity checks

### 4.2. Execution Command

```bash
python scripts/foodguard_data_acquisition/foodguard_genome_processor.py \
  --data-dir /home/jcorre38/Jay_Proyects/FoodGuardAI/data/genometrakr \
  --output-dir data/genome_statistics \
  --workers 4
```

### 4.3. Processing Summary

```
📊 Dataset Overview:
   Total genomes processed: 21,657
   Pathogenic genomes: 13,000
   Non-pathogenic genomes: 8,657
   Species diversity: 9 species
   Collections processed: 9

🧬 Genome Size Statistics:
   Mean size: 4,453,673 bp (4.45 Mb)
   Size range: 2,512,999 - 7,622,298 bp

🔬 GC Content Statistics:
   Mean GC content: 47.53%

🧪 Gene Content Statistics:
   Mean CDS count: 4,341
   Mean total genes: 8,783
```

## 5. Usage and Downstream Integration

### 5.1. ESM-2 Cache Population

**Script:** `bacformer/pp/embed_prot_seqs.py`

```python
from bacformer.pp.preprocess import preprocess_genome_assembly
from bacformer.pp.embed_prot_seqs import load_plm, add_protein_embeddings
import pandas as pd

# Load manifest
manifest = pd.read_csv(
    'data/genome_statistics/gbff_manifest_full_20251020_123050.tsv',
    sep='\t'
)

# Load ESM-2 model once
plm, tok = load_plm('facebook/esm2_t12_35M_UR50D', model_type='esm2')

# Process each genome
for idx, row in manifest.iterrows():
    genome_id = row['genome_id']
    gbff_path = row['gbff_path']

    # Preprocess genome
    pre = preprocess_genome_assembly(gbff_path)

    # Generate and cache embeddings
    result = add_protein_embeddings(
        row={'protein_sequence': pre['protein_sequence']},
        model=plm,
        tokenizer=tok,
        prot_seq_col='protein_sequence',
        output_col='embeddings',
        cache_dir='.cache/esm2_genometrakr',
        cache_overwrite=False,
        model_id='facebook/esm2_t12_35M_UR50D',
    )

    if (idx + 1) % 100 == 0:
        print(f"Processed {idx + 1}/{len(manifest)} genomes")
```

**Expected Runtime:**
- Single GPU (NVIDIA A100): ~40-60 hours for full dataset
- 4 GPUs parallel: ~10-15 hours
- Cache size: ~29-35 GB total

### 5.2. Dataset Splitting

**Script:** Train/validation/test split with stratification

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load manifest
manifest = pd.read_csv(
    'data/genome_statistics/gbff_manifest_full_20251020_123050.tsv',
    sep='\t'
)

print(f"Total genomes: {len(manifest)}")
print(f"Class distribution:\n{manifest['species'].value_counts()}")

# Stratified split (70% train, 15% val, 15% test)
train_df, temp_df = train_test_split(
    manifest,
    test_size=0.3,
    stratify=manifest['species'],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df['species'],
    random_state=42
)

# Save splits
train_df.to_csv('data/splits/train_split.tsv', sep='\t', index=False)
val_df.to_csv('data/splits/val_split.tsv', sep='\t', index=False)
test_df.to_csv('data/splits/test_split.tsv', sep='\t', index=False)

print(f"\nSplit sizes:")
print(f"  Train: {len(train_df)} genomes ({len(train_df)/len(manifest)*100:.1f}%)")
print(f"  Val:   {len(val_df)} genomes ({len(val_df)/len(manifest)*100:.1f}%)")
print(f"  Test:  {len(test_df)} genomes ({len(test_df)/len(manifest)*100:.1f}%)")

# Verify stratification
print(f"\nTrain class distribution:")
print(train_df['is_pathogenic'].value_counts())
print(f"\nVal class distribution:")
print(val_df['is_pathogenic'].value_counts())
print(f"\nTest class distribution:")
print(test_df['is_pathogenic'].value_counts())
```

**Expected Output:**
```
Split sizes:
  Train: 15,159 genomes (70.0%)
  Val:   3,249 genomes (15.0%)
  Test:  3,249 genomes (15.0%)

Train class distribution:
1    9,100
0    6,059
```

### 5.3. Model Training

**Script:** Fine-tune Bacformer for binary pathogenicity classification

```python
from bacformer.modeling import BacformerForGenomeClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd

# Load training split
train_df = pd.read_csv('data/splits/train_split.tsv', sep='\t')
val_df = pd.read_csv('data/splits/val_split.tsv', sep='\t')

# Convert to HuggingFace datasets (after embedding - see Section 5.1)
train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

# Load pre-trained Bacformer with classification head
model = BacformerForGenomeClassification.from_pretrained(
    'macwiatrak/bacformer',
    num_labels=2,  # Binary: pathogenic vs non-pathogenic
    problem_type='single_label_classification',
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./models/bacformer-foodguard-pathogen-v1.0',
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    warmup_ratio=0.1,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    logging_steps=50,
    save_total_limit=3,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

# Train
trainer.train()

# Save final model
model.save_pretrained('models/bacformer-foodguard-pathogen-v1.0')
```

**Target Performance:**
- Balanced Accuracy: >95%
- F1 Score: >93%
- Precision (pathogenic): >90%
- Recall (pathogenic): >95%

## 6. Validation

### 6.1. Automated Validation Checks

```bash
# Check total row count (should be 21,658 including header)
wc -l data/genome_statistics/gbff_manifest_full_20251020_123050.tsv
# Output: 21658

# Check label balance
cut -f3 data/genome_statistics/gbff_manifest_full_20251020_123050.tsv | tail -n +2 | sort | uniq -c
# Output:
#  8657 0
# 13000 1

# Check for missing values in critical columns
awk -F'\t' 'NR>1 && ($1=="" || $2=="" || $3=="") {print "Row " NR " has missing values"}' \
  data/genome_statistics/gbff_manifest_full_20251020_123050.tsv

# Verify file paths exist (sample check)
awk -F'\t' 'NR==2 {print $2}' data/genome_statistics/gbff_manifest_full_20251020_123050.tsv | \
  xargs -I {} sh -c 'if [ -f "{}" ]; then echo "✅ Path exists"; else echo "❌ Path not found"; fi'
```

### 6.2. Validation Results

**✅ All validation checks passed:**
- Total genome count: 21,657 ✓
- Pathogenic count: 13,000 ✓
- Non-pathogenic count: 8,657 ✓
- No missing values in critical columns ✓
- All genome_id fields are valid accessions ✓
- Label balance ratio: 1.5:1 (within acceptable range) ✓

### 6.3. Data Quality Metrics

**Assembly Quality:**
- Mean contig count: 44.2 (acceptable for draft assemblies)
- Mean N50: 2.39 Mb (good contiguity)
- All genomes have complete CDS annotations ✓

**Species Coverage:**
- Pathogenic: 3 species (Salmonella, E. coli O157:H7, Listeria monocytogenes)
- Non-pathogenic: 6 species (well-balanced hard negatives)
- Phylogenetic diversity: High (GC content range: 37-55%)

**Genomic Characteristics:**
- Genome size range: 2.51 - 7.62 Mb (within bacterial norms)
- CDS count range: 2,832 - 5,367 (appropriate for genome sizes)
- No extreme outliers detected ✓

## 7. Next Steps: From Data to Deployment Pipeline

### 7.1. ESM-2 Cache Population (P0 - BLOCKING)

**Goal:** Generate protein embeddings for all 21,657 genomes using Facebook's ESM-2 transformer

**Timeline:** 2-3 weeks (with GPU parallelization)

**Script:** `scripts/populate_esm2_cache.py`

```python
#!/usr/bin/env python3
"""
ESM-2 Cache Population Script
Generates and caches protein embeddings for all genomes in the manifest.
"""

import pandas as pd
from pathlib import Path
from bacformer.pp.preprocess import preprocess_genome_assembly
from bacformer.pp.embed_prot_seqs import load_plm, add_protein_embeddings
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

def main():
    # Configuration
    manifest_path = 'data/genome_statistics/gbff_manifest_full_20251020_123050.tsv'
    cache_dir = '.cache/esm2_genometrakr'
    model_id = 'facebook/esm2_t12_35M_UR50D'

    # Load manifest
    manifest = pd.read_csv(manifest_path, sep='\t')
    print(f"📊 Processing {len(manifest)} genomes...")

    # Load ESM-2 model once
    print(f"🔧 Loading ESM-2 model: {model_id}")
    plm, tok = load_plm(model_id, model_type='esm2')

    # Create cache directory
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Process genomes with progress bar
    success_count = 0
    error_count = 0

    for idx, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Embedding genomes"):
        try:
            genome_id = row['genome_id']
            gbff_path = row['gbff_path']

            # Preprocess genome
            pre = preprocess_genome_assembly(gbff_path)

            # Generate and cache embeddings
            result = add_protein_embeddings(
                row={'protein_sequence': pre['protein_sequence'], 'genome_id': genome_id},
                model=plm,
                tokenizer=tok,
                prot_seq_col='protein_sequence',
                output_col='embeddings',
                cache_dir=cache_dir,
                cache_overwrite=False,
                model_id=model_id,
            )

            success_count += 1

        except Exception as e:
            logging.error(f"Failed to process {genome_id}: {e}")
            error_count += 1
            continue

    print(f"\n✅ Cache population complete!")
    print(f"   Successful: {success_count}/{len(manifest)}")
    print(f"   Errors: {error_count}/{len(manifest)}")

    # Validate cache
    cache_files = list(Path(cache_dir).glob('prot_emb_*.pt'))
    total_size_gb = sum(f.stat().st_size for f in cache_files) / 1e9
    print(f"\n📦 Cache statistics:")
    print(f"   Files: {len(cache_files)}")
    print(f"   Total size: {total_size_gb:.2f} GB")
    print(f"   Avg per genome: {total_size_gb / len(cache_files):.3f} GB")

if __name__ == "__main__":
    main()
```

**Execution:**
```bash
# Single GPU
python scripts/populate_esm2_cache.py

# Multi-GPU (4 GPUs in parallel)
for shard in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$shard python scripts/populate_esm2_cache.py \
      --shard $shard --num-shards 4 &
done
wait
```

**Expected Resources:**
- GPU Memory: 16 GB minimum (A100 recommended)
- Disk Space: 30-35 GB for cache
- Runtime: 10-15 hours (4 GPUs) or 40-60 hours (1 GPU)

### 7.2. Train/Validation/Test Splits (P0 - BLOCKING)

**Goal:** Create stratified dataset partitions maintaining species and pathogenicity balance

**Timeline:** 1 week

**Script:** `scripts/create_splits.py`

```python
#!/usr/bin/env python3
"""
Dataset Splitting Script
Creates stratified train/val/test splits with proper balance.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def create_splits(manifest_path, output_dir, random_seed=42):
    # Load manifest
    manifest = pd.read_csv(manifest_path, sep='\t')

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Stratified split (70/15/15)
    train_df, temp_df = train_test_split(
        manifest,
        test_size=0.3,
        stratify=manifest['species'],
        random_state=random_seed
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['species'],
        random_state=random_seed
    )

    # Save splits
    train_df.to_csv(f'{output_dir}/train_split.tsv', sep='\t', index=False)
    val_df.to_csv(f'{output_dir}/val_split.tsv', sep='\t', index=False)
    test_df.to_csv(f'{output_dir}/test_split.tsv', sep='\t', index=False)

    # Print statistics
    print(f"✅ Splits created successfully!")
    print(f"\n📊 Split Sizes:")
    print(f"   Train: {len(train_df):>6,} ({len(train_df)/len(manifest)*100:>5.1f}%)")
    print(f"   Val:   {len(val_df):>6,} ({len(val_df)/len(manifest)*100:>5.1f}%)")
    print(f"   Test:  {len(test_df):>6,} ({len(test_df)/len(manifest)*100:>5.1f}%)")

    print(f"\n⚖️  Class Balance (Pathogenic/Non-Pathogenic):")
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        path_count = (split_df['is_pathogenic'] == 1).sum()
        nonpath_count = (split_df['is_pathogenic'] == 0).sum()
        print(f"   {split_name:5s}: {path_count:>5,} / {nonpath_count:>5,} ({path_count/len(split_df)*100:>5.1f}%)")

    # Verify no data leakage
    assert len(set(train_df['genome_id']) & set(val_df['genome_id'])) == 0
    assert len(set(train_df['genome_id']) & set(test_df['genome_id'])) == 0
    assert len(set(val_df['genome_id']) & set(test_df['genome_id'])) == 0
    print(f"\n✅ No data leakage detected!")

if __name__ == "__main__":
    create_splits(
        manifest_path='data/genome_statistics/gbff_manifest_full_20251020_123050.tsv',
        output_dir='data/splits',
        random_seed=42
    )
```

**Expected Output:**
```
✅ Splits created successfully!

📊 Split Sizes:
   Train: 15,159 (70.0%)
   Val:    3,249 (15.0%)
   Test:   3,249 (15.0%)

⚖️  Class Balance (Pathogenic/Non-Pathogenic):
   Train:  9,100 /  6,059 (60.0%)
   Val:    1,950 /  1,299 (60.0%)
   Test:   1,950 /  1,299 (60.0%)

✅ No data leakage detected!
```

### 7.3. Bacformer Fine-Tuning (P0 - BLOCKING)

**Goal:** Adapt pre-trained Bacformer for binary pathogenicity classification

**Timeline:** 5-7 weeks (including validation)

**Status:** 🔴 Implementation required (see `/ai_docs/Bacformer_MVP_Implementation_Guide_REVISED.md` Section 6B)

**Key Steps:**
1. Load train/val/test splits with cached embeddings
2. Convert embeddings to Bacformer inputs
3. Fine-tune BacformerForGenomeClassification (binary classification)
4. Validate on held-out test set (target: 95% balanced accuracy)
5. Save trained model to `models/bacformer-foodguard-pathogen-v1.0`

**Target Performance:**
- Balanced Accuracy: ≥95%
- F1 Score: ≥93%
- Precision (pathogenic): ≥90%
- Recall (pathogenic): ≥95%

### 7.4. Risk Component Implementation (P0/P1)

**Goal:** Integrate novelty detection, attention-based evidence, and risk fusion modules

**Components:**

1. **Pathogenicity Score (PS)** - P0 BLOCKING
   - Fine-tuned classifier + per-pathogen calibration (Platt scaling)
   - Timeline: 8 weeks total (5 weeks fine-tuning + 3 weeks calibration)
   - See Section 6C of MVP Implementation Guide

2. **Evidence Score (ES)** - P0 HIGH PRIORITY
   - VFDB/CARD database integration
   - Attention-weighted virulence factor matching
   - Timeline: 3-4 weeks
   - Status: 🔴 Not implemented

3. **Novelty Score (NS)** - P1 (Beta)
   - Embedding deviation from threat library
   - MLM perplexity scoring (optional)
   - Timeline: 2-3 weeks
   - Status: 🟡 Partial (embedding deviation implemented)

4. **Combined Risk Score (CRS)** - P0 BLOCKING
   - Weighted fusion: CRS = 0.7×PS + 0.2×ES + 0.1×NS
   - Alert level mapping (1-5 scale)
   - Timeline: 1 week
   - Status: ✅ Basic implementation exists

### 7.5. FastAPI Deployment (P0 - BLOCKING)

**Goal:** Build production-ready scoring service for real-time pathogenicity assessment

**Timeline:** 2-3 weeks (after model training complete)

**Status:** 🟡 Skeleton implemented (see `/ai_docs/Bacformer_MVP_Implementation_Guide_REVISED.md` Section 9)

**Key Endpoints:**

1. **POST /analyze** - Main genome analysis endpoint
   - Input: GenBank (.gbff) file
   - Output: Complete risk assessment (PS, NS, ES, CRS)
   - SLA: <30s with cache hit

2. **GET /health** - Health check endpoint
   - Returns model status and GPU availability

3. **GET /calibration** - Calibration metadata
   - Returns current calibration version and thresholds

**Deployment Checklist:**
- [ ] Pre-populate ESM-2 cache with GenomeTrakr dataset
- [ ] Load fine-tuned Bacformer model
- [ ] Load calibration scalers and thresholds
- [ ] Implement cache hit/miss handling
- [ ] Set up monitoring and logging
- [ ] Configure GPU inference (NVIDIA T4 minimum)

## 8. Critical Path Summary

### P0 Tasks (Blocking MVP Launch) - 12-14 Weeks Total

| Task | Timeline | Dependencies | Status |
|------|----------|--------------|--------|
| ESM-2 cache population | 2-3 weeks | Manifest ✅ | 🔴 Not started |
| Dataset splitting | 1 week | Manifest ✅ | 🔴 Not started |
| Bacformer fine-tuning | 5-7 weeks | Cache, Splits | 🔴 Not started |
| Per-pathogen calibration | 3 weeks | Fine-tuned model | 🔴 Not started |
| VFDB integration | 3-4 weeks | None | 🔴 Not started |
| FastAPI deployment | 2-3 weeks | All above | 🟡 Partial |

### P1 Tasks (Important but Not Blocking) - 4-6 Weeks

| Task | Timeline | Status |
|------|----------|--------|
| Threat embedding library | 2-3 weeks | 🟡 Partial |
| MLM perplexity scorer | 2-3 weeks | 🔴 Not started |
| End-to-end validation study | 1-2 weeks | 🔴 Not started |

## 9. Risks and Mitigations

### High-Priority Risks

1. **ESM-2 Cache Population Runtime**
   - Risk: 40-60 hours on single GPU
   - Mitigation: Use 4 GPUs in parallel (reduce to 10-15 hours)
   - Fallback: Start with subset (e.g., 5K genomes) for initial development

2. **Fine-Tuning Performance Below Target**
   - Risk: Balanced accuracy <95%
   - Mitigation: Add more training data from NCBI Pathogen Detection
   - Fallback: Use ensemble with ESM-2 zero-shot classifier

3. **Cache Storage Requirements**
   - Risk: 30-35 GB disk space needed
   - Mitigation: Ensure SSD storage available
   - Fallback: Implement LRU eviction policy

## 10. Success Criteria

### Technical Metrics

✅ **Dataset Quality** (ACHIEVED)
- [x] 21,657 genomes collected and validated
- [x] 60:40 pathogenic:non-pathogenic ratio
- [x] 9 species with phylogenetic diversity
- [x] Mean genome size 4.45 Mb (appropriate range)
- [x] All genomes have complete CDS annotations

🔴 **Model Performance** (TARGET)
- [ ] Balanced accuracy ≥95% on test set
- [ ] F1 score ≥93%
- [ ] Pathogenic precision ≥90%
- [ ] Pathogenic recall ≥95%

🔴 **Operational Metrics** (TARGET)
- [ ] Cache hit rate ≥80%
- [ ] Inference latency <30s (with cache hit)
- [ ] GPU memory usage <16 GB
- [ ] API uptime ≥99.5%

## 11. Document History

| Version | Date       | Changes                                                  | Author   |
|---------|------------|----------------------------------------------------------|----------|
| 1.0     | 2025-10-12 | Initial blueprint (proposed)                             | FoodGuard Team |
| 2.0     | 2025-10-20 | Updated with actual data, validation results, next steps | Claude + Jay |

---

**Next Action:** Execute ESM-2 cache population script (Section 7.1)

**For Questions:** See `/ai_docs/Bacformer_MVP_Implementation_Guide_REVISED.md` for detailed implementation guidance.
