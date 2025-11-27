#!/usr/bin/env bash
set -Eeuo pipefail

# Create Unified Labeled Manifest for Bacformer Fine-Tuning
# This script creates the final TSV manifest file that combines all pathogenic 
# and non-pathogenic genomes with proper labels for training.
#
# Output: data/manifests/gbff_manifest_full.tsv
#
# Columns:
#   - genome_id (string): Unique identifier (e.g., GCF_...)
#   - gbff_path (string): Absolute path to the GBFF file
#   - is_pathogenic (integer): 1 for pathogenic, 0 for non-pathogenic
#   - pathogen_class (string): Human-readable category
#   - source_taxon_id (integer): The NCBI taxon ID for traceability
#
# Usage:
#   ./scripts/foodguard_data_acquisition/fg_create_unified_manifest.sh

# Resolve repo root relative to this script
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." &>/dev/null && pwd -P)"
DATA_DIR=${DATA_DIR:-${REPO_ROOT}/data/genometrakr}
MANIFEST_DIR=${MANIFEST_DIR:-${REPO_ROOT}/data/manifests}
OUTPUT_FILE="$MANIFEST_DIR/gbff_manifest_full.tsv"

log() { printf "[%s] %s\n" "$(date +"%F %T")" "$*"; }

# Pathogenic genome directories (from fg_select_gbff_from_zips.sh)
PATHOGENIC_DIRS=(
  "salmonella_gbff_selected:1:Salmonella:590"
  "ecoli_o157h7_gbff_selected:1:E_coli_O157H7:83334"
  "listeria_monocytogenes_gbff_selected:1:L_monocytogenes:1639"
)

# Non-pathogenic genome directories (from fg_extract_hard_negatives.sh)
NON_PATHOGENIC_DIRS=(
  "listeria_innocua_gbff_extracted:0:L_innocua:1640"
  "bacillus_subtilis_gbff_extracted:0:B_subtilis:1423"
  "citrobacter_freundii_gbff_extracted:0:C_freundii:546"
  "citrobacter_koseri_gbff_extracted:0:C_koseri:547"
  "ecoli_all_strains_gbff_extracted:0:E_coli_nonpathogenic:562"
  "escherichia_fergusonii_gbff_extracted:0:E_fergusonii:564"
)

extract_genome_id_from_path() {
  local gbff_path=$1
  # Extract genome ID from path like: .../GCF_000005825.2_ASM582v2/GCF_000005825.2_ASM582v2_genomic.gbff
  local filename
  filename=$(basename "$gbff_path")
  # Remove _genomic.gbff or .gbff suffix and extract the GCF/GCA part
  echo "$filename" | sed -E 's/(_genomic)?\.gbff(\.gz)?$//' | grep -oE '^G[CF][AF]_[0-9]+\.[0-9]+'
}

is_o157h7_strain() {
  local gbff_path=$1
  # Check if the path or filename contains O157:H7 or O157H7 indicators
  if echo "$gbff_path" | grep -qi "o157"; then
    return 0  # True - this is O157:H7
  fi
  return 1  # False - not O157:H7
}

process_genome_directory() {
  local dir_name=$1
  local is_pathogenic=$2
  local pathogen_class=$3
  local taxon_id=$4
  local dir_path="$DATA_DIR/$dir_name"
  
  if [[ ! -d "$dir_path" ]]; then
    log "WARNING: Directory not found: $dir_path"
    return 0
  fi
  
  local count=0
  local o157h7_filtered=0
  
  # Find all GBFF files in the directory
  while IFS= read -r -d '' gbff_file; do
    # Special handling for E. coli non-pathogenic: filter out O157:H7 strains
    if [[ "$pathogen_class" == "E_coli_nonpathogenic" ]] && is_o157h7_strain "$gbff_file"; then
      ((o157h7_filtered++))
      continue
    fi
    
    local genome_id
    genome_id=$(extract_genome_id_from_path "$gbff_file")
    
    if [[ -z "$genome_id" ]]; then
      log "WARNING: Could not extract genome ID from: $gbff_file"
      continue
    fi
    
    # Write to manifest: genome_id, gbff_path, is_pathogenic, pathogen_class, source_taxon_id
    printf "%s\t%s\t%d\t%s\t%d\n" \
      "$genome_id" \
      "$gbff_file" \
      "$is_pathogenic" \
      "$pathogen_class" \
      "$taxon_id" >> "$OUTPUT_FILE"
    
    ((count++))
  done < <(find "$dir_path" -type f -iname "*.gbff*" -print0)
  
  log "$pathogen_class: Added $count genomes to manifest"
  if [[ "$o157h7_filtered" -gt 0 ]]; then
    log "$pathogen_class: Filtered out $o157h7_filtered O157:H7 strains"
  fi
  
  return 0
}

main() {
  log "Creating unified labeled manifest for Bacformer fine-tuning"
  log "Output file: $OUTPUT_FILE"
  
  mkdir -p "$MANIFEST_DIR"
  
  # Create header
  echo -e "genome_id\tgbff_path\tis_pathogenic\tpathogen_class\tsource_taxon_id" > "$OUTPUT_FILE"
  
  local total_pathogenic=0
  local total_non_pathogenic=0
  
  # Process pathogenic genomes
  log "------------------------------------------------------------"
  log "Processing pathogenic genomes..."
  for entry in "${PATHOGENIC_DIRS[@]}"; do
    local dir_name is_pathogenic pathogen_class taxon_id
    IFS=':' read -r dir_name is_pathogenic pathogen_class taxon_id <<<"$entry"
    
    local before_count
    before_count=$(wc -l < "$OUTPUT_FILE")
    
    process_genome_directory "$dir_name" "$is_pathogenic" "$pathogen_class" "$taxon_id"
    
    local after_count
    after_count=$(wc -l < "$OUTPUT_FILE")
    local added=$((after_count - before_count))
    total_pathogenic=$((total_pathogenic + added))
  done
  
  # Process non-pathogenic genomes
  log "------------------------------------------------------------"
  log "Processing non-pathogenic genomes..."
  for entry in "${NON_PATHOGENIC_DIRS[@]}"; do
    local dir_name is_pathogenic pathogen_class taxon_id
    IFS=':' read -r dir_name is_pathogenic pathogen_class taxon_id <<<"$entry"
    
    local before_count
    before_count=$(wc -l < "$OUTPUT_FILE")
    
    process_genome_directory "$dir_name" "$is_pathogenic" "$pathogen_class" "$taxon_id"
    
    local after_count
    after_count=$(wc -l < "$OUTPUT_FILE")
    local added=$((after_count - before_count))
    total_non_pathogenic=$((total_non_pathogenic + added))
  done
  
  # Final summary
  local total_genomes=$((total_pathogenic + total_non_pathogenic))
  local ratio
  if [[ "$total_non_pathogenic" -gt 0 ]]; then
    ratio=$(echo "scale=2; $total_pathogenic / $total_non_pathogenic" | bc -l)
  else
    ratio="undefined"
  fi
  
  log "------------------------------------------------------------"
  log "Manifest Creation Complete!"
  log "Output: $OUTPUT_FILE"
  log "Summary:"
  log "  - Pathogenic genomes: $total_pathogenic"
  log "  - Non-pathogenic genomes: $total_non_pathogenic"
  log "  - Total genomes: $total_genomes"
  log "  - Pathogenic:Non-pathogenic ratio: $ratio:1"
  log ""
  log "Next steps:"
  log "  1. Validate manifest integrity: head -5 $OUTPUT_FILE"
  log "  2. Check balance: cut -f3 $OUTPUT_FILE | sort | uniq -c"
  log "  3. Populate ESM-2 cache for all genomes in manifest"
  log "  4. Create train/validation/test splits"
  log "  5. Begin model fine-tuning"
}

main "$@"
