#!/usr/bin/env bash

# Filter out O157:H7 strains from the non-pathogenic E. coli dataset
# This script identifies and removes pathogenic E. coli strains that may have
# been included in the general E. coli taxon download (562)

# Auto-detect the correct data directory
if [[ -d "/home/jcorre38/Jay_Proyects/FoodGuardAI/data/genometrakr" ]]; then
  DATA_DIR="/home/jcorre38/Jay_Proyects/FoodGuardAI/data/genometrakr"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)"
  REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." &>/dev/null && pwd -P)"
  DATA_DIR="$REPO_ROOT/data/genometrakr"
fi

ECOLI_DIR="$DATA_DIR/ecoli_all_strains_gbff_extracted"
FILTERED_DIR="$DATA_DIR/ecoli_nonpathogenic_gbff_filtered"

echo "E. coli Strain Filtering Script"
echo "==============================="
echo "Source: $ECOLI_DIR"
echo "Target: $FILTERED_DIR"
echo

if [[ ! -d "$ECOLI_DIR" ]]; then
  echo "ERROR: E. coli extraction directory not found: $ECOLI_DIR"
  exit 1
fi

# Create filtered directory
rm -rf "$FILTERED_DIR"
mkdir -p "$FILTERED_DIR"

# Counters
total_genomes=0
o157h7_filtered=0
kept_genomes=0

echo "Scanning E. coli genomes for O157:H7 strains..."

# Find all GBFF files in the E. coli directory
while IFS= read -r -d '' gbff_file; do
  ((total_genomes++))
  
  # Extract genome directory and ID
  genome_dir=$(dirname "$gbff_file")
  genome_id=$(basename "$genome_dir")
  
  # Check if this is an O157:H7 strain by examining the GBFF content
  is_o157h7=false
  
  # Method 1: Check filename/path for O157 indicators
  if echo "$gbff_file" | grep -qi "o157"; then
    is_o157h7=true
  fi
  
  # Method 2: Check GBFF content for O157:H7 serotype indicators
  if [[ "$is_o157h7" == false ]] && [[ -f "$gbff_file" ]]; then
    # Look for O157:H7 serotype indicators in the first 50 lines (organism info)
    if head -50 "$gbff_file" | grep -qi -E "(O157|serotype.*157|serogroup.*157)"; then
      is_o157h7=true
    fi
  fi
  
  # Method 3: Check for H7 flagellar antigen
  if [[ "$is_o157h7" == false ]] && [[ -f "$gbff_file" ]]; then
    if head -50 "$gbff_file" | grep -qi -E "(H7|flagellar.*7)"; then
      # Only mark as O157:H7 if we also found O157 indicators
      if head -50 "$gbff_file" | grep -qi "O157"; then
        is_o157h7=true
      fi
    fi
  fi
  
  if [[ "$is_o157h7" == true ]]; then
    ((o157h7_filtered++))
    echo "  FILTERED: $genome_id (O157:H7 strain detected)"
  else
    # Copy the entire genome directory to filtered location
    target_dir="$FILTERED_DIR/$(basename "$(dirname "$genome_dir")")/$(basename "$genome_dir")"
    mkdir -p "$(dirname "$target_dir")"
    cp -r "$genome_dir" "$(dirname "$target_dir")/"
    ((kept_genomes++))
    
    if (( kept_genomes % 500 == 0 )); then
      echo "  Progress: $kept_genomes non-pathogenic genomes kept..."
    fi
  fi
  
done < <(find "$ECOLI_DIR" -name "*.gbff*" -type f -print0)

echo
echo "Filtering Summary:"
echo "=================="
echo "Total E. coli genomes processed: $total_genomes"
echo "O157:H7 strains filtered out: $o157h7_filtered"
echo "Non-pathogenic genomes kept: $kept_genomes"
echo "Filtering efficiency: $(( (o157h7_filtered * 100) / total_genomes ))% pathogenic strains removed"
echo
echo "Filtered dataset location: $FILTERED_DIR"
echo "Ready for manifest creation with clean non-pathogenic E. coli set!"
