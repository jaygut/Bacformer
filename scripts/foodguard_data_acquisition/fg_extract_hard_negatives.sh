#!/usr/bin/env bash

# Simple script to extract hard-negative GBFF files from zip archives
# This script processes one zip file at a time with basic error handling

# Auto-detect the correct data directory
if [[ -d "/home/jcorre38/Jay_Proyects/FoodGuardAI/data/genometrakr" ]]; then
  DATA_DIR="/home/jcorre38/Jay_Proyects/FoodGuardAI/data/genometrakr"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)"
  REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." &>/dev/null && pwd -P)"
  DATA_DIR="$REPO_ROOT/data/genometrakr"
fi

echo "Data directory: $DATA_DIR"
echo "Starting extraction of hard-negative taxa..."
echo

# Process each zip file individually
cd "$DATA_DIR" || exit 1

# Listeria innocua
if [[ -f "listeria_innocua_gbff.zip" ]]; then
  echo "Extracting listeria_innocua..."
  rm -rf listeria_innocua_gbff_extracted
  mkdir -p listeria_innocua_gbff_extracted
  unzip -q listeria_innocua_gbff.zip -d listeria_innocua_gbff_extracted
  echo "✓ Listeria innocua extracted"
else
  echo "✗ listeria_innocua_gbff.zip not found"
fi

# Bacillus subtilis
if [[ -f "bacillus_subtilis_gbff.zip" ]]; then
  echo "Extracting bacillus_subtilis..."
  rm -rf bacillus_subtilis_gbff_extracted
  mkdir -p bacillus_subtilis_gbff_extracted
  unzip -q bacillus_subtilis_gbff.zip -d bacillus_subtilis_gbff_extracted
  echo "✓ Bacillus subtilis extracted"
else
  echo "✗ bacillus_subtilis_gbff.zip not found"
fi

# Citrobacter freundii
if [[ -f "citrobacter_freundii_gbff.zip" ]]; then
  echo "Extracting citrobacter_freundii..."
  rm -rf citrobacter_freundii_gbff_extracted
  mkdir -p citrobacter_freundii_gbff_extracted
  unzip -q citrobacter_freundii_gbff.zip -d citrobacter_freundii_gbff_extracted
  echo "✓ Citrobacter freundii extracted"
else
  echo "✗ citrobacter_freundii_gbff.zip not found"
fi

# Citrobacter koseri
if [[ -f "citrobacter_koseri_gbff.zip" ]]; then
  echo "Extracting citrobacter_koseri..."
  rm -rf citrobacter_koseri_gbff_extracted
  mkdir -p citrobacter_koseri_gbff_extracted
  unzip -q citrobacter_koseri_gbff.zip -d citrobacter_koseri_gbff_extracted
  echo "✓ Citrobacter koseri extracted"
else
  echo "✗ citrobacter_koseri_gbff.zip not found"
fi

# E. coli (non-pathogenic)
if [[ -f "ecoli_nonpathogenic_gbff.zip" ]]; then
  echo "Extracting ecoli_nonpathogenic (this will take several minutes)..."
  rm -rf ecoli_all_strains_gbff_extracted
  mkdir -p ecoli_all_strains_gbff_extracted
  unzip -q ecoli_nonpathogenic_gbff.zip -d ecoli_all_strains_gbff_extracted
  echo "✓ E. coli (non-pathogenic) extracted"
else
  echo "✗ ecoli_nonpathogenic_gbff.zip not found"
fi

# Escherichia fergusonii
if [[ -f "escherichia_fergusonii_gbff.zip" ]]; then
  echo "Extracting escherichia_fergusonii..."
  rm -rf escherichia_fergusonii_gbff_extracted
  mkdir -p escherichia_fergusonii_gbff_extracted
  unzip -q escherichia_fergusonii_gbff.zip -d escherichia_fergusonii_gbff_extracted
  echo "✓ Escherichia fergusonii extracted"
else
  echo "✗ escherichia_fergusonii_gbff.zip not found"
fi

echo
echo "Extraction complete! Summary:"
echo

# Count GBFF files in each directory
for dir in *_gbff_extracted; do
  if [[ -d "$dir" ]]; then
    count=$(find "$dir" -name "*.gbff*" | wc -l)
    echo "$dir: $count GBFF files"
  fi
done

echo
echo "All extractions finished. Ready for manifest creation."