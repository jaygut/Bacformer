#!/usr/bin/env bash

# Comprehensive Genome Statistics Analyzer for FoodGuardAI Dataset
# This script analyzes all GBFF files across pathogenic and non-pathogenic
# genome collections and generates detailed statistics for stakeholder reporting.

# Auto-detect the correct data directory
if [[ -d "/home/jcorre38/Jay_Proyects/FoodGuardAI/data/genometrakr" ]]; then
  DATA_DIR="/home/jcorre38/Jay_Proyects/FoodGuardAI/data/genometrakr"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)"
  REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." &>/dev/null && pwd -P)"
  DATA_DIR="$REPO_ROOT/data/genometrakr"
fi

OUTPUT_DIR="$DATA_DIR/../genome_statistics"
mkdir -p "$OUTPUT_DIR"

echo "FoodGuardAI Genome Statistics Analyzer"
echo "====================================="
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo

# Define genome collections with their pathogenicity status
declare -A COLLECTIONS=(
  ["salmonella_gbff_selected"]="pathogenic:Salmonella_enterica"
  ["ecoli_o157h7_gbff_selected"]="pathogenic:E_coli_O157H7"
  ["listeria_monocytogenes_gbff_selected"]="pathogenic:L_monocytogenes"
  ["listeria_innocua_gbff_extracted"]="non_pathogenic:L_innocua"
  ["bacillus_subtilis_gbff_extracted"]="non_pathogenic:B_subtilis"
  ["citrobacter_freundii_gbff_extracted"]="non_pathogenic:C_freundii"
  ["citrobacter_koseri_gbff_extracted"]="non_pathogenic:C_koseri"
  ["ecoli_all_strains_gbff_extracted"]="non_pathogenic:E_coli_nonpathogenic"
  ["escherichia_fergusonii_gbff_extracted"]="non_pathogenic:E_fergusonii"
)

# Check if filtered E. coli exists and use it instead
if [[ -d "$DATA_DIR/ecoli_nonpathogenic_gbff_filtered" ]]; then
  COLLECTIONS["ecoli_nonpathogenic_gbff_filtered"]="non_pathogenic:E_coli_nonpathogenic"
  unset COLLECTIONS["ecoli_all_strains_gbff_extracted"]
  echo "Using filtered E. coli dataset (O157:H7 strains removed)"
fi

# Output files
DETAILED_CSV="$OUTPUT_DIR/genome_detailed_statistics.csv"
SUMMARY_CSV="$OUTPUT_DIR/genome_summary_statistics.csv"
COLLECTION_CSV="$OUTPUT_DIR/collection_statistics.csv"

# Initialize CSV files
echo "genome_id,collection,pathogenicity,species,genome_size_bp,gc_content_percent,cds_count,trna_count,rrna_count,gene_count,contig_count,n50_length,assembly_level" > "$DETAILED_CSV"
echo "collection,pathogenicity,species,genome_count,avg_genome_size,std_genome_size,avg_gc_content,avg_cds_count,avg_gene_count,total_bp" > "$SUMMARY_CSV"
echo "pathogenicity,total_genomes,total_bp,avg_genome_size,avg_cds_count,species_diversity" > "$COLLECTION_CSV"

analyze_gbff_file() {
  local gbff_file="$1"
  local collection="$2"
  local pathogenicity="$3"
  local species="$4"
  
  if [[ ! -f "$gbff_file" ]]; then
    return 1
  fi
  
  # Extract genome ID from path
  local genome_id
  genome_id=$(basename "$(dirname "$gbff_file")" | sed 's/\.gbff.*//')
  
  # Initialize variables
  local genome_size=0
  local gc_count=0
  local total_bases=0
  local cds_count=0
  local trna_count=0
  local rrna_count=0
  local gene_count=0
  local contig_count=0
  local assembly_level="unknown"
  
  # Parse GBFF file for statistics
  while IFS= read -r line; do
    # Assembly level
    if [[ "$line" =~ "Assembly Level:" ]]; then
      assembly_level=$(echo "$line" | sed 's/.*Assembly Level: *//' | sed 's/ .*//')
    fi
    
    # Genome size from LOCUS lines
    if [[ "$line" =~ ^LOCUS ]]; then
      local locus_size
      locus_size=$(echo "$line" | awk '{print $3}')
      if [[ "$locus_size" =~ ^[0-9]+$ ]]; then
        ((genome_size += locus_size))
        ((contig_count++))
      fi
    fi
    
    # Features
    if [[ "$line" =~ ^"     CDS " ]]; then
      ((cds_count++))
    elif [[ "$line" =~ ^"     tRNA " ]]; then
      ((trna_count++))
    elif [[ "$line" =~ ^"     rRNA " ]]; then
      ((rrna_count++))
    elif [[ "$line" =~ ^"     gene " ]]; then
      ((gene_count++))
    fi
    
    # DNA sequence for GC content
    if [[ "$line" =~ ^[[:space:]]*[0-9]+[[:space:]]+[acgtACGT[:space:]]+$ ]]; then
      local sequence
      sequence=$(echo "$line" | sed 's/^[[:space:]]*[0-9]*[[:space:]]*//' | tr -d ' \t\n')
      local gc_in_line
      gc_in_line=$(echo "$sequence" | tr -cd 'GCgc' | wc -c)
      local bases_in_line
      bases_in_line=$(echo "$sequence" | tr -cd 'ACGTacgt' | wc -c)
      
      ((gc_count += gc_in_line))
      ((total_bases += bases_in_line))
    fi
    
  done < "$gbff_file"
  
  # Calculate GC content
  local gc_content=0
  if [[ "$total_bases" -gt 0 ]]; then
    gc_content=$(echo "scale=2; ($gc_count * 100) / $total_bases" | bc -l 2>/dev/null || echo "0")
  fi
  
  # Calculate N50 (simplified - using average contig size as proxy)
  local n50_length=0
  if [[ "$contig_count" -gt 0 ]]; then
    n50_length=$((genome_size / contig_count))
  fi
  
  # Output detailed statistics
  echo "$genome_id,$collection,$pathogenicity,$species,$genome_size,$gc_content,$cds_count,$trna_count,$rrna_count,$gene_count,$contig_count,$n50_length,$assembly_level" >> "$DETAILED_CSV"
  
  return 0
}

# Process each collection
echo "Processing genome collections..."
echo

for collection_dir in "${!COLLECTIONS[@]}"; do
  IFS=':' read -r pathogenicity species <<< "${COLLECTIONS[$collection_dir]}"
  
  collection_path="$DATA_DIR/$collection_dir"
  
  if [[ ! -d "$collection_path" ]]; then
    echo "WARNING: Collection directory not found: $collection_path"
    continue
  fi
  
  echo "Analyzing: $collection_dir ($species, $pathogenicity)"
  
  processed_count=0
  collection_genomes=0
  
  # Find all GBFF files in this collection
  while IFS= read -r -d '' gbff_file; do
    if analyze_gbff_file "$gbff_file" "$collection_dir" "$pathogenicity" "$species"; then
      ((processed_count++))
    fi
    ((collection_genomes++))
    
    if (( collection_genomes % 100 == 0 )); then
      echo "  Progress: $collection_genomes genomes processed..."
    fi
    
  done < <(find "$collection_path" -name "*.gbff*" -type f -print0)
  
  echo "  Completed: $processed_count/$collection_genomes genomes analyzed"
  echo
done

# Generate summary statistics using Python/awk
echo "Generating summary statistics..."

# Create summary by collection
{
  echo "collection,pathogenicity,species,genome_count,avg_genome_size,std_genome_size,avg_gc_content,avg_cds_count,avg_gene_count,total_bp"
  
  for collection_dir in "${!COLLECTIONS[@]}"; do
    IFS=':' read -r pathogenicity species <<< "${COLLECTIONS[$collection_dir]}"
    
    # Use awk to calculate statistics for this collection
    awk -F',' -v coll="$collection_dir" -v path="$pathogenicity" -v spec="$species" '
    NR > 1 && $2 == coll {
      count++
      size_sum += $5
      size_sq_sum += ($5 * $5)
      gc_sum += $6
      cds_sum += $7
      gene_sum += $10
      total_bp += $5
    }
    END {
      if (count > 0) {
        avg_size = size_sum / count
        var_size = (size_sq_sum / count) - (avg_size * avg_size)
        std_size = sqrt(var_size > 0 ? var_size : 0)
        avg_gc = gc_sum / count
        avg_cds = cds_sum / count
        avg_gene = gene_sum / count
        printf "%s,%s,%s,%d,%.0f,%.0f,%.2f,%.0f,%.0f,%d\n", coll, path, spec, count, avg_size, std_size, avg_gc, avg_cds, avg_gene, total_bp
      }
    }' "$DETAILED_CSV"
  done
} > "$SUMMARY_CSV"

# Generate pathogenicity group statistics
{
  echo "pathogenicity,total_genomes,total_bp,avg_genome_size,avg_cds_count,species_diversity"
  
  for pathogenicity in "pathogenic" "non_pathogenic"; do
    awk -F',' -v path="$pathogenicity" '
    NR > 1 && $3 == path {
      count++
      total_bp += $5
      cds_sum += $7
      species[$4] = 1
    }
    END {
      if (count > 0) {
        avg_size = total_bp / count
        avg_cds = cds_sum / count
        diversity = length(species)
        printf "%s,%d,%d,%.0f,%.0f,%d\n", path, count, total_bp, avg_size, avg_cds, diversity
      }
    }' "$DETAILED_CSV"
  done
} >> "$COLLECTION_CSV"

echo
echo "Analysis Complete!"
echo "=================="
echo "Output files generated:"
echo "  - Detailed statistics: $DETAILED_CSV"
echo "  - Collection summaries: $SUMMARY_CSV"  
echo "  - Pathogenicity groups: $COLLECTION_CSV"
echo
echo "Key Statistics:"
echo "---------------"

# Display key statistics
if [[ -f "$COLLECTION_CSV" ]]; then
  echo "Dataset Overview:"
  column -t -s',' "$COLLECTION_CSV"
  echo
fi

echo "Ready for stakeholder reporting and visualization!"
echo "Consider creating plots for:"
echo "  - Genome size distributions by pathogenicity"
echo "  - CDS count comparisons across species"
echo "  - GC content variations"
echo "  - Assembly quality metrics"
