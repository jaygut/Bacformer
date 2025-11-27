#!/usr/bin/env bash
set -Eeuo pipefail

# Hard-Negative Genomes Downloader
# - Downloads complete RefSeq/GenBank assemblies for the six hard-negative taxa
# - Fetches GBFF files directly into a zip archive per taxon.
#
# Taxa:
#   - Listeria innocua          (taxid=1640)
#   - Bacillus subtilis         (taxid=1423)
#   - Citrobacter freundii      (taxid=546)
#   - Citrobacter koseri        (taxid=547)
#   - Escherichia coli (non-O157) (taxid=562)
#   - Escherichia fergusonii    (taxid=564)
#
# Requirements:
#   - NCBI Datasets CLI: https://www.ncbi.nlm.nih.gov/datasets/docs/v2/
#
# Usage:
#   ./scripts/fg_download_hard_negatives.sh

# Resolve repo root relative to this script
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd -P)"
OUT_DIR=${OUT_DIR:-${REPO_ROOT}/data/genometrakr}

# Taxon list (name:taxid:extra_flags)
# The third field is optional and contains extra flags for specific taxa.
TAXA=(
  "listeria_innocua:1640:"
  "bacillus_subtilis:1423:"
  "citrobacter_freundii:546:--limit 2000"
  "citrobacter_koseri:547:--limit 2000"
  "ecoli_nonpathogenic:562:--exclude-strain O157:H7 --limit 2000"
  "escherichia_fergusonii:564:"
)

log() { printf "[%s] %s\n" "$(date +"%F %T")" "$*"; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Error: required command '$1' not found in PATH" >&2
    exit 1
  }
}

main() {
  need_cmd datasets

  mkdir -p "$OUT_DIR"
  log "Output directory: $OUT_DIR"
  log "Downloading hard-negative taxa: ${#TAXA[@]} groups"

  for entry in "${TAXA[@]}"; do
    local name taxid extra_taxon_flags
    IFS=':' read -r name taxid extra_taxon_flags <<<"$entry"

    local zip_path="$OUT_DIR/${name}_gbff.zip"
    log "------------------------------------------------------------"
    log "Target: $name (Taxon ID: $taxid)"
    log "Output: $zip_path"

    if [[ -f "$zip_path" ]]; then
        log "INFO: Zip file already exists, skipping download. Delete the file to re-download."
        continue
    fi

    # Build the download command with available flags
    local -a cmd=(
        datasets download genome taxon "$taxid"
        --assembly-level "complete,chromosome"
        --assembly-source "refseq"
        --annotated --exclude-atypical --mag exclude
        --include gbff
        --filename "$zip_path"
    )

    # Handle special cases for limiting downloads
    if [[ "$extra_taxon_flags" == *"--limit"* ]]; then
        local limit_val=$(echo "$extra_taxon_flags" | grep -o -E '[0-9]+')
        log "INFO: Using restrictive filters (complete+chromosome assemblies, RefSeq only) to reduce genome count."
        log "Target: ~$limit_val genomes, but actual count will depend on available data."
    fi

    # Handle E. coli strain exclusion with a warning
    if [[ "$extra_taxon_flags" == *"--exclude-strain"* ]]; then
        log "WARNING: Strain exclusion not directly supported. Will download all E. coli genomes."
        log "Manual filtering may be needed post-download."
    fi

    log "Executing: ${cmd[*]}"
    if ! "${cmd[@]}"; then
        log "ERROR: Download failed for $name. Please check the error message above."
    else
        log "SUCCESS: Completed download for $name."
    fi
  done

  log "------------------------------------------------------------"
  log "All pending downloads attempted. Output is in: $OUT_DIR"
}

main "$@"
