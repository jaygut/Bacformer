#!/usr/bin/env bash
set -Eeuo pipefail

# FoodGuard genomes downloader (efficient GBFF-only)
# - Downloads complete RefSeq reference/representative assemblies for the three FoodGuard taxa
# - Minimizes disk by either:
#   a) dehydrated + rehydrate only GBFF (default), or
#   b) direct include=gbff (MODE=direct)
#
# Taxa:
#   - Salmonella                (taxid=590)
#   - Escherichia coli O157:H7  (taxid=83334)
#   - Listeria monocytogenes    (taxid=1639)
#
# Requirements:
#   - NCBI Datasets CLI: https://www.ncbi.nlm.nih.gov/datasets/docs/v2/
#   - unzip
#
# Usage examples:
#   # Default dehydrated mode, output under data/genometrakr
#   ./scripts/fg_download_foodguard_taxa.sh
#
#   # Choose output dir and speed up rehydration workers
#   OUT_DIR=data/genomes MODE=dehydrated MAX_WORKERS=20 ./scripts/fg_download_foodguard_taxa.sh
#
#   # Direct mode (no dehydrated step), fetch only GBFF straight into a zip per taxon
#   MODE=direct ./scripts/fg_download_foodguard_taxa.sh
#
# Optional environment variables:
#   OUT_DIR         - destination root directory (default: <repo_root>/data/genometrakr)
#   MODE          - dehydrated | direct (default: dehydrated)
#   MAX_WORKERS   - rehydrate parallel workers (default: 10)
#   ASSEMBLY_LEVELS - comma-separated assembly levels for --assembly-level (default: complete,chromosome,scaffold)
#   RELEASED_AFTER  - filter genomes released on/after date (YYYY-MM-DD) [optional]
#   RELEASED_BEFORE - filter genomes released on/before date (YYYY-MM-DD) [optional]
#   NO_REFREP       - set to 1 to disable --reference filter (default: 1)
#   NO_REFSEQ     - set to 1 to disable RefSeq-only filter (default: 0)
#   PREVIEW       - set to 1 to show preview only (no download)
#   OVERRIDE_TAXA   - semicolon-separated list of name:taxid (e.g., "salmonella_enterica:28901;ecoli_o157h7:83334;listeria_monocytogenes:1639")
#   EXTRA_FLAGS     - additional flags passed to 'datasets download genome' (e.g., "--annotated --exclude-multi-isolate --exclude-atypical --mag exclude")
#   SAMPLE_N        - if set (>0) and MODE=dehydrated, limit rehydration to the first N GBFF entries per taxon
#   USE_TAXON_PRESETS - if 1 (default) and OVERRIDE_TAXA is unset, apply per-taxon production configs automatically
#
 # Resolve repo root relative to this script (so default OUT_DIR is repo-root/data/genometrakr)
 SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)"
 REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd -P)"
OUT_DIR=${OUT_DIR:-${REPO_ROOT}/data/genometrakr}
MODE=${MODE:-dehydrated}          # dehydrated | direct
MAX_WORKERS=${MAX_WORKERS:-10}
# Leave key filters unset so per-taxon presets can apply; fallbacks handled in common_flags()
ASSEMBLY_LEVELS=${ASSEMBLY_LEVELS:-}
RELEASED_AFTER=${RELEASED_AFTER:-}
RELEASED_BEFORE=${RELEASED_BEFORE:-}
NO_REFREP=${NO_REFREP:-}
NO_REFSEQ=${NO_REFSEQ:-}
PREVIEW=${PREVIEW:-0}
USE_TAXON_PRESETS=${USE_TAXON_PRESETS:-1}

# Taxon list (name:taxid)
TAXA=(
  "salmonella:590"
  "ecoli_o157h7:83334"
  "listeria_monocytogenes:1639"
)

# Allow overriding taxa from environment (semicolon-separated name:taxid entries)
if [[ -n "${OVERRIDE_TAXA:-}" ]]; then
  IFS=';' read -r -a TAXA <<<"$OVERRIDE_TAXA"
fi

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Error: required command '$1' not found in PATH" >&2
    exit 1
  }
}

log() { printf "[%s] %s\n" "$(date +"%F %T")" "$*"; }

set_taxon_presets() {
  # Apply production defaults per taxon name. This overrides global defaults
  # when USE_TAXON_PRESETS=1 and OVERRIDE_TAXA is not set.
  local name=$1
  case "$name" in
    salmonella)
      ASSEMBLY_LEVELS="complete,chromosome,scaffold"
      NO_REFSEQ=0
      NO_REFREP=1
      SAMPLE_N=7000
      ;;
    ecoli_o157h7)
      ASSEMBLY_LEVELS="complete,chromosome,scaffold,contig"
      NO_REFSEQ=1
      NO_REFREP=1
      SAMPLE_N=1500
      ;;
    listeria_monocytogenes)
      ASSEMBLY_LEVELS="complete,chromosome,scaffold"
      NO_REFSEQ=1
      NO_REFREP=1
      SAMPLE_N=1500
      ;;
  esac
  # Baseline extra flags if not provided
  if [[ -z "${EXTRA_FLAGS:-}" ]]; then
    EXTRA_FLAGS="--annotated --exclude-atypical --mag exclude"
  fi
  log "Presets → $name: levels=$ASSEMBLY_LEVELS refseq_only=$(( NO_REFSEQ==0 ? 1 : 0 )) sample_n=${SAMPLE_N:-0}"
}

main() {
  need_cmd datasets
  need_cmd unzip

  mkdir -p "$OUT_DIR"
  log "Targets: ${TAXA[*]}"

  for entry in "${TAXA[@]}"; do
    local name=${entry%%:*}
    local taxid=${entry##*:}

    # Apply per-taxon presets when enabled; explicit env vars still override if set before execution
    if [[ "$USE_TAXON_PRESETS" == "1" ]]; then
      set_taxon_presets "$name"
    fi

    if [[ "$MODE" == "dehydrated" ]]; then
      download_dehydrated "$name" "$taxid"
      if [[ "$PREVIEW" != "1" ]]; then
        rehydrate_gbff_only "$name" "$taxid"
        summarize_result "$name"
      fi
    elif [[ "$MODE" == "direct" ]]; then
      download_direct_gbff "$name" "$taxid"
      summarize_zip "$name"
    else
      echo "Error: unknown MODE='$MODE' (use 'dehydrated' or 'direct')" >&2
      exit 1
    fi
  done

  log "Done. Output under: $OUT_DIR"
}

common_flags() {
  # Build an array of common filters to reduce volume
  local -a flags=()
  local eff_levels="${ASSEMBLY_LEVELS:-complete,chromosome,scaffold}"
  flags+=(--assembly-level "$eff_levels")
  flags+=(--assembly-version latest)
  # Default to RefSeq-only unless explicitly disabled
  local no_refseq_val="${NO_REFSEQ:-0}"
  if [[ "$no_refseq_val" != "1" ]]; then
    flags+=(--assembly-source refseq)
  fi
  # By default do not restrict to reference/representative unless explicitly enabled
  local no_refrep_val="${NO_REFREP:-1}"
  if [[ "$no_refrep_val" != "1" ]]; then
    flags+=(--reference)
  fi
  if [[ -n "$RELEASED_AFTER" ]]; then
    flags+=(--released-after "$RELEASED_AFTER")
  fi
  if [[ -n "$RELEASED_BEFORE" ]]; then
    flags+=(--released-before "$RELEASED_BEFORE")
  fi
  printf '%s\n' "${flags[@]}"
}

download_dehydrated() {
  local name=$1
  local taxid=$2
  local zip_path="$OUT_DIR/${name}_dry.zip"
  local -a flags
  mapfile -t flags < <(common_flags)

  log "Downloading (dehydrated) taxon=$taxid name=$name → $zip_path"
  # Include GBFF so fetch.txt contains GBFF entries for rehydration
  local -a cmd=(datasets download genome taxon "$taxid" "${flags[@]}" --include gbff)
  # Inject extra flags if provided
  if [[ -n "${EXTRA_FLAGS:-}" ]]; then
    # shellcheck disable=SC2206
    extra_array=( ${EXTRA_FLAGS} )
    cmd+=("${extra_array[@]}")
  fi
  cmd+=(--dehydrated --filename "$zip_path")
  if [[ "$PREVIEW" == "1" ]]; then
    cmd+=(--preview)
  fi
  # shellcheck disable=SC2068
  ${cmd[@]}

  if [[ "$PREVIEW" != "1" ]]; then
    local outdir="$OUT_DIR/${name}_dry"
    if [[ -d "$outdir" ]]; then
      log "Cleaning existing dir: $outdir"
      if ! rm -rf "$outdir" 2>/dev/null; then
        log "rm -rf failed (likely NFS busy). Moving aside..."
        mv "$outdir" "${outdir}.stale.$(date +%s)" || true
      fi
    fi
    mkdir -p "$outdir"
    unzip -q -o "$zip_path" -d "$outdir"
  fi
}

rehydrate_gbff_only() {
  local name=$1
  local taxid=$2
  local bag_root="$OUT_DIR/${name}_dry"
  local bag_dir="$bag_root/ncbi_dataset"
  if [[ ! -d "$bag_dir" ]]; then
    echo "Error: bag directory not found: $bag_dir" >&2
    exit 1
  fi
  log "Rehydrating GBFF only for $name (workers=$MAX_WORKERS)"
  # If dehydrated bag contains no GBFF entries, fall back to direct mode for this taxon
  local fetch_file="$bag_dir/fetch.txt"
  if [[ -f "$fetch_file" ]] && ! grep -qiE '\\.gbff(\\.gz)?' "$fetch_file"; then
    log "WARN: No GBFF entries in fetch list for $name; falling back to direct include=gbff"
    download_direct_gbff "$name" "$taxid"
    summarize_zip "$name"
    return 0
  fi
  # If sampling requested, reduce fetch list to first N GBFF entries
  if [[ -n "${SAMPLE_N:-}" && "$SAMPLE_N" -gt 0 ]]; then
    local fetch_all="$bag_dir/fetch_all.txt"
    local fetch_file="$bag_dir/fetch.txt"
    if [[ -f "$fetch_file" ]]; then
      mv -f "$fetch_file" "$fetch_all"
      # Use a broader, more robust GBFF match to avoid empty sampling
      grep -Ei "\\.gbff(\\.gz)?$" "$fetch_all" | head -n "$SAMPLE_N" > "$fetch_file" || true
      if [[ ! -s "$fetch_file" ]]; then
        # If no GBFF lines matched, restore original fetch file to avoid breaking rehydrate
        mv -f "$fetch_all" "$fetch_file"
        log "WARN: No GBFF entries found during sampling for $name; proceeding without sampling."
      else
        log "Sampling enabled: limited to first $SAMPLE_N GBFF entries for $name"
      fi
    fi
  fi
  # Important: pass the parent bag directory to avoid double 'ncbi_dataset' in the path
  if ! datasets rehydrate \
    --directory "$bag_root" \
    --match "gbff" \
    --gzip \
    --max-workers "$MAX_WORKERS"; then
    log "WARN: rehydrate reported an error for $name; continuing to next taxon"
  fi

  # Convenience: copy assembly report to top-level as manifest
  if [[ -f "$bag_dir/assembly_data_report.jsonl" ]]; then
    cp -f "$bag_dir/assembly_data_report.jsonl" "$OUT_DIR/${name}_assembly_report.jsonl"
  fi
}

summarize_result() {
  local name=$1
  local data_dir="$OUT_DIR/${name}_dry/ncbi_dataset/data"
  if [[ -d "$data_dir" ]]; then
    local n_gz n_raw n
    n_gz=$(find "$data_dir" -type f -iname "*.gbff.gz" | wc -l | awk '{print $1}')
    n_raw=$(find "$data_dir" -type f -iname "*.gbff" ! -name "*.gz" | wc -l | awk '{print $1}')
    n=$(( n_gz + n_raw ))
    log "${name}: fetched $n GBFF files (gz=$n_gz, plain=$n_raw)"
  fi
}

download_direct_gbff() {
  local name=$1
  local taxid=$2
  local zip_path="$OUT_DIR/${name}_gbff.zip"
  local -a flags
  mapfile -t flags < <(common_flags)

  # Direct v2-style include (fetches only GBFF into one zip)
  log "Downloading (direct include=gbff) taxon=$taxid name=$name → $zip_path"
  local -a cmd=(datasets download genome taxon "$taxid" "${flags[@]}")
  if [[ -n "${EXTRA_FLAGS:-}" ]]; then
    # shellcheck disable=SC2206
    extra_array=( ${EXTRA_FLAGS} )
    cmd+=("${extra_array[@]}")
  fi
  cmd+=(--include gbff --filename "$zip_path")
  if [[ "$PREVIEW" == "1" ]]; then
    cmd+=(--preview)
  fi
  # shellcheck disable=SC2068
  ${cmd[@]}
}

summarize_zip() {
  local name=$1
  local zip_path="$OUT_DIR/${name}_gbff.zip"
  if [[ -f "$zip_path" ]]; then
    # List how many GBFFs are inside the zip (best-effort; requires 'zipinfo')
    if command -v zipinfo >/dev/null 2>&1; then
      local n
      n=$(zipinfo -1 "$zip_path" | grep -Ei "\\.gbff(\\.gz)?$" | wc -l | awk '{print $1}')
      log "${name}: zip contains ~$n GBFF files"
    else
      log "${name}: downloaded $zip_path (install 'zipinfo' to summarize contents)"
    fi
  fi
}

main "$@"
