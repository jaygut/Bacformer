#!/usr/bin/env bash
# Select-and-unzip a capped number of GBFF files from the three FoodGuard taxa zips.
# Defaults:
# - Salmonella: first 7000 GBFFs
# - E. coli O157:H7: first 1500 GBFFs
# - Listeria monocytogenes: first 1500 GBFFs (user-adjustable)
#
# You can override paths and caps with env vars before running, e.g.:
#   SAL_SAMPLE_N=5000 LISTERIA_SAMPLE_N=1200 ./scripts/fg_select_gbff_from_zips.sh
#   SAL_ZIP=/path/to/salmonella.zip SAL_OUT=/some/dir ./scripts/fg_select_gbff_from_zips.sh
# To delete the original *_gbff.zip after a successful extract, set:
#   DELETE_ZIPS=1 ./scripts/fg_select_gbff_from_zips.sh
#
# Paths default to the ones produced by fg_download_foodguard_taxa.sh (direct mode fallback).

set -Euo pipefail

log() { printf "[%s] %s\n" "$(date +"%F %T")" "$*"; }

# Defaults (override via env)
SAL_ZIP=${SAL_ZIP:-/home/jcorre38/Jay_Proyects/FoodGuardAI/data/genometrakr/salmonella_gbff.zip}
ECOLI_ZIP=${ECOLI_ZIP:-/home/jcorre38/Jay_Proyects/FoodGuardAI/data/genometrakr/ecoli_o157h7_gbff.zip}
LISTERIA_ZIP=${LISTERIA_ZIP:-/home/jcorre38/Jay_Proyects/FoodGuardAI/data/genometrakr/listeria_monocytogenes_gbff.zip}

SAL_OUT=${SAL_OUT:-/home/jcorre38/Jay_Proyects/FoodGuardAI/data/genometrakr/salmonella_gbff_selected}
ECOLI_OUT=${ECOLI_OUT:-/home/jcorre38/Jay_Proyects/FoodGuardAI/data/genometrakr/ecoli_o157h7_gbff_selected}
LISTERIA_OUT=${LISTERIA_OUT:-/home/jcorre38/Jay_Proyects/FoodGuardAI/data/genometrakr/listeria_monocytogenes_gbff_selected}

SAL_SAMPLE_N=${SAL_SAMPLE_N:-7000}
ECOLI_SAMPLE_N=${ECOLI_SAMPLE_N:-1500}
LISTERIA_SAMPLE_N=${LISTERIA_SAMPLE_N:-4500}

# If set to 1, delete the source zip after a successful extraction (count > 0)
DELETE_ZIPS=${DELETE_ZIPS:-0}

# Use zipinfo if available; otherwise fallback to unzip -Z -1
list_zip_entries() {
  local zip_path=$1
  if command -v zipinfo >/dev/null 2>&1; then
    zipinfo -1 "$zip_path" || unzip -Z -1 "$zip_path"
  else
    unzip -Z -1 "$zip_path"
  fi
}

extract_first_n_gbff() {
  local zip_path=$1
  local out_dir=$2
  local n=$3
  local tag=$4

  if [[ ! -f "$zip_path" ]]; then
    log "SKIP $tag: zip not found: $zip_path"
    return 0
  fi

  mkdir -p "$out_dir"
  local tmp_list
  tmp_list=$(mktemp -t ${tag}_gbff_list.XXXXXX)
  # Expand variable at trap-set time to avoid set -u issues on RETURN
  trap "rm -f '$tmp_list'" RETURN

  # List GBFF entries (both .gbff and .gbff.gz), take first N
  # Disable pipefail in subshell to avoid grep SIGPIPE from head causing a false failure
  (
    set +o pipefail
    list_zip_entries "$zip_path" | grep -Ei '\.gbff(\.gz)?$' | head -n "$n" > "$tmp_list"
  )

  # If the list is empty, log and return
  if [[ ! -s "$tmp_list" ]]; then
    log "WARN $tag: no GBFF entries found in $zip_path"
    return 1
  fi

  # Extract only the selected entries; preserve internal paths
  local list_count
  list_count=$(wc -l < "$tmp_list" | awk '{print $1}')
  log "$tag: extracting $list_count selected GBFF paths (this may take a while)"

  if command -v bsdtar >/dev/null 2>&1; then
    # Fast, native bulk extract using bsdtar
    if ! bsdtar -x -f "$zip_path" -C "$out_dir" -T "$tmp_list"; then
      log "WARN $tag: bsdtar reported errors; attempting per-file unzip fallback"
      (
        cd "$out_dir" && xargs -I{} -r bash -c 'unzip -q -o "$0" "$1" || true' "$zip_path" '{}' < "$tmp_list"
      ) || {
        log "ERROR $tag: unzip fallback failed while extracting entries from $zip_path"
        return 1
      }
    fi
  else
    # Portable fallback: extract one-by-one with unzip to avoid -@ quirks
    (
      cd "$out_dir" && xargs -I{} -r bash -c 'unzip -q -o "$0" "$1" || true' "$zip_path" '{}' < "$tmp_list"
    ) || {
      log "ERROR $tag: unzip failed while extracting entries from $zip_path"
      return 1
    }
  fi

  # Verify count
  local count
  count=$(find "$out_dir" -type f -iname '*.gbff*' | wc -l | awk '{print $1}')
  log "$tag: extracted ~$count GBFF files into $out_dir"

  if [[ "$count" -lt "$list_count" ]]; then
    log "WARN $tag: extracted $count of $list_count requested GBFF files; archive may be partially corrupted. Consider re-downloading with filters and retrying."
  fi

  # Optionally delete the source zip if we extracted something
  if [[ "$DELETE_ZIPS" == "1" ]]; then
    if [[ "$count" -gt 0 ]]; then
      rm -f "$zip_path" && log "$tag: deleted source zip $zip_path"
    else
      log "WARN $tag: no GBFF extracted; not deleting $zip_path"
    fi
  fi
}

main() {
  log "Selecting GBFF subsets from direct zips"
  log "Salmonella → N=$SAL_SAMPLE_N | zip=$SAL_ZIP | out=$SAL_OUT"
  extract_first_n_gbff "$SAL_ZIP" "$SAL_OUT" "$SAL_SAMPLE_N" "salmonella" || true

  log "E. coli O157:H7 → N=$ECOLI_SAMPLE_N | zip=$ECOLI_ZIP | out=$ECOLI_OUT"
  extract_first_n_gbff "$ECOLI_ZIP" "$ECOLI_OUT" "$ECOLI_SAMPLE_N" "ecoli_o157h7" || true

  log "Listeria monocytogenes → N=$LISTERIA_SAMPLE_N | zip=$LISTERIA_ZIP | out=$LISTERIA_OUT"
  extract_first_n_gbff "$LISTERIA_ZIP" "$LISTERIA_OUT" "$LISTERIA_SAMPLE_N" "listeria_monocytogenes" || true

  log "Done."
}

main "$@"
