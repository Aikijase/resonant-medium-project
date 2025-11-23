#!/usr/bin/env bash
# Build Tier-3 Evidence Pack for Resonant Medium Project
# Usage: bash tools/build_tier3_evidence_pack.sh [--dry-run]
set -euo pipefail

DRYRUN=0
[[ "${1:-}" == "--dry-run" ]] && DRYRUN=1

log()  { printf "[tier3] %s\n" "$*"; }
run()  { if [[ $DRYRUN -eq 1 ]]; then echo "DRY: $*"; else eval "$@"; fi; }
exists() { shopt -s nullglob; local arr=("$@"); ((${#arr[@]})); }

# --- Paths ---
ROOT="$(pwd)"
OUTDIR="$ROOT/tier3_evidence"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ARCHIVE="$ROOT/tier3_evidence_$STAMP.zip"
MANIFEST="$OUTDIR/SHA256SUMS.txt"

# --- Create structure ---
log "Creating folder structure at $OUTDIR"
for d in \
  01_memory_term \
  02_resonant_modes \
  03_rmt_vs_lcdm \
  04_universal_resonance \
  05_robustness \
  06_dark_matter_nodal \
  07_black_hole_recycling \
  99_project_state
do
  run "mkdir -p '$OUTDIR/$d'"
done

copy_globs() {
  # copy_globs <dest> <glob1> [glob2 ...]
  local dest="$1"; shift
  shopt -s nullglob
  local any=0
  for g in "$@"; do
    local matches=($g)
    if ((${#matches[@]})); then
      any=1
      for f in "${matches[@]}"; do
        run "install -D -m 0644 '$f' '$dest/$(basename "$f")'"
      done
    fi
  done
  if [[ $any -eq 0 ]]; then
    log "No matches for: $* (dest $dest)"
  fi
}

# --- 01: Memory term (Phase-2) ---
copy_globs "$OUTDIR/01_memory_term" \
  outputs/phase2/*.json \
  outputs/phase2/*summary*.csv \
  outputs/phase2/*REPORT*.md \
  outputs/joint_tau*.json

# --- 02: Resonant modes (Phase-8) ---
copy_globs "$OUTDIR/02_resonant_modes" \
  outputs/phase8/*spectrum*.png \
  outputs/phase8/*.json \
  outputs/phase8/*lorentz*.json \
  outputs/phase8/*surrogate*.json
# Helper script for provenance
copy_globs "$OUTDIR/02_resonant_modes" \
  tools/phase8_spectral/real_quick_check.py

# --- 03: RMT vs ΛCDM (Phase-4 + metrics) ---
copy_globs "$OUTDIR/03_rmt_vs_lcdm" \
  outputs/phase4/*.json \
  outputs/phase4/*.txt \
  outputs/phase4/*report* \
  outputs/phase2/*WWI* \
  outputs/*/fit_summary.csv

# --- 04: Universal resonance (Phase-19/20) ---
copy_globs "$OUTDIR/04_universal_resonance" \
  outputs/phase19/*.csv \
  outputs/phase19/*.png \
  outputs/phase20/*.png \
  outputs/phase20/*.csv

# --- 05: Robustness (Phase-15/17) ---
copy_globs "$OUTDIR/05_robustness" \
  outputs/phase15/* \
  outputs/phase17/*

# --- 06: Dark-matter-as-nodes (emerging) ---
copy_globs "$OUTDIR/06_dark_matter_nodal" \
  outputs/phase19/*node* \
  outputs/phase19/*filament* \
  outputs/phase20/*fit*

# --- 07: Black hole recycling (Phase-21 in progress) ---
copy_globs "$OUTDIR/07_black_hole_recycling" \
  tools/phase21_recycling_ode.py \
  outputs/phase21/*

# --- 99: Project state (env, configs, tools, readme) ---
# Lock the environment
if command -v python3 >/dev/null 2>&1; then
  run "python3 -V > '$OUTDIR/99_project_state/python_version.txt' 2>&1 || true"
  run "python3 -m pip freeze > '$OUTDIR/99_project_state/requirements_lock.txt' 2>/dev/null || true"
fi

# System snapshot
run "uname -a > '$OUTDIR/99_project_state/system_uname.txt'"
run "printf 'Timestamp (UTC): %s\n' '$STAMP' > '$OUTDIR/99_project_state/build_stamp.txt'"

# Git snapshot if repo present
if [ -d .git ]; then
  run "git rev-parse HEAD > '$OUTDIR/99_project_state/git_commit.txt'"
  run "git status --porcelain > '$OUTDIR/99_project_state/git_status_porcelain.txt'"
  run "git diff --stat > '$OUTDIR/99_project_state/git_diffstat.txt'"
fi

# Configs and top-level docs
copy_globs "$OUTDIR/99_project_state" \
  README.md \
  requirements.txt \
  pyproject.toml \
  setup.cfg \
  configs/*

# Keep tool sources for reproducibility (shallow copy of .py under tools/)
if [ -d tools ]; then
  run "mkdir -p '$OUTDIR/99_project_state/tools_src'"
  shopt -s globstar nullglob
  for f in tools/**/*.py; do
    run "install -D -m 0644 '$f' '$OUTDIR/99_project_state/tools_src/$(basename "$f")'"
  done
fi

# Optional: copy plots & logs if present (lightweight)
if [ -d plots ]; then
  run "mkdir -p '$OUTDIR/99_project_state/plots_refs'"
  copy_globs "$OUTDIR/99_project_state/plots_refs" \
    plots/*.png plots/*/*.png
fi
if [ -d logs ]; then
  run "mkdir -p '$OUTDIR/99_project_state/logs_refs'"
  copy_globs "$OUTDIR/99_project_state/logs_refs" \
    logs/* logs/**/*.log
fi

# --- SHA256 manifest ---
log "Computing SHA256 manifest"
if [[ $DRYRUN -eq 0 ]]; then
  (cd "$OUTDIR" && \
    find . -type f -print0 | sort -z | xargs -0 sha256sum > "SHA256SUMS.txt")
else
  echo "DRY: would run sha256sum over $OUTDIR"
fi

# --- Zip archive ---
log "Creating archive: $ARCHIVE"
run "cd '$OUTDIR' && zip -r '$ARCHIVE' . >/dev/null"

log "Done."
log "Packed: $ARCHIVE"
log "Manifest: $MANIFEST"
[[ $DRYRUN -eq 1 ]] && log "NOTE: Dry run only. Re-run without --dry-run to execute."
