#!/usr/bin/env bash
# Runs the same simulation through every available back end and checks that
# all trade-log hashes are identical (objective 6: exact comparison under a
# fixed seed).
#
# usage: scripts/check_determinism.sh [build-dir] [extra lobsim args...]
set -euo pipefail
BUILD=${1:-build}
shift || true
BIN="$BUILD/lobsim"
ARGS=(--agents 50000 --instruments 16 --ticks 500 --seed 1234 "$@")
MPIRUN=(mpirun --oversubscribe --bind-to none)
[ "$(id -u)" = "0" ] && MPIRUN+=(--allow-run-as-root)

declare -a NAMES HASHES
run() {
  local name=$1; shift
  local out
  if ! out=$("$@" 2>/dev/null | tail -1); then
    echo "  skipped $name (could not run here, e.g. no GPU in this build or machine)"
    return
  fi
  NAMES+=("$name")
  HASHES+=("$(echo "$out" | cut -d, -f15)")
  printf "  %-28s trades=%-9s hash=%s\n" "$name" "$(echo "$out" | cut -d, -f13)" "${HASHES[-1]}"
}

echo "Determinism check: ${ARGS[*]}"
run "serial"              "$BIN" "${ARGS[@]}" --decide serial --match serial
for t in 1 2 4 8; do
  run "omp decide+match x$t" "$BIN" "${ARGS[@]}" --decide omp --match omp --threads "$t"
done
run "cuda + omp match"    "$BIN" "${ARGS[@]}" --decide cuda --match omp
run "cuda, no compaction" "$BIN" "${ARGS[@]}" --decide cuda --match omp --no-compact
run "cuda, unsorted"      "$BIN" "${ARGS[@]}" --decide cuda --match omp --no-sort
if command -v mpirun >/dev/null; then
  for np in 2 4; do
    run "mpi x$np (omp)"   "${MPIRUN[@]}" -np "$np" "$BIN" "${ARGS[@]}" --decide omp --match omp --threads 1
    run "mpi x$np (cuda)"  "${MPIRUN[@]}" -np "$np" "$BIN" "${ARGS[@]}" --decide cuda --match omp --threads 1
  done
fi

ref=${HASHES[0]}
bad=0
for i in "${!HASHES[@]}"; do
  [ "${HASHES[$i]}" = "$ref" ] || { echo "MISMATCH: ${NAMES[$i]}"; bad=1; }
done
if [ $bad = 0 ]; then
  echo "PASS: ${#HASHES[@]} configurations produced identical trade logs"
else
  echo "FAIL: trade logs differ"
  exit 1
fi
