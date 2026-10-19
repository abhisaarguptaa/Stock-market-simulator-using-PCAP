#!/usr/bin/env bash
# A short live demo of the project, for presenting. Runs in a few minutes on
# a laptop and prints three results:
#   1. the unit tests pass
#   2. every back end (serial, OpenMP, MPI, and CUDA if this build and machine
#      have it) produces the identical trade log, and how long each takes
#   3. matching only speeds up across instruments: the serial core
#
# usage: bash scripts/demo.sh [build-dir]
# Environment overrides: AGENTS, INSTRUMENTS, TICKS, THREADS
set -uo pipefail
BUILD=${1:-build}
BIN="$BUILD/lobsim"
AGENTS=${AGENTS:-200000}
INSTRUMENTS=${INSTRUMENTS:-64}
TICKS=${TICKS:-200}
THREADS=${THREADS:-$(nproc)}
MPIRUN=(mpirun --oversubscribe --bind-to none)
[ "$(id -u)" = "0" ] && MPIRUN+=(--allow-run-as-root)

if [ ! -x "$BIN" ]; then
  echo "No $BIN found. Build first:  cmake -S . -B $BUILD -DLOB_USE_MPI=ON && cmake --build $BUILD -j"
  exit 1
fi

line() { printf '%s\n' "------------------------------------------------------------------"; }

echo
line
echo " Heterogeneous parallel limit order book simulator: live demo"
line

echo
echo "1. Unit tests (order book, RNG, every back end against serial)"
"$BUILD/lob_tests" | tail -1 | sed 's/^/   /'

echo
echo "2. Same trades on every back end"
echo "   $AGENTS agents, $INSTRUMENTS instruments, $TICKS ticks, seed 42"
echo
printf "   %-22s %9s %12s   %s\n" "back end" "time" "trades" "trade-log fingerprint"
ARGS=(--agents "$AGENTS" --instruments "$INSTRUMENTS" --ticks "$TICKS" --seed 42)
declare -a HASHES
SERIAL_T=""
row() {
  local name=$1; shift
  local out
  out=$("$@" 2>/dev/null | tail -1) || return 0
  [ -z "$out" ] && return 0
  # A build without MPI runs each process as a full copy (1 rank each): skip.
  [ "$name" = "MPI, 2 processes" ] && [ "$(echo "$out" | cut -d, -f4)" != "2" ] && return 0
  local trades hash t
  trades=$(echo "$out" | cut -d, -f13)
  hash=$(echo "$out" | cut -d, -f15)
  t=$(echo "$out" | cut -d, -f16)
  HASHES+=("$hash")
  [ -z "$SERIAL_T" ] && SERIAL_T=$t
  local speed
  speed=$(awk -v s="$SERIAL_T" -v t="$t" 'BEGIN { printf "%.2fx", s / t }')
  printf "   %-22s %7.2f s %12s   %s   (%s vs serial)\n" "$name" "$t" "$trades" "$hash" "$speed"
}
row "serial"                "$BIN" "${ARGS[@]}" --decide serial --match serial
row "OpenMP, $THREADS threads" "$BIN" "${ARGS[@]}" --decide omp --match omp --threads "$THREADS"
before=${#HASHES[@]}
if command -v mpirun >/dev/null; then
  row "MPI, 2 processes"    "${MPIRUN[@]}" -np 2 "$BIN" "${ARGS[@]}" --decide omp --match omp --threads $(( THREADS / 2 > 0 ? THREADS / 2 : 1 ))
fi
MPI_RAN=$(( ${#HASHES[@]} > before ))
before=${#HASHES[@]}
row "CUDA GPU + OpenMP"     "$BIN" "${ARGS[@]}" --decide cuda --match omp --threads "$THREADS"
CUDA_RAN=$(( ${#HASHES[@]} > before ))
echo
if [ "${#HASHES[@]}" -gt 1 ] && [ "$(printf '%s\n' "${HASHES[@]}" | sort -u | wc -l)" -eq 1 ]; then
  echo "   RESULT: all ${#HASHES[@]} back ends produced the identical trade log."
else
  echo "   RESULT: fingerprints differ (or only one back end ran)."
fi
[ "$MPI_RAN" -eq 0 ] && echo "   (MPI not installed or not in this build, so it was skipped)"
[ "$CUDA_RAN" -eq 0 ] && echo "   (no GPU in this build or machine, so CUDA was skipped)"

echo
echo "3. The serial core: matching only speeds up across instruments"
echo "   $AGENTS agents, 100 ticks, matching time serial vs OpenMP ($THREADS threads)"
echo
printf "   %-14s %14s %14s %10s\n" "instruments" "serial match" "OpenMP match" "speedup"
for m in 1 64; do
  s=$("$BIN" --agents "$AGENTS" --instruments $m --ticks 100 --decide omp --match serial --threads "$THREADS" | tail -1 | cut -d, -f21)
  p=$("$BIN" --agents "$AGENTS" --instruments $m --ticks 100 --decide omp --match omp --threads "$THREADS" | tail -1 | cut -d, -f21)
  awk -v m="$m" -v s="$s" -v p="$p" 'BEGIN { printf "   %-14s %12.2f s %12.2f s %9.2fx\n", m, s, p, s / p }'
done
echo
echo "   With 1 instrument there is nothing to split: orders must be matched"
echo "   one at a time (price-time priority). More instruments, more speedup."
line
