#!/usr/bin/env bash
# Benchmark experiments for the scaling study. Each experiment appends CSV
# rows (one per run) to results/<experiment>.csv; plot them with
# analysis/scaling.py.
#
# usage: scripts/bench.sh <experiment|all|cpu|gpu> [build-dir]
#   strong      OpenMP strong scaling (fixed problem, 1..N threads)
#   weak        OpenMP weak scaling (agents grow with threads)
#   instruments OpenMP matching vs number of instruments
#   crossover   serial vs OpenMP vs CUDA across agent counts (objective 8)
#   gustafson   CUDA with fixed instruments and growing agents (Amdahl vs Gustafson)
#   cuda        CUDA ablations: block size, type sorting, compaction
#   mpi         MPI ranks on one machine (communication share of tick time)
#   falseshare  padded vs unpadded counters (needs build-nopad, see README)
#   cpu = strong weak instruments mpi falseshare   gpu = crossover gustafson cuda
#
# Environment overrides: TICKS, REPS, MAX_THREADS, OUT
set -euo pipefail
EXP=${1:-all}
BUILD=${2:-build}
BIN="$BUILD/lobsim"
OUT=${OUT:-results}
TICKS=${TICKS:-300}
REPS=${REPS:-3}
MAX_THREADS=${MAX_THREADS:-$(nproc)}
MPIRUN=(mpirun --oversubscribe --bind-to none)
[ "$(id -u)" = "0" ] && MPIRUN+=(--allow-run-as-root)
mkdir -p "$OUT"

# run <csv> <label> <lobsim args...>: REPS repetitions, header written once.
run() {
  local csv="$OUT/$1.csv" label=$2 r; shift 2
  for ((r = 0; r < REPS; r++)); do
    if [ -s "$csv" ]; then "$@" --label "$label" >> "$csv"
    else "$@" --label "$label" --header >> "$csv"; fi
  done
  tail -1 "$csv" | awk -F, '{printf "  %-12s %-7s %-7s ranks=%s thr=%-3s agents=%-8s instr=%-4s %8.2f ticks/s\n", $1,$2,$3,$4,$5,$6,$7,$24}'
}

threads_list() { local t=1; while [ $t -le "$MAX_THREADS" ]; do echo $t; t=$((t * 2)); done
                 [ $((MAX_THREADS & (MAX_THREADS - 1))) -ne 0 ] && echo "$MAX_THREADS"; true; }

# Strong and weak scaling compare runs against each other, so they are
# interleaved: each round runs every configuration once. If the machine slows
# down partway (power saving, heat, background work), every thread count is
# affected equally instead of whichever happened to run at that moment.
strong() {
  echo "== strong scaling (OpenMP, interleaved rounds)"
  local reps=$REPS round t; REPS=1
  for ((round = 0; round < reps; round++)); do
    echo " round $((round + 1))/$reps"
    run strong serial "$BIN" --agents 400000 --instruments 64 --ticks "$TICKS"
    for t in $(threads_list); do
      run strong omp "$BIN" --agents 400000 --instruments 64 --ticks "$TICKS" --decide omp --match omp --threads "$t"
    done
  done
  REPS=$reps
}

weak() {
  echo "== weak scaling (OpenMP, 50k agents and 8 instruments per thread, interleaved rounds)"
  local reps=$REPS round t; REPS=1
  for ((round = 0; round < reps; round++)); do
    echo " round $((round + 1))/$reps"
    for t in $(threads_list); do
      run weak omp "$BIN" --agents $((50000 * t)) --instruments $((8 * t)) --ticks "$TICKS" --decide omp --match omp --threads "$t"
    done
  done
  REPS=$reps
}

instruments() {
  echo "== matching vs instrument count (OpenMP, fixed 200k agents)"
  for m in 1 2 4 8 16 32 64 128; do
    run instruments serial "$BIN" --agents 200000 --instruments $m --ticks "$TICKS" --decide omp --match serial --threads "$MAX_THREADS"
    run instruments omp    "$BIN" --agents 200000 --instruments $m --ticks "$TICKS" --decide omp --match omp --threads "$MAX_THREADS"
  done
}

crossover() {
  echo "== crossover: serial vs OpenMP vs CUDA decide (matching on OpenMP)"
  for n in 1000 4000 16000 64000 256000 1000000 2000000; do
    local m=$(( n / 4000 )); [ $m -lt 4 ] && m=4; [ $m -gt 256 ] && m=256
    run crossover serial "$BIN" --agents $n --instruments $m --ticks "$TICKS" --decide serial --match serial
    run crossover omp    "$BIN" --agents $n --instruments $m --ticks "$TICKS" --decide omp --match omp --threads "$MAX_THREADS"
    run crossover cuda   "$BIN" --agents $n --instruments $m --ticks "$TICKS" --decide cuda --match omp --threads "$MAX_THREADS"
  done
}

gustafson() {
  echo "== fixed 32 instruments, growing agents (serial fraction vs problem size)"
  for n in 32000 128000 512000 2000000; do
    run gustafson serial "$BIN" --agents $n --instruments 32 --ticks "$TICKS" --decide serial --match serial
    run gustafson cuda   "$BIN" --agents $n --instruments 32 --ticks "$TICKS" --decide cuda --match omp --threads "$MAX_THREADS"
  done
}

cuda() {
  echo "== CUDA ablations (1M agents, 128 instruments)"
  local A=(--agents 1000000 --instruments 128 --ticks "$TICKS" --decide cuda --match omp --threads "$MAX_THREADS")
  for b in 32 64 128 256 512 1024; do run cuda_block "block$b" "$BIN" "${A[@]}" --block $b; done
  run cuda_ablation sorted          "$BIN" "${A[@]}"
  run cuda_ablation unsorted        "$BIN" "${A[@]}" --no-sort
  run cuda_ablation full_copy       "$BIN" "${A[@]}" --no-compact
  run cuda_ablation unsorted_full   "$BIN" "${A[@]}" --no-sort --no-compact
}

mpi() {
  echo "== MPI ranks on one machine"
  for np in 1 2 4; do
    [ $np -gt "$MAX_THREADS" ] && break
    run mpi "np$np" "${MPIRUN[@]}" -np $np "$BIN" --agents 400000 --instruments 64 --ticks "$TICKS" \
        --decide omp --match omp --threads $(( MAX_THREADS / np > 0 ? MAX_THREADS / np : 1 ))
  done
}

falseshare() {
  echo "== false sharing: padded vs unpadded per-instrument counters"
  local A=(--agents 200000 --instruments 256 --ticks "$TICKS" --decide omp --match omp --threads "$MAX_THREADS" --ttl 20)
  run falseshare padded "$BIN" "${A[@]}"
  if [ -x build-nopad/lobsim ]; then run falseshare unpadded build-nopad/lobsim "${A[@]}"
  else echo "  build-nopad/lobsim missing: cmake -S . -B build-nopad -DLOB_NO_PAD=ON && cmake --build build-nopad"; fi
}

case "$EXP" in
  all) strong; weak; instruments; mpi; falseshare; crossover; gustafson; cuda ;;
  cpu) strong; weak; instruments; mpi; falseshare ;;
  gpu) crossover; gustafson; cuda ;;
  strong|weak|instruments|crossover|gustafson|cuda|mpi|falseshare) "$EXP" ;;
  *) echo "unknown experiment $EXP"; exit 1 ;;
esac
echo "results in $OUT/"
