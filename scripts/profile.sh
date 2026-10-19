#!/usr/bin/env bash
# Profiles the CUDA run with Nsight Systems (timeline: kernel vs copies vs CPU
# matching) and Nsight Compute (occupancy, divergence, memory bandwidth, and
# the numbers analysis/roofline.py needs).
#
# usage: scripts/profile.sh [build-dir]
set -euo pipefail
BUILD=${1:-build}
BIN="$BUILD/lobsim"
OUT=${OUT:-results/profile}
mkdir -p "$OUT"
ARGS=(--agents 1000000 --instruments 128 --ticks 50 --decide cuda --match omp)

echo "== Nsight Systems timeline"
nsys profile -o "$OUT/timeline" --force-overwrite true --trace cuda,openmp,osrt \
  "$BIN" "${ARGS[@]}" > /dev/null
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum --format csv \
  --output "$OUT/nsys" "$OUT/timeline.nsys-rep" > /dev/null || true
echo "  wrote $OUT/timeline.nsys-rep (open in the Nsight Systems GUI) and $OUT/nsys_*.csv"

# Key metrics, one row per profiled kernel launch.
METRICS=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,\
smsp__sass_thread_inst_executed_op_integer_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fp32_pred_on.sum,\
smsp__thread_inst_executed_per_inst_executed.ratio,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__maximum_warps_per_active_cycle_pct,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
launch__registers_per_thread

for variant in sorted unsorted; do
  extra=()
  [ $variant = unsorted ] && extra=(--no-sort)
  echo "== Nsight Compute metrics ($variant agents)"
  ncu --kernel-name regex:decide --launch-skip 10 --launch-count 5 \
      --metrics "$METRICS" --csv --page raw \
      "$BIN" "${ARGS[@]}" "${extra[@]}" > "$OUT/ncu_$variant.csv"
  echo "  wrote $OUT/ncu_$variant.csv"
done

echo "== Nsight Compute full report (one launch)"
ncu --kernel-name regex:decide --launch-skip 10 --launch-count 1 --set full \
    -o "$OUT/decide_full" --force-overwrite "$BIN" "${ARGS[@]}" > /dev/null
echo "  wrote $OUT/decide_full.ncu-rep (open in the Nsight Compute GUI)"
echo "next: python analysis/roofline.py $OUT/ncu_sorted.csv $OUT/ncu_unsorted.csv"
