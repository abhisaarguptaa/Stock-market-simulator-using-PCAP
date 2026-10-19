"""Roofline placement of the agent kernel from Nsight Compute metrics.

usage: python analysis/roofline.py results/profile/ncu_sorted.csv [results/profile/ncu_unsorted.csv ...]
       [--peak-ops 4.07e12] [--peak-bw 320e9] [--out results/plots/roofline.png]

The kernel is integer-only (no floating point), so the compute roof is the
GPU's INT32 throughput. Defaults are for a Tesla T4: 2560 INT32 lanes x
1.59 GHz boost ~= 4.07 Tera integer ops/s, and 320 GB/s GDDR6 bandwidth.
Pass your GPU's numbers if you profile on something else.
"""
import argparse
import io
import os

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_ncu(path):
    # ncu also prints the program's own output and "==PROF==" lines before the
    # CSV, so start reading at the header row.
    with open(path) as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('"ID"'))
    df = pd.read_csv(io.StringIO("".join(lines[start:])))
    # Nsight Compute --page raw: row 0 holds units; keep only numeric data rows.
    df = df[pd.to_numeric(df["gpu__time_duration.sum"].astype(str).str.replace(",", ""),
                          errors="coerce").notna()].copy()
    for c in df.columns:
        if "__" in c or c.startswith("launch"):
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--peak-ops", type=float, default=4.07e12)
    ap.add_argument("--peak-bw", type=float, default=320e9)
    ap.add_argument("--out", default="results/plots/roofline.png")
    a = ap.parse_args()

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ai = np.logspace(-2, 3, 200)
    ax.plot(ai, np.minimum(a.peak_ops, ai * a.peak_bw), "k-", lw=1.5, label="roofline")
    ridge = a.peak_ops / a.peak_bw
    ax.axvline(ridge, color="gray", ls=":", lw=1)
    ax.text(ridge * 1.1, a.peak_ops * 0.02, f"ridge {ridge:.1f} ops/byte", fontsize=8, color="gray")

    for path in a.files:
        df = read_ncu(path)
        name = os.path.splitext(os.path.basename(path))[0].replace("ncu_", "")
        t = df["gpu__time_duration.sum"] * 1e-9            # ns -> s
        bytes_ = df["dram__bytes_read.sum"] + df["dram__bytes_write.sum"]
        ops = df["smsp__sass_thread_inst_executed_op_integer_pred_on.sum"]
        if "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum" in df:
            ops = ops + df["smsp__sass_thread_inst_executed_op_fp32_pred_on.sum"].fillna(0)
        intensity = (ops / bytes_).median()
        perf = (ops / t).median()
        ax.plot(intensity, perf, "o", ms=8, label=name)
        print(f"[{name}] launches={len(df)}")
        print(f"  kernel time            {t.median() * 1e3:.3f} ms")
        print(f"  arithmetic intensity   {intensity:.2f} integer ops / DRAM byte")
        print(f"  achieved               {perf / 1e9:.1f} G ops/s ({perf / a.peak_ops:.1%} of peak)")
        print(f"  DRAM bandwidth         {(bytes_ / t).median() / 1e9:.1f} GB/s ({(bytes_ / t).median() / a.peak_bw:.1%} of peak)")
        for col, label in [
            ("smsp__thread_inst_executed_per_inst_executed.ratio", "active threads per warp instruction (32 = no divergence)"),
            ("sm__warps_active.avg.pct_of_peak_sustained_active", "achieved occupancy %"),
            ("sm__maximum_warps_per_active_cycle_pct", "theoretical occupancy %"),
            ("launch__registers_per_thread", "registers per thread"),
        ]:
            if col in df:
                print(f"  {label:<56} {df[col].median():.2f}")
        print("  bound by:", "memory bandwidth" if intensity < ridge else "compute (integer throughput)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (integer ops per DRAM byte)")
    ax.set_ylabel("performance (integer ops / s)")
    ax.set_title("Agent decision kernel on the roofline")
    ax.legend()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
