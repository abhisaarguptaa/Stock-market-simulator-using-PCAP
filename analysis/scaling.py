"""Turns benchmark CSVs from scripts/bench.sh into plots and a summary.

usage: python analysis/scaling.py [results-dir]
Writes <results-dir>/plots/*.png and <results-dir>/summary.md. Experiments
whose CSV is missing are skipped, so CPU results (laptop) and GPU results
(Colab) can be collected separately into the same folder.
"""
import os
import sys

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
PLOTS = os.path.join(RES, "plots")
os.makedirs(PLOTS, exist_ok=True)
summary = ["# Scaling study summary", ""]
PHASES = ["t_decide", "t_h2d", "t_kernel", "t_d2h", "t_match", "t_publish", "t_comm"]


def load(name):
    path = os.path.join(RES, name + ".csv")
    if not os.path.exists(path):
        print(f"skip {name}: {path} not found")
        return None
    df = pd.read_csv(path)
    keys = ["label", "decide", "match", "ranks", "threads", "agents", "instruments", "ticks",
            "sorted", "block", "compact"]
    # Median over repetitions.
    return df.groupby(keys, as_index=False).median(numeric_only=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, name), dpi=150)
    plt.close(fig)
    print("wrote", os.path.join(PLOTS, name))


def amdahl(p, serial_frac):
    return 1.0 / (serial_frac + (1.0 - serial_frac) / p)


def fit_serial_fraction(p, s):
    """Least-squares fit of Amdahl's law; returns the serial fraction."""
    grid = np.linspace(0.0, 1.0, 10001)
    err = [np.sum((amdahl(p, f) - s) ** 2) for f in grid]
    return float(grid[int(np.argmin(err))])


def karp_flatt(p, s):
    p, s = np.asarray(p, float), np.asarray(s, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (1.0 / s - 1.0 / p) / (1.0 - 1.0 / p)


def strong():
    df = load("strong")
    if df is None:
        return
    omp = df[df["label"] == "omp"].sort_values("threads")
    ser = df[df["label"] == "serial"]
    base = float(ser["t_total"].iloc[0]) if len(ser) else float(omp["t_total"].iloc[0])
    p = omp["threads"].to_numpy(float)
    s = base / omp["t_total"].to_numpy(float)
    f = fit_serial_fraction(p, s)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(p, s, "o-", label="measured")
    pp = np.linspace(1, p.max(), 100)
    ax[0].plot(pp, amdahl(pp, f), "--", label=f"Amdahl fit, serial fraction = {f:.3f}")
    ax[0].plot(pp, pp, ":", color="gray", label="ideal")
    ax[0].set_xlabel("OpenMP threads")
    ax[0].set_ylabel("speedup vs serial")
    ax[0].set_title("Strong scaling")
    ax[0].legend()
    # Phase breakdown per thread count.
    bottom = np.zeros(len(omp))
    for ph in ["t_decide", "t_match", "t_publish"]:
        v = omp[ph].to_numpy(float)
        ax[1].bar([str(int(x)) for x in p], v, bottom=bottom, label=ph[2:])
        bottom += v
    ax[1].set_xlabel("OpenMP threads")
    ax[1].set_ylabel("seconds")
    ax[1].set_title("Where the time goes")
    ax[1].legend()
    save(fig, "strong_scaling.png")
    kf = karp_flatt(p[1:], s[1:])
    summary.append("## Strong scaling (OpenMP)")
    summary.append(f"- Max speedup {s.max():.2f}x on {int(p[np.argmax(s)])} threads")
    summary.append(f"- Amdahl fit: serial fraction = **{f:.3f}** (max possible speedup {1 / max(f, 1e-9):.1f}x)")
    summary.append("- Karp-Flatt serial fraction per thread count: " +
                   ", ".join(f"{int(a)}: {b:.3f}" for a, b in zip(p[1:], kf)))
    summary.append("")


def weak():
    df = load("weak")
    if df is None:
        return
    df = df.sort_values("threads")
    p = df["threads"].to_numpy(float)
    eff = df["t_total"].iloc[0] / df["t_total"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(p, eff, "o-", label="weak-scaling efficiency")
    ax.plot(p, p * eff, "s-", label="scaled speedup (Gustafson)")
    ax.plot(p, p, ":", color="gray", label="ideal scaled speedup")
    ax.set_xlabel("OpenMP threads (problem grows with threads)")
    ax.legend()
    ax.set_title("Weak scaling")
    save(fig, "weak_scaling.png")
    summary.append("## Weak scaling (OpenMP)")
    summary.append("- Efficiency: " + ", ".join(f"{int(a)} thr: {b:.2f}" for a, b in zip(p, eff)))
    summary.append("")


def instruments():
    df = load("instruments")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for lab, g in df.groupby("match"):
        g = g.sort_values("instruments")
        ax.plot(g["instruments"], g["t_match"] / g["ticks"] * 1e3, "o-", label=f"{lab} matching")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("instruments (agents fixed)")
    ax.set_ylabel("matching time per tick (ms)")
    ax.set_title("Matching parallelises only across instruments")
    ax.legend()
    save(fig, "instruments.png")
    a = df[df["match"] == "serial"].set_index("instruments")["t_match"]
    b = df[df["match"] == "omp"].set_index("instruments")["t_match"]
    sp = (a / b).dropna()
    summary.append("## Matching vs instrument count")
    summary.append("- Matching speedup (OpenMP over serial): " +
                   ", ".join(f"{int(k)} instr: {v:.2f}x" for k, v in sp.items()))
    summary.append("")


def crossover():
    df = load("crossover")
    if df is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for lab, g in df.groupby("decide"):
        g = g.sort_values("agents")
        ax[0].plot(g["agents"], g["agent_updates_per_sec"], "o-", label=lab)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("agents")
    ax[0].set_ylabel("agent updates per second")
    ax[0].set_title("Throughput by back end")
    ax[0].legend()
    cu = df[df["decide"] == "cuda"].sort_values("agents")
    if len(cu):
        x = [str(int(a)) for a in cu["agents"]]
        bottom = np.zeros(len(cu))
        for ph in ["t_h2d", "t_kernel", "t_d2h", "t_match", "t_publish"]:
            v = (cu[ph] / cu["t_total"]).to_numpy(float)
            ax[1].bar(x, v, bottom=bottom, label=ph[2:])
            bottom += v
        ax[1].set_ylabel("share of tick time")
        ax[1].set_xlabel("agents")
        ax[1].set_title("CUDA run: phase shares")
        ax[1].tick_params(axis="x", rotation=45)
        ax[1].legend(fontsize=8)
    save(fig, "crossover.png")
    summary.append("## CPU vs GPU crossover")
    om = df[df["decide"] == "omp"].set_index("agents")["t_total"]
    if len(cu):
        c = cu.set_index("agents")["t_total"]
        ratio = (om / c).dropna()
        summary.append("- CUDA speedup over the multicore OpenMP baseline: " +
                       ", ".join(f"{int(k)}: {v:.2f}x" for k, v in ratio.items()))
        wins = ratio[ratio > 1.0]
        summary.append(f"- CUDA overtakes OpenMP from **{int(wins.index.min())} agents**" if len(wins)
                       else "- CUDA did not overtake OpenMP in the tested range")
        boundary = ((cu["t_h2d"] + cu["t_d2h"]) / cu["t_total"]).to_numpy()
        summary.append("- CPU-GPU copy share of tick time: " +
                       ", ".join(f"{int(a)}: {b:.1%}" for a, b in zip(cu["agents"], boundary)))
    summary.append("")


def gustafson():
    df = load("gustafson")
    if df is None:
        return
    s = df[df["decide"] == "serial"].set_index("agents")
    c = df[df["decide"] == "cuda"].set_index("agents")
    common = s.index.intersection(c.index)
    if len(common) == 0:
        return
    speed = s.loc[common, "t_total"] / c.loc[common, "t_total"]
    serial_share = c.loc[common, "t_match"] / c.loc[common, "t_total"]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(common, speed, "o-", label="speedup over serial")
    ax2 = ax.twinx()
    ax2.plot(common, serial_share, "s--", color="C1", label="matching share (serial core)")
    ax.set_xscale("log")
    ax.set_xlabel("agents (32 instruments)")
    ax.set_ylabel("speedup")
    ax2.set_ylabel("share of tick time in matching")
    ax.set_title("Amdahl vs Gustafson")
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
    save(fig, "gustafson.png")
    summary.append("## Growing agents with fixed instruments")
    summary.append("- Speedup: " + ", ".join(f"{int(k)}: {v:.2f}x" for k, v in speed.items()))
    summary.append("- Matching share of tick time: " +
                   ", ".join(f"{int(k)}: {v:.1%}" for k, v in serial_share.items()))
    summary.append("")


def cuda():
    blk = load("cuda_block")
    abl = load("cuda_ablation")
    if blk is None and abl is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    summary.append("## CUDA kernel experiments")
    if blk is not None:
        blk = blk.sort_values("block")
        ax[0].plot(blk["block"], blk["t_kernel"] / blk["ticks"] * 1e3, "o-")
        ax[0].set_xscale("log", base=2)
        ax[0].set_xlabel("threads per block")
        ax[0].set_ylabel("kernel time per tick (ms)")
        ax[0].set_title("Block size")
        best = blk.loc[blk["t_kernel"].idxmin()]
        summary.append(f"- Best block size: {int(best['block'])}")
    if abl is not None:
        x = abl["label"].tolist()
        k = (abl["t_kernel"] / abl["ticks"] * 1e3).to_numpy()
        d = (abl["t_d2h"] / abl["ticks"] * 1e3).to_numpy()
        ax[1].bar(x, k, label="kernel")
        ax[1].bar(x, d, bottom=k, label="device-to-host copy")
        ax[1].set_ylabel("ms per tick")
        ax[1].set_title("Type sorting and compaction")
        ax[1].tick_params(axis="x", rotation=20)
        ax[1].legend()
        for _, r in abl.iterrows():
            summary.append(f"- {r['label']}: kernel {r['t_kernel'] / r['ticks'] * 1e3:.3f} ms/tick, "
                           f"copy back {r['t_d2h'] / r['ticks'] * 1e3:.3f} ms/tick")
    save(fig, "cuda_experiments.png")
    summary.append("")


def mpi():
    df = load("mpi")
    if df is None:
        return
    df = df.sort_values("ranks")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    x = [str(int(r)) for r in df["ranks"]]
    bottom = np.zeros(len(df))
    for ph in ["t_decide", "t_match", "t_comm"]:
        v = df[ph].to_numpy(float)
        ax.bar(x, v, bottom=bottom, label=ph[2:])
        bottom += v
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("seconds (slowest rank)")
    ax.set_title("MPI: compute vs communication")
    ax.legend()
    save(fig, "mpi.png")
    summary.append("## MPI")
    summary.append("- Communication + barrier share of tick time: " +
                   ", ".join(f"{int(r)} ranks: {c / t:.1%}" for r, c, t in
                             zip(df["ranks"], df["t_comm"], df["t_total"])))
    summary.append("")


def falseshare():
    df = load("falseshare")
    if df is None:
        return
    summary.append("## False sharing")
    for _, r in df.iterrows():
        summary.append(f"- {r['label']}: matching {r['t_match'] / r['ticks'] * 1e3:.3f} ms/tick")
    summary.append("")


for fn in [strong, weak, instruments, crossover, gustafson, cuda, mpi, falseshare]:
    fn()
with open(os.path.join(RES, "summary.md"), "w") as f:
    f.write("\n".join(summary) + "\n")
print("wrote", os.path.join(RES, "summary.md"))
