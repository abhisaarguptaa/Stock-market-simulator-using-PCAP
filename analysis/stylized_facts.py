"""Checks simulated prices against the standard stylised facts of returns
(Cont, 2001): heavy tails, no linear autocorrelation of returns, and
volatility clustering (slowly decaying autocorrelation of |returns|).

usage: python analysis/stylized_facts.py prices.csv [--horizon 10] [--out results/facts]
prices.csv comes from `lobsim --prices`. Multiple files (MPI ranks) can be given.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def acf(x, lags):
    x = np.asarray(x, dtype=float) - np.mean(x)
    denom = np.dot(x, x)
    return np.array([np.dot(x[:-k], x[k:]) / denom if denom > 0 else 0.0 for k in lags])


def hill_tail_index(x, frac=0.05):
    """Hill estimator of the tail exponent on the largest |x| values."""
    a = np.sort(np.abs(x[x != 0]))[::-1]
    k = max(int(len(a) * frac), 10)
    if len(a) <= k:
        return float("nan")
    return float(1.0 / np.mean(np.log(a[:k] / a[k])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--horizon", type=int, default=10, help="return horizon in ticks")
    ap.add_argument("--burn", type=int, default=200, help="ticks to discard at the start")
    ap.add_argument("--out", default="results/facts")
    args = ap.parse_args()

    df = pd.concat([pd.read_csv(f) for f in args.files])
    os.makedirs(args.out, exist_ok=True)
    lags = np.arange(1, 51)
    rets, acf_r, acf_abs = [], [], []
    for _, g in df.groupby("instrument"):
        p = g.sort_values("tick")["price"].to_numpy(dtype=float)[args.burn:]
        r = np.diff(np.log(p[:: args.horizon]))
        if len(r) < 60 or np.std(r) == 0:
            continue
        rets.append((r - r.mean()) / r.std())
        acf_r.append(acf(r, lags))
        acf_abs.append(acf(np.abs(r), lags))
    if not rets:
        raise SystemExit("not enough data: run more ticks or lower --horizon")
    z = np.concatenate(rets)
    acf_r = np.mean(acf_r, axis=0)
    acf_abs = np.mean(acf_abs, axis=0)
    kurt = float(np.mean(z**4) - 3.0)
    summary = {
        "instruments": len(rets),
        "returns": int(len(z)),
        "horizon_ticks": args.horizon,
        "excess_kurtosis": kurt,                      # > 0 means heavier tails than Gaussian
        "hill_tail_index": hill_tail_index(z),        # real markets: roughly 3 to 5
        "acf_returns_lag1": float(acf_r[0]),          # should be near 0
        "acf_abs_returns_lag1": float(acf_abs[0]),    # should be clearly > 0
        "acf_abs_returns_lag10": float(acf_abs[9]),   # and decay slowly
        "heavy_tails": bool(kurt > 1.0),
        "no_linear_autocorrelation": bool(abs(acf_r[0]) < 0.1),
        "volatility_clustering": bool(acf_abs[0] > 0.1 and acf_abs[9] > 0.02),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy import stats
    except ImportError:
        print("matplotlib/scipy not installed: skipping plots")
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    xs = np.linspace(-6, 6, 400)
    ax[0].hist(z, bins=120, density=True, alpha=0.6, label="simulated")
    ax[0].plot(xs, stats.norm.pdf(xs), label="Gaussian")
    ax[0].set_yscale("log")
    ax[0].set_ylim(1e-5, 1)
    ax[0].set_title(f"Return distribution (excess kurtosis {kurt:.2f})")
    ax[0].set_xlabel("standardised return")
    ax[0].legend()
    ax[1].bar(lags, acf_r)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_title("ACF of returns")
    ax[1].set_xlabel("lag")
    ax[2].plot(lags, acf_abs, marker="o", ms=3)
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_title("ACF of |returns| (volatility clustering)")
    ax[2].set_xlabel("lag")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "stylized_facts.png"), dpi=150)
    print("wrote", os.path.join(args.out, "stylized_facts.png"))


if __name__ == "__main__":
    main()
