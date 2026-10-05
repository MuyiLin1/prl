#!/usr/bin/env python3
"""Analyze dirty-pool curriculum runs: difficulty-tier breakdown + proper stats.

Usage:
    python analyze_dirty_pool.py PREFIX [PREFIX ...]
where each PREFIX has PREFIX_results.csv and PREFIX_pool.csv next to it
(e.g. sweep2/p020). Prints, for each point:
  - the pool's difficulty-tier composition (this is the key diagnosis: medium==0)
  - per-condition mean, bootstrap 95% CI, and one-sided Mann-Whitney vs random.

A result is only worth reporting if a condition beats random with a *substantial*
gap (say >=0.05) AND non-overlapping CIs. +0.02 at p<0.05 is not substantial.
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    means = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])

def analyze(prefix):
    res = pd.read_csv(f"{prefix}_results.csv")
    print(f"\n{'='*70}\n{prefix}  ({len(res)} runs)\n{'='*70}")

    # ---- difficulty-tier composition of the pool (THE diagnosis) ----
    try:
        pool = pd.read_csv(f"{prefix}_pool.csv")
        d = pool["d_probe"]
        lo, hi = d.quantile([0.33, 0.67])
        easy = int((d <= lo).sum()); med = int(((d > lo) & (d < hi)).sum()); hard = int((d >= hi).sum())
        print(f" pool: N={len(pool)}, d_probe range {d.min():.2f}-{d.max():.2f}, mean {d.mean():.2f}")
        print(f"       tiers -> easy={easy}  medium={med}  hard={hard}")
        if med == 0:
            print("       *** WARNING: medium tier is EMPTY -> degenerate (binary) label. ***")
            print("       *** Curation has no graded structure to exploit here.        ***")
    except FileNotFoundError:
        print(" (no _pool.csv found; skipping tier breakdown)")

    # ---- per-condition stats vs random ----
    if "random" not in set(res.condition):
        print(" (no 'random' condition; skipping stats)"); return
    r = res[res.condition == "random"].heldout_reward.values
    order = [c for c in ["random", "band", "coverage", "both", "full"] if c in set(res.condition)]
    print(f"\n {'condition':10s} {'mean':>7s} {'95% CI':>18s}  {'Δ vs rnd':>9s} {'p(>rnd)':>8s}")
    for c in order:
        x = res[res.condition == c].heldout_reward.values
        lo, hi = boot_ci(x)
        line = f" {c:10s} {x.mean():7.3f}  [{lo:6.3f},{hi:6.3f}]"
        if c not in ("random", "full") and len(x) > 1 and len(r) > 1:
            _, p = mannwhitneyu(x, r, alternative="greater")
            line += f"  {x.mean()-r.mean():+9.3f} {p:8.3f}"
        print(line)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    for pfx in sys.argv[1:]:
        analyze(pfx)
