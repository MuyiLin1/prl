#!/usr/bin/env python3
"""Cheap-rollout probe vs geometric descriptor on the RELIABLE averaged label.

The averaged-label files give a stabilized (R=12) difficulty per grid for TopOff
and Snake (reliable: split-half rho 0.57 / 0.71). The multiseed_with_rollout.csv
file gives cheap-rollout probe features (rewards of a handful of fixed simple
programs + random-program reward stats) for seeds 0-49 of each task.

We JOIN the two on (task, grid_seed==seed) for the overlapping 50 grids per task,
then compare, against the SAME reliable label:
  - geometry descriptor   (graph_* + spectral_* + reward_* + grid_*)  -> RF 5-fold CV
  - cheap-rollout probe   (probe_*_reward + random_*_reward + best_probe_reward) -> RF 5-fold CV
  - best single probe feature (Spearman)

This isolates whether a cheap dynamics-aware probe recovers difficulty that the
static geometric descriptor misses.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rf_cv(X, y, seed=0):
    X = np.nan_to_num(X.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    if np.std(y) < 1e-9 or X.shape[1] == 0:
        return float("nan")
    oof = np.zeros(len(y))
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=seed).split(X):
        rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=seed)
        rf.fit(X[tr], y[tr])
        oof[te] = rf.predict(X[te])
    rho, _ = spearmanr(oof, y)
    return float(rho)


def best_single(df, cols, y):
    bf, br = None, 0.0
    for c in cols:
        if df[c].std() < 1e-9:
            continue
        rho, _ = spearmanr(df[c].values, y)
        if not np.isnan(rho) and abs(rho) > abs(br):
            br, bf = rho, c
    return bf, br


probe = pd.read_csv(os.path.join(ROOT, "multiseed_with_rollout.csv"))
probe_cols = [c for c in probe.columns
              if (c.startswith("probe_") and c.endswith("_reward"))
              or c.startswith("random_") or c == "best_probe_reward"]

label_files = {
    "TopOff": "averaged_label_topoff200.csv",
    "Snake": "averaged_label_screen_snake_seeder.csv",
}

print("=" * 78)
print("CHEAP-ROLLOUT PROBE vs GEOMETRY on the reliable averaged label")
print("=" * 78)

for task, fname in label_files.items():
    av = pd.read_csv(os.path.join(ROOT, fname))
    av = av[av["task"] == task].copy()
    geo_cols = [c for c in av.columns
                if c not in ("task", "grid_seed", "mean_difficulty", "std_difficulty")]

    pr = probe[probe["task_name"] == task].copy()
    merged = av.merge(pr, left_on="grid_seed", right_on="seed", how="inner",
                      suffixes=("", "_probe"))
    if len(merged) < 10:
        print(f"\n[{task}] only {len(merged)} joined grids -- skipping")
        continue

    y = merged["mean_difficulty"].values
    geo_rho = rf_cv(merged[geo_cols].values, y)
    probe_rho = rf_cv(merged[probe_cols].values, y)
    bf, br = best_single(merged, probe_cols, y)
    gbf, gbr = best_single(merged, geo_cols, y)

    print(f"\n[{task}]  joined grids = {len(merged)}  (reliable averaged label)")
    print(f"   label std={y.std():.3f}")
    print(f"   geometry        RF 5-fold CV rho = {geo_rho:+.3f}   best single: {gbf} rho={gbr:+.3f}")
    print(f"   cheap probe     RF 5-fold CV rho = {probe_rho:+.3f}   best single: {bf} rho={br:+.3f}")
