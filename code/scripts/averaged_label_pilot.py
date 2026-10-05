#!/usr/bin/env python3
"""Averaged-label pilot (Direction B, stabilized).

The single-run HC difficulty is HC-seed noise (mean self_rho ~ 0.03). Here we
redefine difficulty as the MEAN over R HC seeds per grid, quantify how much REAL
across-grid signal survives averaging, then re-test whether geometry predicts the
stabilized label within each task.

Outputs:
  - per-grid mean difficulty + std (saved to CSV)
  - reliability: signal-to-noise of the averaged label
  - within-task geometry test against the stabilized label (best single |rho|
    and 5-fold RF CV rho)
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'leaps'))

from prog_policies.utils import get_env_name  # noqa: F401 - resolves circular import
from prog_policies.karel_tasks import get_task_cls
from prog_policies.karel import KarelDSL
from prog_policies.search_space import ProgrammaticSpace
from prog_policies.search_methods import HillClimbing

from generate_multiseed_dataset import get_env_args, extract_features_from_state

parser = argparse.ArgumentParser()
parser.add_argument("--tasks", nargs="+", default=["TopOff", "Harvester"])
parser.add_argument("--grids", type=int, default=80)
parser.add_argument("--reps", type=int, default=12, help="HC search seeds per grid")
parser.add_argument("--max-programs", type=int, default=5000)
parser.add_argument("--output", default="averaged_label_pilot.csv")
parser.add_argument("--summary-output", default=None,
                    help="optional path to write the per-task summary table as CSV")
args = parser.parse_args()

TASKS = args.tasks
N_GRIDS = args.grids
R = args.reps               # HC search seeds per grid
MAX_PROGRAMS = args.max_programs

NON_FEATURE = {"task_name", "seed", "hc_best_reward", "difficulty"}


def run_hc(task_name, env_seed, search_seed, max_programs=MAX_PROGRAMS):
    dsl = KarelDSL()
    task_cls = get_task_cls(task_name)
    env_args = get_env_args(task_name)
    task_envs = [task_cls(env_args, env_seed)]
    search_space = ProgrammaticSpace(dsl, sigma=0.1)
    search_space.set_seed(search_seed)
    search_method = HillClimbing(k=250, e=2)
    _, rewards = search_method.search(
        search_space, task_envs, seed=search_seed, n_iterations=max_programs
    )
    return 1.0 - (max(rewards) if rewards else 0.0)


def get_features(task_name, env_seed):
    task_cls = get_task_cls(task_name)
    env_args = get_env_args(task_name)
    task = task_cls(env_args, env_seed)
    state = task.initial_environment.state
    return extract_features_from_state(state, task_name)


def split_half_reliability(diff):
    """Spearman-Brown corrected split-half reliability of the averaged label."""
    g, r = diff.shape
    if r < 2:
        return float("nan")
    cols = np.arange(r)
    rhos = []
    rng = np.random.default_rng(0)
    for _ in range(20):
        perm = rng.permutation(cols)
        a, b = perm[: r // 2], perm[r // 2:]
        ma, mb = diff[:, a].mean(axis=1), diff[:, b].mean(axis=1)
        if ma.std() > 1e-9 and mb.std() > 1e-9:
            rho, _ = spearmanr(ma, mb)
            if not np.isnan(rho):
                # Spearman-Brown step up from half-length to full-length
                rhos.append(2 * rho / (1 + rho))
    return float(np.mean(rhos)) if rhos else float("nan")


all_rows = []
summary = []

for task in TASKS:
    print(f"\n{'='*70}\nTask: {task}  ({N_GRIDS} grids x {R} HC seeds)\n{'='*70}")
    diff = np.zeros((N_GRIDS, R))
    feats = []
    for g in range(N_GRIDS):
        f = get_features(task, g)
        feats.append(f if f is not None else {})
        for s in range(R):
            search_seed = 100_000 * (s + 1) + g
            diff[g, s] = run_hc(task, env_seed=g, search_seed=search_seed)
        print(f"  grid {g:>3}: mean_diff={diff[g].mean():.3f} std={diff[g].std():.3f}")

    np.save(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         args.output.replace(".csv", f"_{task}_per_search.npy")), diff)
    mean_diff = diff.mean(axis=1)
    within_std = diff.std(axis=1).mean()

    # Variance decomposition: total across-grid var of mean label vs noise floor
    var_mean_label = mean_diff.var()
    noise_var_of_mean = (diff.var(axis=1) / R).mean()   # var of a mean of R draws
    signal_var = max(0.0, var_mean_label - noise_var_of_mean)
    snr = signal_var / noise_var_of_mean if noise_var_of_mean > 1e-12 else float("inf")
    reliability = split_half_reliability(diff)

    # Build feature frame
    fdf = pd.DataFrame(feats)
    fdf = fdf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_cols = [c for c in fdf.columns if c not in NON_FEATURE]
    usable = [c for c in feature_cols if fdf[c].std() > 1e-9]

    # Best single-feature Spearman vs stabilized label
    best_feat, best_rho = None, 0.0
    for c in usable:
        rho, _ = spearmanr(fdf[c].values, mean_diff)
        if not np.isnan(rho) and abs(rho) > abs(best_rho):
            best_rho, best_feat = rho, c

    # Multivariate RF, 5-fold CV vs stabilized label
    if mean_diff.std() < 1e-9 or len(usable) == 0:
        rf_cv = float("nan")
    else:
        X = fdf[usable].values
        oof = np.zeros(N_GRIDS)
        for tr, te in KFold(n_splits=5, shuffle=True, random_state=0).split(X):
            rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0)
            rf.fit(X[tr], mean_diff[tr])
            oof[te] = rf.predict(X[te])
        rf_cv, _ = spearmanr(oof, mean_diff)

    summary.append({
        "task": task,
        "across_grid_std(mean_label)": round(float(np.sqrt(var_mean_label)), 3),
        "within_grid_std": round(float(within_std), 3),
        "signal_std": round(float(np.sqrt(signal_var)), 3),
        "SNR": round(float(snr), 2),
        "label_reliability": round(float(reliability), 3),
        "best_feature": best_feat,
        "best_single_|rho|": round(abs(best_rho), 3),
        "rf_cv_rho": round(float(rf_cv), 3) if not np.isnan(rf_cv) else np.nan,
    })

    for g in range(N_GRIDS):
        row = {"task": task, "grid_seed": g,
               "mean_difficulty": mean_diff[g], "std_difficulty": diff[g].std()}
        row.update(feats[g] if feats[g] else {})
        all_rows.append(row)

pd.DataFrame(all_rows).to_csv(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 args.output), index=False)

print("\n" + "=" * 90)
print("AVERAGED-LABEL PILOT SUMMARY")
print("=" * 90)
sdf = pd.DataFrame(summary)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print(sdf.to_string(index=False))

if args.summary_output:
    sdf.to_csv(args.summary_output, index=False)

print("\nInterpretation per task:")
for _, r in sdf.iterrows():
    rel = r["label_reliability"]
    rf = r["rf_cv_rho"]
    if np.isnan(rel) or rel < 0.5:
        stab = "label still UNRELIABLE (need more HC seeds)"
    elif r["SNR"] < 0.3:
        stab = "stable but TINY real signal (intrinsically luck-driven)"
    else:
        stab = "stable label with real across-grid signal"
    if not np.isnan(rf) and rf >= 0.5:
        geo = "geometry PREDICTS it"
    elif not np.isnan(rf) and rf >= 0.3:
        geo = "geometry partially predicts"
    else:
        geo = "geometry FAILS (geometry-blind)"
    print(f"  {r['task']:<10} reliability={rel!s:>6} SNR={r['SNR']!s:>5} -> {stab}; rf_cv={rf!s:>6} -> {geo}")
