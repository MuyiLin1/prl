#!/usr/bin/env python3
"""Direction C combined collector: Options A + B from one set of shaped-HC runs.

Per (grid, rep) on shaped LavaGap, record:
  - shaped_best   : best shaped reward in [0,1]   -> Option A difficulty = 1 - mean(shaped_best)
  - solved        : 1 if the run reached the goal (shaped_best == 1.0)
  - evals_used    : number of programs evaluated (wrapper.program_num) -> search cost

Derives three difficulty instruments per grid:
  A  partial-progress difficulty = 1 - mean(shaped_best)
  B1 success-rate difficulty      = 1 - fraction_solved
  B2 search-cost difficulty       = mean(evals_used) normalised

Then tests geometry (gap features) against each.
"""
import sys
import os
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'leaps'))

from prog_policies.utils import get_env_name  # noqa: F401 - resolves circular import
from prog_policies.minigrid.dsl import MinigridDSL
from prog_policies.search_space import MinigridProgrammaticSpace
from prog_policies.search_methods import HillClimbing
from minigrid_shaped import make_shaped_lavagap, bfs_dist_to_goal
from option_a_lavagap import lavagap_features, split_half_reliability


def run_one(env_seed, search_seed, n_iterations, size):
    dsl = MinigridDSL()
    env = make_shaped_lavagap(env_seed, size=size)
    task_envs = [env]
    search_space = MinigridProgrammaticSpace(dsl, sigma=0.25)
    search_space.set_seed(search_seed)
    hc = HillClimbing(k=250, e=2)
    _, rewards = hc.search(search_space, task_envs, seed=search_seed, n_iterations=n_iterations)
    best = max(rewards) if rewards else 0.0
    solved = 1.0 if best >= 1.0 else 0.0
    evals = env.program_num  # programs evaluated through the wrapper
    return best, solved, evals


def rf_cv(fdf, feature_cols, y):
    use = [c for c in feature_cols if fdf[c].std() > 1e-9]
    if len(use) == 0 or np.std(y) < 1e-9:
        return float("nan"), use
    X = fdf[use].values
    oof = np.zeros(len(y))
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=0).split(X):
        rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0)
        rf.fit(X[tr], y[tr])
        oof[te] = rf.predict(X[te])
    rho, _ = spearmanr(oof, y)
    return float(rho), use


def best_single(fdf, feature_cols, y):
    bf, br = None, 0.0
    for c in feature_cols:
        if fdf[c].std() < 1e-9:
            continue
        rho, _ = spearmanr(fdf[c].values, y)
        if not np.isnan(rho) and abs(rho) > abs(br):
            br, bf = rho, c
    return bf, br


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", type=int, default=60)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--size", type=int, default=9)
    ap.add_argument("--n-iter", type=int, default=2000)
    ap.add_argument("--output", default="direction_c_AB.csv")
    args = ap.parse_args()

    print(f"Direction C A+B collector: shaped LavaGap size={args.size}, "
          f"{args.grids} grids x {args.reps} reps\n", flush=True)

    feats = []
    shaped = np.zeros((args.grids, args.reps))
    solved = np.zeros((args.grids, args.reps))
    evals = np.zeros((args.grids, args.reps))

    for g in range(args.grids):
        feats.append(lavagap_features(g, args.size))
        for s in range(args.reps):
            ss = 100_000 * (s + 1) + g
            b, sol, ev = run_one(g, ss, args.n_iter, args.size)
            shaped[g, s] = b
            solved[g, s] = sol
            evals[g, s] = ev
        print(f"  grid {g:>3}: mean_shaped={shaped[g].mean():.3f} "
              f"success={solved[g].mean():.2f} mean_evals={evals[g].mean():.0f}", flush=True)

    # difficulty instruments
    diff_A = 1.0 - shaped.mean(axis=1)               # partial-progress
    diff_B1 = 1.0 - solved.mean(axis=1)              # success-rate
    diff_B2 = evals.mean(axis=1)                     # search cost (raw)
    diff_B2n = (diff_B2 - diff_B2.min()) / (diff_B2.ptp() + 1e-9)

    fdf = pd.DataFrame(feats)
    feat_cols = [c for c in fdf.columns]

    # reliability of each label
    rel_A = split_half_reliability(shaped)
    rel_B1 = split_half_reliability(solved)
    rel_B2 = split_half_reliability(evals)

    # save raw
    out = fdf.copy()
    out.insert(0, "grid_seed", range(args.grids))
    out["diff_A_partial"] = diff_A
    out["diff_B1_successrate"] = diff_B1
    out["diff_B2_evals"] = diff_B2
    out.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            args.output), index=False)
    np.savez(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          args.output.replace(".csv", "_per_search.npz")),
             shaped=shaped, solved=solved, evals=evals)

    print("\n" + "=" * 78)
    print("DIRECTION C — Options A & B: geometry vs difficulty instruments (shaped LavaGap)")
    print("=" * 78)
    print(f"grids={args.grids} reps={args.reps} size={args.size}\n")

    for name, y, rel in [
        ("A  partial-progress (1-shaped)", diff_A, rel_A),
        ("B1 success-rate (1-solved)", diff_B1, rel_B1),
        ("B2 search-cost (evals)", diff_B2n, rel_B2),
    ]:
        bf, br = best_single(fdf, feat_cols, y)
        rho, use = rf_cv(fdf, feat_cols, y)
        verdict = ("GEOMETRY PREDICTS" if (not np.isnan(rho) and rho >= 0.5)
                   else "partial" if (not np.isnan(rho) and rho >= 0.3)
                   else "geometry FAILS")
        print(f"[{name}]")
        print(f"   difficulty std={y.std():.3f}  label_reliability={rel:.3f}")
        print(f"   best single: {bf} rho={br:+.3f}")
        print(f"   RF 5-fold CV rho={rho:.3f}  -> {verdict}")
        # per-feature
        fl = []
        for c in feat_cols:
            if fdf[c].std() < 1e-9:
                continue
            r, _ = spearmanr(fdf[c].values, y)
            fl.append(f"{c}={r:+.2f}")
        print("   per-feature: " + ", ".join(fl) + "\n")


if __name__ == "__main__":
    main()
