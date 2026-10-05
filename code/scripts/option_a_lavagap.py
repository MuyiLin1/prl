#!/usr/bin/env python3
"""Direction C / Option A pipeline: does geometry predict SHAPED difficulty on LavaGap?

LavaGap is the canonical "structure gates progress" task: a vertical lava wall
with a single gap. The geometry hypothesis says difficulty is governed by WHERE
the gap is (path length / detour the agent must take).

For each grid (env seed):
  - extract geometric features (gap position, BFS path length start->goal, detour)
  - run R HC search seeds with distance-shaped reward, average the difficulty
Then test whether geometry predicts the stabilized shaped difficulty.
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
from prog_policies.minigrid_tasks.lavagap import LavaGapEnv_modified
from minigrid_shaped import run_hc_shaped_lavagap, bfs_dist_to_goal, _passable


def lavagap_features(env_seed, size):
    env = LavaGapEnv_modified(size=size)
    env.reset(seed=env_seed)
    u = env
    gx, gy = int(u.gap_pos[0]), int(u.gap_pos[1])
    start = (int(u.agent_pos[0]), int(u.agent_pos[1]))
    goal = (int(u.goal_pos[0]), int(u.goal_pos[1]))

    # BFS path length start -> goal (through the single gap)
    dist_from_goal = bfs_dist_to_goal(u.grid, u.goal_pos, u.width, u.height)
    path_len = dist_from_goal.get(start, -1)

    manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    detour = path_len - manhattan if path_len >= 0 else -1

    # gap offset relative to the straight diagonal line between start and goal
    # (how far the gap row is from where the agent would naturally cross)
    gap_row_offset = abs(gy - start[1])
    gap_col = gx

    return {
        "gap_x": gx,
        "gap_y": gy,
        "gap_col": gap_col,
        "gap_row_offset": gap_row_offset,
        "path_len": path_len,
        "manhattan": manhattan,
        "detour": detour,
        "size": size,
    }


def split_half_reliability(diff):
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
                rhos.append(2 * rho / (1 + rho))
    return float(np.mean(rhos)) if rhos else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", type=int, default=60)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--size", type=int, default=9)
    ap.add_argument("--n-iter", type=int, default=2000)
    ap.add_argument("--output", default="option_a_lavagap.csv")
    args = ap.parse_args()

    print(f"Option A: shaped LavaGap, size={args.size}, "
          f"{args.grids} grids x {args.reps} HC seeds\n")

    feats = []
    diff = np.zeros((args.grids, args.reps))
    for g in range(args.grids):
        feats.append(lavagap_features(g, args.size))
        for s in range(args.reps):
            search_seed = 100_000 * (s + 1) + g
            best = run_hc_shaped_lavagap(g, search_seed, n_iterations=args.n_iter,
                                         size=args.size)
            diff[g, s] = 1.0 - best
        print(f"  grid {g:>3}: mean_diff={diff[g].mean():.3f} std={diff[g].std():.3f} "
              f"path_len={feats[g]['path_len']}")

    mean_diff = diff.mean(axis=1)
    within_std = diff.std(axis=1).mean()
    reliability = split_half_reliability(diff)

    fdf = pd.DataFrame(feats)
    feature_cols = [c for c in fdf.columns if fdf[c].std() > 1e-9]

    # Best single-feature Spearman
    best_feat, best_rho = None, 0.0
    for c in feature_cols:
        rho, _ = spearmanr(fdf[c].values, mean_diff)
        if not np.isnan(rho) and abs(rho) > abs(best_rho):
            best_rho, best_feat = rho, c

    # Multivariate RF 5-fold CV
    if mean_diff.std() < 1e-9 or len(feature_cols) == 0:
        rf_cv = float("nan")
    else:
        X = fdf[feature_cols].values
        oof = np.zeros(args.grids)
        for tr, te in KFold(n_splits=5, shuffle=True, random_state=0).split(X):
            rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0)
            rf.fit(X[tr], mean_diff[tr])
            oof[te] = rf.predict(X[te])
        rf_cv, _ = spearmanr(oof, mean_diff)

    # save
    out = fdf.copy()
    out.insert(0, "grid_seed", range(args.grids))
    out["mean_difficulty"] = mean_diff
    out["std_difficulty"] = diff.std(axis=1)
    out.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            args.output), index=False)

    print("\n" + "=" * 70)
    print("OPTION A RESULT — geometry vs SHAPED LavaGap difficulty")
    print("=" * 70)
    print(f"grids={args.grids} reps={args.reps} size={args.size}")
    print(f"difficulty mean={mean_diff.mean():.3f} std={mean_diff.std():.3f} "
          f"range=[{mean_diff.min():.2f},{mean_diff.max():.2f}]")
    print(f"within-grid std (HC noise)={within_std:.3f}")
    print(f"label reliability (split-half)={reliability:.3f}")
    print(f"best single feature: {best_feat}  |rho|={abs(best_rho):.3f}")
    print(f"multivariate RF 5-fold CV rho={rf_cv:.3f}")
    # per-feature correlations
    print("\nper-feature Spearman vs difficulty:")
    for c in feature_cols:
        rho, _ = spearmanr(fdf[c].values, mean_diff)
        print(f"  {c:<16} rho={rho:+.3f}")

    verdict = ("GEOMETRY PREDICTS (positive pole!)" if (not np.isnan(rf_cv) and rf_cv >= 0.5)
               else "geometry partial" if (not np.isnan(rf_cv) and rf_cv >= 0.3)
               else "geometry FAILS")
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
