#!/usr/bin/env python3
"""Section 6.4 confirming experiments: strengthen the cheap-probe result.

Three parts, all with NO language model anywhere:

(i)  FULL-POPULATION fixed-library probe.
     Recompute the cheap fixed-library probe (cheap_rollout_features) for EVERY
     grid in the reliable averaged-label populations (TopOff = 200, Snake = 80)
     instead of the 50-grid join, then compare geometry vs probe against the same
     reliable label (best single feature + 5-fold CV RF).

(ii) TRUNCATED-SEARCH probe variant (Eq. mini-hc).
     From a RANDOM base program r_0 (no LLM), evaluate a handful (~T_probe) of
     mutations and keep the best reward; D_probe = 1 - best_reward. Report the
     Spearman correlation rho_s(D_probe, D_HC) against the reliable averaged label
     for Karel TopOff/Snake and for shaped MiniGrid LavaGap.

(iii) WALL-CLOCK cost: time the fixed-library probe, the truncated search, and a
      full reference-HC run per grid, to quantify the claimed speedup.

Run:
  python scripts/probe_full_eval.py            # full run
  python scripts/probe_full_eval.py --quick     # smoke test (few grids)
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.append(os.path.join(ROOT, "leaps"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

# Resolve circular import (prog_policies.utils must import first)
from prog_policies.utils import get_env_name  # noqa: F401
from prog_policies.karel_tasks import get_task_cls
from prog_policies.karel import KarelDSL
from prog_policies.search_space import ProgrammaticSpace
from prog_policies.search_methods import HillClimbing

from cheap_rollout_features import extract_rollout_features, get_env_args

LABEL_FILES = {
    "TopOff": "averaged_label_topoff200.csv",
    "Snake": "averaged_label_screen_snake_seeder.csv",
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
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
        if c not in df or df[c].std() < 1e-9:
            continue
        rho, _ = spearmanr(df[c].values, y)
        if not np.isnan(rho) and abs(rho) > abs(br):
            br, bf = rho, c
    return bf, br


# --------------------------------------------------------------------------- #
# (ii) truncated-search probe variant (Eq. mini-hc), NO LLM
# --------------------------------------------------------------------------- #
def truncated_search_difficulty_karel(task_name, env_seed, t_probe=10, search_seed=0):
    """From a random base program, evaluate ~t_probe mutations; keep best reward.

    Returns (D_probe, n_evals). NO language model: the base is a random program.
    """
    dsl = KarelDSL()
    task_cls = get_task_cls(task_name)
    env_args = get_env_args(task_name)
    task_envs = [task_cls(env_args, env_seed)]
    space = ProgrammaticSpace(dsl, sigma=0.1)
    space.set_seed(search_seed + 7919)
    hc = HillClimbing(k=t_probe, e=2)

    base_ind, base_prog = space.initialize_individual()
    best = hc.evaluate_program(base_prog, task_envs)
    n_evals = 1
    for _, prog in space.get_neighbors(base_ind, k=t_probe):
        r = hc.evaluate_program(prog, task_envs)
        n_evals += 1
        if r > best:
            best = r
    return 1.0 - best, n_evals


def truncated_search_difficulty_lavagap(env_seed, t_probe=10, search_seed=0, size=9):
    from prog_policies.minigrid.dsl import MinigridDSL
    from prog_policies.search_space import MinigridProgrammaticSpace
    from minigrid_shaped import make_shaped_lavagap

    dsl = MinigridDSL()
    task_envs = [make_shaped_lavagap(env_seed, size=size)]
    space = MinigridProgrammaticSpace(dsl, sigma=0.25)
    space.set_seed(search_seed + 7919)
    hc = HillClimbing(k=t_probe, e=2)

    base_ind, base_prog = space.initialize_individual()
    best = hc.evaluate_program(base_prog, task_envs)
    n_evals = 1
    for _, prog in space.get_neighbors(base_ind, k=t_probe):
        r = hc.evaluate_program(prog, task_envs)
        n_evals += 1
        if r > best:
            best = r
    return 1.0 - best, n_evals


# --------------------------------------------------------------------------- #
# (iii) full reference-HC run (for wall-clock comparison only)
# --------------------------------------------------------------------------- #
def full_hc_karel(task_name, env_seed, search_seed=0, max_programs=5000):
    dsl = KarelDSL()
    task_cls = get_task_cls(task_name)
    env_args = get_env_args(task_name)
    task_envs = [task_cls(env_args, env_seed)]
    space = ProgrammaticSpace(dsl, sigma=0.1)
    space.set_seed(search_seed)
    hc = HillClimbing(k=250, e=2)
    _, rewards = hc.search(space, task_envs, seed=search_seed, n_iterations=max_programs)
    return 1.0 - (max(rewards) if rewards else 0.0)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test on a few grids")
    ap.add_argument("--t-probe", type=int, default=10)
    ap.add_argument("--n-random", type=int, default=50)
    args = ap.parse_args()

    print("=" * 78)
    print("SECTION 6.4 CONFIRMING EXPERIMENTS (no LLM anywhere)")
    print("=" * 78)

    karel_summary = {}

    # ----- (i) + (ii) Karel: full-population probe + truncated search ------- #
    for task, fname in LABEL_FILES.items():
        av = pd.read_csv(os.path.join(ROOT, fname))
        av = av[av["task"] == task].copy().reset_index(drop=True)
        if args.quick:
            av = av.head(12)
        geo_cols = [c for c in av.columns
                    if c not in ("task", "grid_seed", "mean_difficulty", "std_difficulty")]

        rows = []
        t0 = time.time()
        for _, row in av.iterrows():
            seed = int(row["grid_seed"])
            feats = extract_rollout_features(task, seed, n_random=args.n_random)
            dprobe, _ = truncated_search_difficulty_karel(
                task, seed, t_probe=args.t_probe, search_seed=seed)
            feats["trunc_search_difficulty"] = dprobe
            feats["grid_seed"] = seed
            feats["mean_difficulty"] = float(row["mean_difficulty"])
            rows.append(feats)
        probe_df = pd.DataFrame(rows)
        merged = av.merge(probe_df.drop(columns=["mean_difficulty"]),
                          on="grid_seed", how="inner")
        elapsed = time.time() - t0

        y = merged["mean_difficulty"].values
        probe_cols = [c for c in probe_df.columns
                      if (c.startswith("probe_") and c.endswith("_reward"))
                      or c.startswith("random_") or c == "best_probe_reward"]
        geo_rho = rf_cv(merged[geo_cols].values, y)
        probe_rho = rf_cv(merged[probe_cols].values, y)
        gbf, gbr = best_single(merged, geo_cols, y)
        pbf, pbr = best_single(merged, probe_cols, y)
        trunc_rho, _ = spearmanr(merged["trunc_search_difficulty"].values, y)

        out = os.path.join(ROOT, f"probe_full_{task.lower()}.csv")
        merged.to_csv(out, index=False)
        karel_summary[task] = dict(n=len(merged), geo_rho=geo_rho, probe_rho=probe_rho,
                                   gbf=gbf, gbr=gbr, pbf=pbf, pbr=pbr,
                                   trunc_rho=float(trunc_rho), label_std=float(y.std()))
        print(f"\n[{task}]  full population n={len(merged)}  (reliable averaged label, "
              f"std={y.std():.3f})   [{elapsed:.0f}s]")
        print(f"   geometry      RF 5-fold CV rho = {geo_rho:+.3f}   best single: {gbf} {gbr:+.3f}")
        print(f"   cheap probe   RF 5-fold CV rho = {probe_rho:+.3f}   best single: {pbf} {pbr:+.3f}")
        print(f"   trunc-search  rho_s(D_probe, D_HC) = {trunc_rho:+.3f}")
        print(f"   -> wrote {out}")

    # ----- (ii) LavaGap truncated search vs shaped label -------------------- #
    print("\n" + "-" * 78)
    lava_path = os.path.join(ROOT, "direction_c_AB.csv")
    lava_rho = None
    if os.path.exists(lava_path):
        lv = pd.read_csv(lava_path)
        if args.quick:
            lv = lv.head(12)
        dvals = []
        for _, row in lv.iterrows():
            g = int(row["grid_seed"])
            d, _ = truncated_search_difficulty_lavagap(g, t_probe=args.t_probe, search_seed=g)
            dvals.append(d)
        lv = lv.assign(trunc_search_difficulty=dvals)
        lava_rho, _ = spearmanr(lv["trunc_search_difficulty"].values,
                                lv["diff_A_partial"].values)
        lv.to_csv(os.path.join(ROOT, "lavagap_trunc_search.csv"), index=False)
        print(f"[LavaGap shaped]  n={len(lv)}   "
              f"trunc-search rho_s(D_probe, D_HC) = {lava_rho:+.3f}")
    else:
        print("[LavaGap] direction_c_AB.csv not found -- skipping")

    # ----- (iii) wall-clock cost ------------------------------------------- #
    print("\n" + "-" * 78)
    print("WALL-CLOCK COST (per grid, mean over sample)")
    n_time = 3 if args.quick else 8
    fixed_t, trunc_t, full_t = [], [], []
    for seed in range(n_time):
        t = time.time(); extract_rollout_features("TopOff", seed, n_random=args.n_random); fixed_t.append(time.time() - t)
        t = time.time(); truncated_search_difficulty_karel("TopOff", seed, t_probe=args.t_probe, search_seed=seed); trunc_t.append(time.time() - t)
        t = time.time(); full_hc_karel("TopOff", seed, search_seed=seed, max_programs=5000); full_t.append(time.time() - t)
    fm, tm, hm = np.mean(fixed_t), np.mean(trunc_t), np.mean(full_t)
    print(f"   fixed-library probe : {fm*1000:8.1f} ms/grid")
    print(f"   truncated search    : {tm*1000:8.1f} ms/grid")
    print(f"   full reference HC   : {hm*1000:8.1f} ms/grid")
    print(f"   speedup fixed vs HC : {hm/fm:6.0f}x")
    print(f"   speedup trunc vs HC : {hm/tm:6.0f}x")

    # ----- summary --------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for task, s in karel_summary.items():
        print(f"{task:8s} n={s['n']:3d}  geom CV {s['geo_rho']:+.3f} | probe CV {s['probe_rho']:+.3f} | "
              f"probe-best {s['pbf']} {s['pbr']:+.3f} | trunc {s['trunc_rho']:+.3f}")
    if lava_rho is not None:
        print(f"LavaGap  trunc rho_s(D_probe, D_HC) = {lava_rho:+.3f}")
    print(f"\nwall-clock: fixed {fm*1000:.0f}ms  trunc {tm*1000:.0f}ms  fullHC {hm*1000:.0f}ms  "
          f"(HC/fixed={hm/fm:.0f}x, HC/trunc={hm/tm:.0f}x)")


if __name__ == "__main__":
    main()
