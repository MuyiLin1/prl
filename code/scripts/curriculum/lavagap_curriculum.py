#!/usr/bin/env python3
"""Section 6.6 curriculum prefilter experiment (Option A: program-search learner).

Goal (paper Section 3.5, empirical claims 2 and 3): show that a cheap difficulty
predictor can PREFILTER a large pool of procedurally generated tasks down to a
small, well-chosen subset on which "training" matches full-pool generalization at
a fraction of the cost, and that this curated subset beats a random subset of the
same size (the ablation that proves the *predictor* is doing the work).

NO LLM anywhere. Domain: shaped LavaGap (the geometric pole where the difficulty
predictor actually works, rho ~ 0.7).

What "train an agent on a set of levels" means here (Option A):
  The agent is a *program*. Training on a set S = searching (Hill Climbing) for the
  single program that maximizes MEAN shaped reward across all levels in S. This is
  faithful to how the paper defines difficulty (the same DSL+HC solver), needs no
  GPU, and reuses trusted code. HillClimbing.evaluate_program already averages
  reward over a list of task envs, so passing S as task_envs IS multi-level training.

Pipeline:
  1. Pool: N LavaGap layouts (env seeds). Objective fixed, gap position varies.
  2. Cheap scoring (the prefilter, no full solve):
       - geometry descriptor  g(theta)  via lavagap_features   (instant, no rollout)
       - mini-HC difficulty    D_probe   via truncated search   (~10 evals, Eq. mini-hc)
  3. Curate k << N: keep a target-difficulty band (zone of proximal development),
     then pick k for descriptor-space COVERAGE (greedy farthest-point / k-center).
  4. Three training sets: full pool (N), curated-k, random-k (several draws).
  5. Train (Option A HC) on each -> one program; record training cost (env-evals).
  6. Evaluate every trained program on a held-out set of M fresh layouts:
       mean shaped reward + solve rate (reward == 1.0 -> reached goal).
  7. Report full vs curated-k vs random-k generalization and cost.

Target result: curated-k ~ full pool on held-out reward at a fraction of the cost,
and curated-k > random-k.
"""
from __future__ import annotations

import sys
import os
import argparse
import json
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))      # scripts/curriculum
_SCRIPTS = os.path.dirname(_HERE)                        # scripts
_ROOT = os.path.dirname(_SCRIPTS)                        # repo root
sys.path.insert(0, _ROOT)
sys.path.append(os.path.join(_ROOT, 'leaps'))
sys.path.insert(0, _SCRIPTS)  # reuse existing helper scripts

# Heavy imports are deferred to _lazy_imports() so that --merge works without
# triggering prog_policies / sklearn / minigrid chains.
_imports_done = False

def _lazy_imports():
    global _imports_done, MinigridDSL, MinigridProgrammaticSpace, HillClimbing
    global make_shaped_lavagap, lavagap_features, truncated_search_difficulty_lavagap
    if _imports_done:
        return
    from prog_policies.utils import get_env_name  # noqa: F401 -- circular-import fix
    from prog_policies.minigrid.dsl import MinigridDSL as _DSL
    from prog_policies.search_space import MinigridProgrammaticSpace as _Space
    from prog_policies.search_methods import HillClimbing as _HC
    from minigrid_shaped import make_shaped_lavagap as _make
    from option_a_lavagap import lavagap_features as _feats
    MinigridDSL = _DSL
    MinigridProgrammaticSpace = _Space
    HillClimbing = _HC
    make_shaped_lavagap = _make
    lavagap_features = _feats
    _imports_done = True


def truncated_search_difficulty_lavagap(env_seed, t_probe=10, search_seed=0, size=9):
    """Inlined from probe_full_eval: mini-HC probe difficulty for shaped LavaGap."""
    _lazy_imports()
    dsl = MinigridDSL()
    task_envs = [make_shaped_lavagap(env_seed, size=size)]
    space = MinigridProgrammaticSpace(dsl, sigma=0.25)
    space.set_seed(search_seed + 7919)
    hc = HillClimbing(k=t_probe, e=2)
    base_ind, base_prog = space.initialize_individual()
    best = hc.evaluate_program(base_prog, task_envs)
    for _, prog in space.get_neighbors(base_ind, k=t_probe):
        r = hc.evaluate_program(prog, task_envs)
        if r > best:
            best = r
    return 1.0 - best, t_probe + 1


# --------------------------------------------------------------------------- #
# Step 2: cheap per-level scoring (the prefilter signals)
# --------------------------------------------------------------------------- #
# Descriptor columns that actually vary on LavaGap at the sizes we use.
_DESCRIPTOR_COLS = ["gap_x", "gap_y", "gap_col", "gap_row_offset", "path_len", "detour"]


def score_pool(pool_seeds, size, probe_seeds, verbose=True):
    """Compute geometry descriptor + mini-HC difficulty probe for each pool seed."""
    _lazy_imports()
    rows = []
    for i, s in enumerate(pool_seeds):
        feats = lavagap_features(s, size)
        # mini-HC difficulty probe, averaged over a few probe seeds for stability
        d_probe = np.mean([
            truncated_search_difficulty_lavagap(s, t_probe=10, search_seed=1000 * (j + 1) + s, size=size)[0]
            for j in range(probe_seeds)
        ])
        row = {"env_seed": int(s), "d_probe": float(d_probe)}
        row.update(feats)
        rows.append(row)
        if verbose and (i + 1) % 20 == 0:
            print(f"    scored {i + 1}/{len(pool_seeds)} pool levels", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Step 3: curation = difficulty band + descriptor-space coverage (k-center)
# --------------------------------------------------------------------------- #
def _standardize(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    return (X - mu) / sd


def curate_subset(df, k, band_lo=0.15, band_hi=0.85, rng=None,
                  subsample=1.0, random_start=False):
    """Predictor-curated subset of size k.

    (a) Zone of proximal development: keep only levels whose mini-HC difficulty
        D_probe lies in the [band_lo, band_hi] quantile band (drop trivially easy
        and hopeless levels).
    (b) Coverage: among the band, greedily pick k levels that are maximally spread
        in standardized descriptor space (farthest-point / k-center sampling).

    Controlled randomness for a *fair* ablation (Section 6.6): to obtain an
    ensemble of curated subsets (matching the random condition's n), each draw
    optionally (i) subsamples the pool to a fraction (``subsample`` < 1, without
    replacement) before curating, and (ii) randomizes the greedy start point
    (``random_start``). With subsample==1 and random_start==False this reduces to
    the deterministic median-seeded curation.
    """
    rng = rng or np.random.default_rng(0)
    if subsample < 1.0:
        m = max(k + 1, int(round(subsample * len(df))))
        m = min(m, len(df))
        idx = rng.choice(len(df), size=m, replace=False)
        df = df.iloc[idx].reset_index(drop=True)
    lo, hi = df["d_probe"].quantile([band_lo, band_hi])
    band = df[(df["d_probe"] >= lo) & (df["d_probe"] <= hi)].reset_index(drop=True)
    if len(band) <= k:
        # band too small: relax to full pool but still farthest-point select
        band = df.reset_index(drop=True)
    cols = [c for c in _DESCRIPTOR_COLS if band[c].std() > 1e-9]
    X = _standardize(band[cols].to_numpy(dtype=float))

    # seed greedy: random start (ensemble) or median-difficulty level (deterministic)
    if random_start:
        start = int(rng.integers(len(band)))
    else:
        med = band["d_probe"].median()
        start = int((band["d_probe"] - med).abs().to_numpy().argmin())
    chosen = [start]
    min_d = np.linalg.norm(X - X[start], axis=1)
    while len(chosen) < k:
        nxt = int(min_d.argmax())
        if nxt in chosen:
            # all remaining are duplicates; fill randomly
            remaining = [j for j in range(len(band)) if j not in chosen]
            chosen.append(int(rng.choice(remaining)))
        else:
            chosen.append(nxt)
        d_new = np.linalg.norm(X - X[chosen[-1]], axis=1)
        min_d = np.minimum(min_d, d_new)
    return [int(band.loc[j, "env_seed"]) for j in chosen]


def random_subset(df, k, rng):
    return [int(x) for x in rng.choice(df["env_seed"].to_numpy(), size=k, replace=False)]


# --------------------------------------------------------------------------- #
# Step 5: train (Option A) = HC for one program over the level set
# --------------------------------------------------------------------------- #
def train_on_set(train_seeds, size, search_seed, n_iter, k_neighbors, max_calls):
    _lazy_imports()
    """Hill Climbing for a single program maximizing mean shaped reward over the set.

    Returns (program, train_mean_reward, n_program_evals, n_env_evals).
    """
    dsl = MinigridDSL()
    task_envs = [make_shaped_lavagap(s, size=size, max_calls=max_calls) for s in train_seeds]
    space = MinigridProgrammaticSpace(dsl, sigma=0.25)
    space.set_seed(search_seed)
    hc = HillClimbing(k=k_neighbors, e=2)
    progs, rewards = hc.search(space, task_envs, seed=search_seed, n_iterations=n_iter)
    best_prog = progs[-1]
    # cost: each evaluate_program call touches len(task_envs) envs. We count the
    # program evals HC actually consumed via the recorded reward trajectory length
    # is a lower bound; instead instrument program_num on the first env.
    n_prog_evals = task_envs[0].program_num
    n_env_evals = n_prog_evals * len(task_envs)
    return best_prog, float(rewards[-1]), int(n_prog_evals), int(n_env_evals)


# --------------------------------------------------------------------------- #
# Step 6: held-out generalization
# --------------------------------------------------------------------------- #
def eval_program(program, eval_seeds, size, max_calls):
    """Mean shaped reward + solve rate of one program on held-out layouts."""
    rewards = []
    for s in eval_seeds:
        env = make_shaped_lavagap(s, size=size, max_calls=max_calls)
        rewards.append(float(env.evaluate_program(program)))
    rewards = np.array(rewards)
    return float(rewards.mean()), float((rewards >= 0.999).mean())


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_subsets(args):
    """Deterministically (given args.seed) build the pool scores and the
    full/curated/random subsets. Every shard calls this and agrees on the result."""
    rng = np.random.default_rng(args.seed)
    pool_seeds = list(range(args.pool_size))

    print(f"\n[1] Scoring pool with geometry + mini-HC probe ...", flush=True)
    t0 = time.time()
    df = score_pool(pool_seeds, args.size, args.probe_seeds, verbose=True)
    print(f"    done in {time.time() - t0:.1f}s  "
          f"(D_probe range {df.d_probe.min():.2f}-{df.d_probe.max():.2f})", flush=True)

    # Curated ENSEMBLE: multiple curated subsets (controlled randomness) so the
    # curated condition has the same n as random, making the ablation fair.
    # The first draw is the canonical deterministic curation; subsequent draws
    # subsample the pool + randomize the greedy start to vary subset selection.
    curateds = []
    for d in range(args.curated_draws):
        crng = np.random.default_rng(args.seed * 7919 + 131 * d + 1)
        if d == 0 and args.curated_draws == 1:
            sub, rstart = 1.0, False  # backward-compatible single deterministic set
        else:
            sub, rstart = args.pool_subsample, True
        curateds.append(curate_subset(df, args.k, args.band_lo, args.band_hi,
                                      crng, subsample=sub, random_start=rstart))
    randoms = [random_subset(df, args.k, rng) for _ in range(args.random_draws)]
    print(f"[2] Curated ensemble ({len(curateds)} subsets, k={args.k}): difficulty "
          f"band [{args.band_lo},{args.band_hi}] + k-center coverage", flush=True)
    return df, pool_seeds, curateds, randoms


def build_jobs(args, pool_seeds, curateds, randoms):
    """Deterministic ordered job list: (condition, draw_idx, subset, search_seed_idx)."""
    jobs = []
    for si in range(args.train_seeds):
        jobs.append(("full", 0, pool_seeds, si))
    for di, subset in enumerate(curateds):
        for si in range(args.train_seeds):
            jobs.append(("curated", di, subset, si))
    for di, subset in enumerate(randoms):
        for si in range(args.train_seeds):
            jobs.append(("random", di, subset, si))
    return jobs


def summarize(res):
    summ = res.groupby("condition").agg(
        heldout_reward_mean=("heldout_reward", "mean"),
        heldout_reward_std=("heldout_reward", "std"),
        heldout_solve_mean=("heldout_solve", "mean"),
        train_size=("train_size", "max"),
        env_evals_mean=("n_env_evals", "mean"),
    ).reindex(["full", "curated", "random"])
    return summ


def run(args):
    if args.merge:
        import glob
        parts = sorted(glob.glob(f"{args.out}_shard*of*.csv"))
        if not parts:
            print(f"no shard files matching {args.out}_shard*of*.csv", flush=True)
            return
        res = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        res.to_csv(f"{args.out}_results.csv", index=False)
        summ = summarize(res)
        print(f"\n== MERGED SUMMARY ({len(parts)} shards, {len(res)} runs) ==", flush=True)
        print(summ.to_string(), flush=True)
        summ.to_csv(f"{args.out}_summary.csv")
        return

    print(f"== LavaGap curriculum prefilter (Option A) ==", flush=True)
    print(f"pool N={args.pool_size}  curated k={args.k}  held-out M={args.heldout}  "
          f"size={args.size}  shard {args.shard}/{args.num_shards}", flush=True)

    heldout_seeds = list(range(100_000, 100_000 + args.heldout))
    df, pool_seeds, curateds, randoms = build_subsets(args)
    if args.shard == 0:
        df.to_csv(f"{args.out}_pool_scores.csv", index=False)
        with open(f"{args.out}_meta.json", "w") as f:
            json.dump({"args": vars(args), "curated_seeds": curateds,
                       "random_seeds": randoms}, f, indent=2)

    jobs = build_jobs(args, pool_seeds, curateds, randoms)
    my_jobs = [(i, j) for i, j in enumerate(jobs) if i % args.num_shards == args.shard]
    print(f"\n[3] Running {len(my_jobs)}/{len(jobs)} training jobs for this shard\n", flush=True)

    results = []
    for idx, (cond, di, subset, si) in my_jobs:
        search_seed = 7000 + 101 * si + 13 * di
        t0 = time.time()
        prog, train_r, n_prog, n_env = train_on_set(
            subset, args.size, search_seed, args.n_iter,
            args.k_neighbors, args.max_calls)
        ho_r, ho_solve = eval_program(prog, heldout_seeds, args.size, args.max_calls)
        dt = time.time() - t0
        results.append({
            "job": idx, "condition": cond, "draw": di, "search_seed": search_seed,
            "train_size": len(subset), "train_reward": train_r,
            "heldout_reward": ho_r, "heldout_solve": ho_solve,
            "n_prog_evals": n_prog, "n_env_evals": n_env, "wall_s": dt,
        })
        print(f"  job{idx:>3} {cond:>8} draw{di} seed{si}: train_r={train_r:.3f} "
              f"heldout_r={ho_r:.3f} solve={ho_solve:.2f} "
              f"env_evals={n_env} ({dt:.1f}s)", flush=True)

    res = pd.DataFrame(results)
    out_csv = (f"{args.out}_results.csv" if args.num_shards == 1
               else f"{args.out}_shard{args.shard}of{args.num_shards}.csv")
    res.to_csv(out_csv, index=False)

    if args.num_shards == 1:
        print(f"\n== SUMMARY (mean +/- std over runs) ==", flush=True)
        print(summarize(res).to_string(), flush=True)
        summarize(res).to_csv(f"{args.out}_summary.csv")
    print(f"\nwrote {out_csv}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-size", type=int, default=120)
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--heldout", type=int, default=60)
    ap.add_argument("--size", type=int, default=9)
    ap.add_argument("--probe-seeds", type=int, default=3)
    ap.add_argument("--n-iter", type=int, default=150)
    ap.add_argument("--k-neighbors", type=int, default=64)
    ap.add_argument("--train-seeds", type=int, default=3)
    ap.add_argument("--random-draws", type=int, default=3)
    ap.add_argument("--curated-draws", type=int, default=1,
                    help="number of curated subsets (ensemble) for a fair ablation")
    ap.add_argument("--pool-subsample", type=float, default=0.85,
                    help="pool fraction kept per curated draw (>1st) for selection variance")
    ap.add_argument("--band-lo", type=float, default=0.15)
    ap.add_argument("--band-hi", type=float, default=0.85)
    ap.add_argument("--max-calls", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="lavagap_curriculum")
    ap.add_argument("--shard", type=int, default=0, help="this shard index (0-based)")
    ap.add_argument("--num-shards", type=int, default=1, help="total shards")
    ap.add_argument("--merge", action="store_true", help="merge shard CSVs and summarize")
    ap.add_argument("--quick", action="store_true", help="tiny smoke config")
    args = ap.parse_args()

    if args.quick:
        args.pool_size = 24
        args.k = 8
        args.heldout = 12
        args.probe_seeds = 1
        args.n_iter = 30
        args.k_neighbors = 16
        args.train_seeds = 1
        args.random_draws = 1
        args.curated_draws = 1

    run(args)


if __name__ == "__main__":
    main()
