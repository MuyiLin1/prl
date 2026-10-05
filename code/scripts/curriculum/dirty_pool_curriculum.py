#!/usr/bin/env python3
"""Section 6.6 follow-up: does predictor-curation beat random on a *contaminated* pool?

Motivation
----------
The clean-pool experiment (``lavagap_curriculum.py``) found curation == random:
when the training pool is an IID sample of the deployment distribution, a random
k-subset is already representative, so a cheap difficulty predictor has nothing to
correct. That is the honest null. This script tests the complementary, realistic
regime: you are handed a *dirty* unlabeled pool (mostly trivial junk, a hard/degenerate
tail, and duplicate clusters) but you want to perform well on the *clean* deployment
distribution. We ask whether "looking at the pool before training" (cheap-probe
curation) recovers deployment performance better than blind random subsampling.

Key design (pre-registered; see /memories/session/dirty_pool_preregistration.md)
--------------------------------------------------------------------------------
* Pool (N levels): CONTAMINATED on purpose, from real generator output:
    - trivial-heavy   : oversample easy gap positions (low difficulty)
    - hard/degenerate : include the natural hard tail (probe stuck near 1.0)
    - duplicate cluster: a few informative gap positions repeated many times
  No synthetic levels: every level is real generator output; we only choose an
  unflattering, realistic *mixture*.
* Held-out TEST: the NATURAL generator distribution (seeds 100000+, UNFILTERED).
  This is the deployment target and is defined with ZERO reference to the probe or
  the curation rule, so the comparison is not circular.
* 4-way ablation (matched n), each vs the full dirty pool:
    - random      : k levels drawn uniformly from the dirty pool
    - band-only   : keep the medium-difficulty band, then random k within it
    - coverage-only: farthest-point (k-center) over the whole dirty pool
    - both         : band + coverage (the full curation rule)
  Decomposing band vs coverage tells us WHICH knob (junk-dropping vs de-duplication)
  pays off under WHICH contamination.

NO LLM anywhere. Reuses the trusted DSL+HC machinery from lavagap_curriculum.py.
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
sys.path.insert(0, _SCRIPTS)
sys.path.insert(0, _HERE)

# Reuse all trusted helpers from the clean-pool experiment.
import lavagap_curriculum as lc
from lavagap_curriculum import (
    _lazy_imports, _DESCRIPTOR_COLS, _standardize,
    truncated_search_difficulty_lavagap,
    curate_subset, random_subset, train_on_set, eval_program,
)


# --------------------------------------------------------------------------- #
# Step 1: characterize a large seed bank by gap position (cheap) + d_probe
# --------------------------------------------------------------------------- #
def score_seed_bank(gen_size, size, probe_seeds, verbose=True):
    """For seeds [0, gen_size): geometry descriptor (instant) + cached per-position
    mini-HC difficulty probe. d_probe is computed ONCE per distinct gap position and
    broadcast to every seed sharing it (positions repeat heavily on LavaGap)."""
    _lazy_imports()
    from option_a_lavagap import lavagap_features

    feats = []
    for s in range(gen_size):
        f = lavagap_features(s, size)
        f["env_seed"] = int(s)
        feats.append(f)
    df = pd.DataFrame(feats)
    df["pos"] = list(zip(df["gap_x"].astype(int), df["gap_y"].astype(int)))

    # cache one d_probe per distinct gap position
    positions = sorted(df["pos"].unique())
    if verbose:
        print(f"    seed bank {gen_size} seeds -> {len(positions)} distinct gap positions; "
              f"probing each ...", flush=True)
    pos_d = {}
    t0 = time.time()
    for i, p in enumerate(positions):
        s = int(df[df["pos"] == p]["env_seed"].iloc[0])
        d = np.mean([
            truncated_search_difficulty_lavagap(
                s, t_probe=10, search_seed=1000 * (j + 1) + s, size=size)[0]
            for j in range(probe_seeds)
        ])
        pos_d[p] = float(d)
        if verbose and (i + 1) % 25 == 0:
            print(f"      probed {i + 1}/{len(positions)} positions "
                  f"({time.time() - t0:.0f}s)", flush=True)
    df["d_probe"] = df["pos"].map(pos_d)
    return df, pos_d


# --------------------------------------------------------------------------- #
# Step 2: compose the contaminated pool (real levels, unflattering mixture)
# --------------------------------------------------------------------------- #
def compose_dirty_pool(bank, args, rng):
    """Build an N-seed contaminated pool from the seed bank `bank`.

    Tiers by the position's d_probe (terciles by default), then samples seeds
    trivial-heavy, with a hard/degenerate tail and a few forced duplicate clusters.
    Returns a DataFrame with one row per chosen (distinct) seed; gap positions repeat
    across rows by construction (that IS the duplication pathology).
    """
    lo_q, hi_q = bank["d_probe"].quantile([args.trivial_q, args.hard_q])
    trivial = bank[bank["d_probe"] <= lo_q]
    hard = bank[bank["d_probe"] >= hi_q]
    informative = bank[(bank["d_probe"] > lo_q) & (bank["d_probe"] < hi_q)]

    n_triv = int(round(args.frac_trivial * args.pool_size))
    n_hard = int(round(args.frac_hard * args.pool_size))
    n_dup = int(round(args.frac_dup * args.pool_size))
    n_info = args.pool_size - n_triv - n_hard - n_dup

    def draw(pool_df, n):
        if n <= 0 or len(pool_df) == 0:
            return []
        replace = n > len(pool_df)
        idx = rng.choice(pool_df.index.to_numpy(), size=n, replace=replace)
        return list(idx)

    chosen_idx = []
    chosen_idx += draw(trivial, n_triv)
    chosen_idx += draw(hard, n_hard)
    chosen_idx += draw(informative, n_info)

    # forced duplicate clusters: pick a few informative positions, repeat them
    if n_dup > 0 and len(informative) > 0:
        info_positions = sorted(informative["pos"].unique())
        n_clusters = min(args.dup_clusters, len(info_positions))
        cluster_pos = [info_positions[i] for i in
                       rng.choice(len(info_positions), size=n_clusters, replace=False)]
        per = max(1, n_dup // n_clusters)
        for p in cluster_pos:
            rows = informative[informative["pos"] == p]
            chosen_idx += draw(rows, per)

    pool = bank.loc[chosen_idx].reset_index(drop=True)
    # give every pool entry a unique row key so subset selection picks ROWS, not
    # collapsing repeated positions; env_seed may repeat if a tier was small.
    pool["pool_id"] = np.arange(len(pool))
    return pool, dict(trivial=len(trivial), hard=len(hard), informative=len(informative),
                      n_triv=n_triv, n_hard=n_hard, n_info=n_info, n_dup=n_dup,
                      lo_q=float(lo_q), hi_q=float(hi_q))


# --------------------------------------------------------------------------- #
# Step 3: the four selection rules (band-only / coverage-only / both / random)
# --------------------------------------------------------------------------- #
def band_only_subset(df, k, band_lo, band_hi, rng):
    """Keep the medium-difficulty band, then pick k uniformly at random within it.
    Isolates the junk-dropping knob (no coverage)."""
    lo, hi = df["d_probe"].quantile([band_lo, band_hi])
    band = df[(df["d_probe"] >= lo) & (df["d_probe"] <= hi)]
    if len(band) <= k:
        band = df
    idx = rng.choice(band.index.to_numpy(), size=k, replace=k > len(band))
    return [int(df.loc[i, "env_seed"]) for i in idx]


def coverage_only_subset(df, k, rng, random_start=True):
    """Farthest-point (k-center) coverage over the WHOLE pool, no difficulty band.
    Isolates the de-duplication / spread knob."""
    return curate_subset(df, k, band_lo=0.0, band_hi=1.0, rng=rng,
                         subsample=1.0, random_start=random_start)


def both_subset(df, k, band_lo, band_hi, rng, random_start=True):
    """Full curation: difficulty band + descriptor-space coverage."""
    return curate_subset(df, k, band_lo=band_lo, band_hi=band_hi, rng=rng,
                         subsample=1.0, random_start=random_start)


def dirty_random_subset(df, k, rng):
    """Uniform random k from the dirty pool, picking ROWS (so duplicate positions can
    be drawn more than once in expectation)."""
    idx = rng.choice(df.index.to_numpy(), size=k, replace=k > len(df))
    return [int(df.loc[i, "env_seed"]) for i in idx]


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
_CONDITIONS = ["random", "band", "coverage", "both"]


def build_subsets(args):
    rng = np.random.default_rng(args.seed)

    print(f"\n[1] Scoring seed bank ({args.gen_size} seeds) ...", flush=True)
    t0 = time.time()
    bank, _pos_d = score_seed_bank(args.gen_size, args.size, args.probe_seeds, verbose=True)
    print(f"    done in {time.time() - t0:.1f}s  "
          f"(d_probe {bank.d_probe.min():.2f}-{bank.d_probe.max():.2f})", flush=True)

    print(f"[2] Composing contaminated pool (N={args.pool_size}) ...", flush=True)
    pool, info = compose_dirty_pool(bank, args, rng)
    print(f"    tiers: trivial<= {info['lo_q']:.2f} ({info['n_triv']}), "
          f"informative ({info['n_info']}), hard>= {info['hi_q']:.2f} ({info['n_hard']}), "
          f"dup-cluster ({info['n_dup']})", flush=True)
    print(f"    pool distinct gap positions: {pool['pos'].nunique()} / {len(pool)} rows", flush=True)

    # ensembles (matched n across all conditions) via controlled randomness
    subsets = {c: [] for c in _CONDITIONS}
    for d in range(args.draws):
        drng = np.random.default_rng(args.seed * 7919 + 131 * d + 1)
        subsets["random"].append(dirty_random_subset(pool, args.k, drng))
        subsets["band"].append(band_only_subset(pool, args.k, args.band_lo, args.band_hi, drng))
        subsets["coverage"].append(coverage_only_subset(pool, args.k, drng, random_start=(d > 0)))
        subsets["both"].append(both_subset(pool, args.k, args.band_lo, args.band_hi, drng,
                                            random_start=(d > 0)))
    return bank, pool, subsets


def build_jobs(args, pool, subsets):
    """Ordered job list: full pool baseline, then each condition x draw x train-seed."""
    jobs = []
    full_seeds = [int(s) for s in pool["env_seed"].to_numpy()]
    full_train_seeds = args.full_train_seeds if args.full_train_seeds > 0 else args.train_seeds
    if args.full_train_seeds < 0:
        full_train_seeds = 0  # skip the expensive full-pool baseline entirely
    for si in range(full_train_seeds):
        jobs.append(("full", 0, full_seeds, si))
    for cond in _CONDITIONS:
        for di, subset in enumerate(subsets[cond]):
            for si in range(args.train_seeds):
                jobs.append((cond, di, subset, si))
    return jobs


def summarize(res):
    order = ["full"] + _CONDITIONS
    summ = res.groupby("condition").agg(
        heldout_reward_mean=("heldout_reward", "mean"),
        heldout_reward_std=("heldout_reward", "std"),
        heldout_solve_mean=("heldout_solve", "mean"),
        train_size=("train_size", "max"),
        env_evals_mean=("n_env_evals", "mean"),
    ).reindex([c for c in order if c in res["condition"].unique()])
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

    print(f"== Dirty-pool curriculum (Option A) ==", flush=True)
    print(f"gen={args.gen_size}  pool N={args.pool_size}  k={args.k}  held-out M={args.heldout}  "
          f"size={args.size}  shard {args.shard}/{args.num_shards}", flush=True)

    # CLEAN test = natural generator output, disjoint seed range, UNFILTERED.
    heldout_seeds = list(range(100_000, 100_000 + args.heldout))

    bank, pool, subsets = build_subsets(args)
    if args.shard == 0:
        pool.drop(columns=["pos"]).to_csv(f"{args.out}_pool.csv", index=False)
        with open(f"{args.out}_meta.json", "w") as f:
            json.dump({"args": vars(args),
                       "subsets": {c: [list(map(int, s)) for s in subsets[c]]
                                   for c in _CONDITIONS}}, f, indent=2)

    jobs = build_jobs(args, pool, subsets)
    my_jobs = [(i, j) for i, j in enumerate(jobs) if i % args.num_shards == args.shard]
    print(f"\n[3] Running {len(my_jobs)}/{len(jobs)} training jobs for this shard\n", flush=True)

    results = []
    for idx, (cond, di, subset, si) in my_jobs:
        search_seed = 7000 + 101 * si + 13 * di
        t0 = time.time()
        prog, train_r, n_prog, n_env = train_on_set(
            subset, args.size, search_seed, args.n_iter, args.k_neighbors, args.max_calls)
        ho_r, ho_solve = eval_program(prog, heldout_seeds, args.size, args.max_calls)
        dt = time.time() - t0
        results.append({
            "job": idx, "condition": cond, "draw": di, "search_seed": search_seed,
            "train_size": len(subset), "train_reward": train_r,
            "heldout_reward": ho_r, "heldout_solve": ho_solve,
            "n_prog_evals": n_prog, "n_env_evals": n_env, "wall_s": dt,
        })
        print(f"  job{idx:>3} {cond:>9} draw{di} seed{si}: train_r={train_r:.3f} "
              f"heldout_r={ho_r:.3f} solve={ho_solve:.2f} env_evals={n_env} ({dt:.1f}s)",
              flush=True)

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
    ap.add_argument("--gen-size", type=int, default=2000,
                    help="seeds [0,gen_size) over-generated to characterize positions")
    ap.add_argument("--pool-size", type=int, default=1000, help="N contaminated pool levels")
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--heldout", type=int, default=80)
    ap.add_argument("--size", type=int, default=13)
    ap.add_argument("--probe-seeds", type=int, default=2)
    ap.add_argument("--n-iter", type=int, default=250)
    ap.add_argument("--k-neighbors", type=int, default=128)
    ap.add_argument("--train-seeds", type=int, default=6)
    ap.add_argument("--full-train-seeds", type=int, default=0,
                    help="train-seeds for the (expensive) full-pool baseline; "
                         "0 = use --train-seeds, <0 = skip full entirely. "
                         "Set lower to save cluster time.")
    ap.add_argument("--draws", type=int, default=6,
                    help="ensemble subsets per condition (matched n for a fair ablation)")
    ap.add_argument("--band-lo", type=float, default=0.15)
    ap.add_argument("--band-hi", type=float, default=0.85)
    # contamination recipe
    ap.add_argument("--frac-trivial", type=float, default=0.55)
    ap.add_argument("--frac-hard", type=float, default=0.15)
    ap.add_argument("--frac-dup", type=float, default=0.15)
    ap.add_argument("--dup-clusters", type=int, default=3)
    ap.add_argument("--trivial-q", type=float, default=0.33, help="d_probe quantile for trivial tier")
    ap.add_argument("--hard-q", type=float, default=0.67, help="d_probe quantile for hard tier")
    ap.add_argument("--max-calls", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="dirty_pool_curriculum")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny smoke config")
    args = ap.parse_args()

    if args.quick:
        args.gen_size = 200
        args.pool_size = 80
        args.k = 12
        args.heldout = 16
        args.probe_seeds = 1
        args.n_iter = 30
        args.k_neighbors = 16
        args.train_seeds = 1
        args.draws = 2

    run(args)


if __name__ == "__main__":
    main()
