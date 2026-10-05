#!/usr/bin/env python3
"""Path 1: does predictor-curation beat random on a *contaminated* Sokoban pool?

Motivation
----------
The LavaGap dirty-pool experiment (``dirty_pool_curriculum.py``) found curation ==
random even under heavy contamination. That null is *structural*: LavaGap's effective
descriptor space collapses to a single gap position, so a random k-subset already
covers it, and the mean-reward HC learner is junk-robust. To test whether difficulty-
aware curation has any traction on a domain that does NOT have that pathology, we move
to Sokoban (Boxoban), which has (i) real difficulty heterogeneity (D in [0, 0.64]),
(ii) a rich, varying descriptor space (box->target matching lower bound, dead cells,
box-target distances, grid-graph geometry), and (iii) externally-valid difficulty tiers
(unfiltered < medium < hard).

Design (mirrors dirty_pool_curriculum.py)
-----------------------------------------
* Level bank: real Boxoban levels sampled across tiers. Each level carries its ASCII
  grid, its tier, cheap static descriptors (``sokoban_core.extract_features``), and a
  cheap mini-HC difficulty probe ``d_probe`` (truncated random-restart HC, a few seeds).
* Contaminated POOL (N levels): trivial-heavy (low d_probe) + a hard/degenerate tail +
  forced duplicate clusters (a few near-identical levels repeated). Every level is real
  Boxoban output; only the *mixture* is unflattering.
* CLEAN held-out TEST: a disjoint, natural tier-mix sample of Boxoban levels, defined
  with ZERO reference to the probe or curation rule (breaks circularity). This is the
  deployment target.
* 4-way ablation (matched n), each vs the full dirty pool:
    - random       : k levels drawn uniformly from the dirty pool
    - band-only    : keep the medium-difficulty d_probe band, then random k within it
    - coverage-only : farthest-point (k-center) over the whole dirty pool descriptors
    - both         : band + coverage (the full curation rule)
* Learner (Option A, identical in spirit to LavaGap / the difficulty labeler): "training
  on a set S" = Hill Climbing for the single SokobanDSL program that maximizes MEAN
  shaped reward across all levels in S. Generalization = freeze the program, evaluate
  mean shaped reward + solve rate on the clean held-out levels.

NO LLM anywhere. Reuses the trusted DSL+HC machinery and the Sokoban feature extractor.
"""
from __future__ import annotations

import sys
import os
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))      # scripts/curriculum
_SCRIPTS = os.path.dirname(_HERE)                        # scripts
_ROOT = os.path.dirname(_SCRIPTS)                        # repo root
sys.path.insert(0, _ROOT)
sys.path.append(os.path.join(_ROOT, 'leaps'))
sys.path.insert(0, _SCRIPTS)
sys.path.insert(0, os.path.join(_SCRIPTS, 'sokoban'))

_DATA = Path(_ROOT) / "data" / "boxoban"
TIER_DIRS = {
    "unfiltered": _DATA / "unfiltered" / "train",
    "medium": _DATA / "medium" / "train",
    "hard": _DATA / "hard",
}

# Sokoban descriptor columns used for coverage (k-center). Constant columns are
# dropped automatically by the std>1e-9 filter, so a generous list is safe.
_SOKOBAN_DESCRIPTOR_COLS = [
    "box_target_matching_lb", "box_target_dist_sum", "box_target_dist_mean",
    "box_target_dist_max", "num_dead_cells", "graph_diameter", "mean_pair_distance",
    "num_corridor_cells", "num_deadends", "num_branch_cells", "player_eccentricity",
    "player_to_nearest_box", "wall_fraction", "num_free_cells", "num_bridges",
    "num_articulation_points",
]

# Heavy imports deferred so --merge works without the prog_policies / scipy chains.
_imports_done = False


def _lazy_imports():
    global _imports_done, SokobanDSL, BoxobanTask, ProgrammaticSpace, HillClimbing
    global extract_features, iter_levels_in_file
    if _imports_done:
        return
    import prog_policies.utils  # noqa: F401 -- establishes safe init order
    from prog_policies.sokoban.dsl import SokobanDSL as _DSL
    from prog_policies.sokoban_tasks.boxoban import BoxobanTask as _Task
    from prog_policies.search_space import ProgrammaticSpace as _Space
    from prog_policies.search_methods import HillClimbing as _HC
    from sokoban_core import extract_features as _feats, iter_levels_in_file as _iter
    SokobanDSL = _DSL
    BoxobanTask = _Task
    ProgrammaticSpace = _Space
    HillClimbing = _HC
    extract_features = _feats
    iter_levels_in_file = _iter
    _imports_done = True


# --------------------------------------------------------------------------- #
# Step 0: load a level bank (real Boxoban levels across tiers), split pool/test
# --------------------------------------------------------------------------- #
def _sample_tier_levels(tier, n, rng):
    """Sample n parsed levels from a tier by reading a few random files."""
    _lazy_imports()
    files = sorted(TIER_DIRS[tier].glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no Boxoban level files in {TIER_DIRS[tier]}")
    rng.shuffle(files)
    out = []
    for f in files:
        levels = iter_levels_in_file(f, tier=tier)
        rng.shuffle(levels)
        out.extend(levels)
        if len(out) >= n:
            break
    return out[:n]


def load_level_bank(args, rng):
    """Sample a bank of real Boxoban levels across tiers and split into a disjoint
    pool-eligible set and a clean held-out test set (natural tier mix)."""
    # Bank composition: deliberately spans the full difficulty range so the dirty
    # pool can be built trivial-heavy while the clean test stays representative.
    plan = {"unfiltered": args.n_unfiltered, "medium": args.n_medium, "hard": args.n_hard}
    levels = []
    for tier, n in plan.items():
        levels.extend(_sample_tier_levels(tier, n, rng))
    rng.shuffle(levels)

    # Held-out selection. Default "mix": natural tier mixture drawn uniformly
    # (Step 2). A specific tier (e.g. "hard"): the DISTRIBUTION-SHIFT test (Step 3)
    # -- the deployment target is that tier, labelled by the environment's own
    # ground-truth tier and defined with zero reference to d_probe or the curation
    # rule, while the pool stays trivial-heavy.
    tier_sel = getattr(args, "heldout_tier", "mix")
    if tier_sel != "mix":
        in_tier = [l for l in levels if l.tier == tier_sel]
        off_tier = [l for l in levels if l.tier != tier_sel]
        test = in_tier[:args.heldout]
        pool_eligible = in_tier[args.heldout:] + off_tier
        rng.shuffle(pool_eligible)
        return pool_eligible, test

    # Reserve a CLEAN, natural-mix held-out test set FIRST (disjoint from the pool).
    # Drawn uniformly from the shuffled bank => reflects the natural tier mixture,
    # defined with zero reference to d_probe or the curation rule.
    test = levels[:args.heldout]
    pool_eligible = levels[args.heldout:]
    return pool_eligible, test


# --------------------------------------------------------------------------- #
# Step 1: cheap per-level scoring (static descriptors + mini-HC difficulty probe)
# --------------------------------------------------------------------------- #
def mini_hc_probe(ascii_str, sigma, probe_budget, probe_k, max_calls,
                  w_on_target, w_progress, seed):
    """Truncated random-restart hill climbing on ONE level; returns difficulty
    1 - best_reward. Cheap analogue of the full HC label (small budget)."""
    space = ProgrammaticSpace(SokobanDSL(), sigma)
    space.set_seed(seed)
    task = BoxobanTask({"ascii": ascii_str, "max_calls": max_calls, "crashable": False,
                        "w_on_target": w_on_target, "w_progress": w_progress}, seed=seed)
    best_ind, best_prog = space.initialize_individual()
    best_reward = task.evaluate_program(best_prog)
    global_best = best_reward
    while task.program_num < probe_budget and global_best < 1.0:
        improved = False
        for ind, prog in space.get_neighbors(best_ind, k=probe_k):
            r = task.evaluate_program(prog)
            if r > best_reward:
                best_ind, best_prog, best_reward = ind, prog, r
                global_best = max(global_best, r)
                improved = True
                break
            if task.program_num >= probe_budget:
                break
        if not improved:
            best_ind, best_prog = space.initialize_individual()
            best_reward = task.evaluate_program(best_prog)
            global_best = max(global_best, best_reward)
    return 1.0 - float(global_best)


def score_levels(levels, args, verbose=True):
    """Static descriptors (instant) + mini-HC d_probe for each level. Returns a
    DataFrame carrying level_id, tier, ascii, d_probe, and the descriptor columns."""
    _lazy_imports()
    rows = []
    t0 = time.time()
    for i, lvl in enumerate(levels):
        ascii_str = "\n".join(lvl.grid)
        feats = extract_features(lvl)
        d = np.mean([
            mini_hc_probe(ascii_str, args.sigma, args.probe_budget, args.probe_k,
                          args.max_calls, args.w_on_target, args.w_progress,
                          seed=1000 * (j + 1) + i)
            for j in range(args.probe_seeds)
        ])
        row = {"level_id": lvl.level_id, "tier": lvl.tier, "ascii": ascii_str,
               "d_probe": float(d)}
        row.update(feats)
        rows.append(row)
        if verbose and (i + 1) % 20 == 0:
            print(f"    scored {i + 1}/{len(levels)} levels "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Step 2: compose the contaminated pool (real levels, unflattering mixture)
# --------------------------------------------------------------------------- #
def compose_dirty_pool(bank, args, rng):
    """Build an N-level contaminated pool from the scored level bank.

    Tiers levels by d_probe (terciles by default), then samples trivial-heavy with a
    hard/degenerate tail and a few forced duplicate clusters. Returns (pool_df, info).
    Rows may repeat the same level_id by construction (the duplication pathology).
    """
    lo_q, hi_q = bank["d_probe"].quantile([args.trivial_q, args.hard_q])
    trivial = bank[bank["d_probe"] <= lo_q]
    hard = bank[bank["d_probe"] >= hi_q]
    informative = bank[(bank["d_probe"] > lo_q) & (bank["d_probe"] < hi_q)]

    n_triv = int(round(args.frac_trivial * args.pool_size))
    n_hard = int(round(args.frac_hard * args.pool_size))
    n_dup = int(round(args.frac_dup * args.pool_size))
    n_info = args.pool_size - n_triv - n_hard - n_dup
    if n_info < 0:  # guard: fractions sum > 1 -> clamp duplicates
        n_dup = max(0, n_dup + n_info)
        n_info = 0

    def draw(pool_df, n):
        if n <= 0 or len(pool_df) == 0:
            return []
        replace = n > len(pool_df)
        idx = rng.choice(pool_df.index.to_numpy(), size=n, replace=replace)
        return list(idx)

    chosen = []
    chosen += draw(trivial, n_triv)
    chosen += draw(hard, n_hard)
    chosen += draw(informative, n_info)

    # Forced duplicate clusters: pick a few informative levels, repeat each many times.
    dup_src = informative if len(informative) >= args.dup_clusters else bank
    if n_dup > 0 and len(dup_src) > 0:
        clusters = rng.choice(dup_src.index.to_numpy(),
                              size=min(args.dup_clusters, len(dup_src)), replace=False)
        per = int(np.ceil(n_dup / len(clusters)))
        dup_idx = []
        for c in clusters:
            dup_idx += [c] * per
        chosen += dup_idx[:n_dup]

    pool = bank.loc[chosen].reset_index(drop=True)
    pool["pool_id"] = np.arange(len(pool))
    info = {"lo_q": float(lo_q), "hi_q": float(hi_q), "n_trivial": n_triv,
            "n_hard": n_hard, "n_info": n_info, "n_dup": n_dup,
            "distinct_levels": int(pool["level_id"].nunique())}
    return pool, info


# --------------------------------------------------------------------------- #
# Step 3: the four subset rules (operate on pool ROWS; return pool_id lists)
# --------------------------------------------------------------------------- #
def _coverage_select(df, k, rng, random_start):
    """Greedy farthest-point (k-center) over standardized descriptors. Returns
    positional row indices into df."""
    from lavagap_curriculum import _standardize
    cols = [c for c in _SOKOBAN_DESCRIPTOR_COLS if c in df and df[c].std() > 1e-9]
    X = _standardize(df[cols].to_numpy(dtype=float))
    if random_start:
        start = int(rng.integers(len(df)))
    else:
        med = df["d_probe"].median()
        start = int((df["d_probe"] - med).abs().to_numpy().argmin())
    chosen = [start]
    min_d = np.linalg.norm(X - X[start], axis=1)
    while len(chosen) < k:
        nxt = int(min_d.argmax())
        if nxt in chosen:
            remaining = [j for j in range(len(df)) if j not in chosen]
            if not remaining:
                break
            nxt = int(rng.choice(remaining))
        chosen.append(nxt)
        min_d = np.minimum(min_d, np.linalg.norm(X - X[nxt], axis=1))
    return chosen


def random_subset(pool, k, rng):
    """Uniform random k pool ROWS (so duplicate rows can be drawn)."""
    idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
    return [int(pool.loc[i, "pool_id"]) for i in idx]


def band_only_subset(pool, k, band_lo, band_hi, rng):
    """Keep the medium d_probe quantile band, then random k within the band."""
    lo, hi = pool["d_probe"].quantile([band_lo, band_hi])
    band = pool[(pool["d_probe"] >= lo) & (pool["d_probe"] <= hi)].reset_index(drop=True)
    if len(band) <= k:
        band = pool.reset_index(drop=True)
    idx = rng.choice(len(band), size=min(k, len(band)), replace=False)
    return [int(band.loc[i, "pool_id"]) for i in idx]


def coverage_only_subset(pool, k, rng, random_start):
    """Farthest-point coverage over the WHOLE dirty pool (no band filter)."""
    df = pool.reset_index(drop=True)
    sel = _coverage_select(df, k, rng, random_start)
    return [int(df.loc[j, "pool_id"]) for j in sel]


def both_subset(pool, k, band_lo, band_hi, rng, random_start):
    """Full curation: difficulty band THEN farthest-point coverage within it."""
    lo, hi = pool["d_probe"].quantile([band_lo, band_hi])
    band = pool[(pool["d_probe"] >= lo) & (pool["d_probe"] <= hi)].reset_index(drop=True)
    if len(band) <= k:
        band = pool.reset_index(drop=True)
    sel = _coverage_select(band, k, rng, random_start)
    return [int(band.loc[j, "pool_id"]) for j in sel]


_CONDITIONS = ["random", "band", "coverage", "both"]


# --------------------------------------------------------------------------- #
# Step 4: learner (Option A) = HC one SokobanDSL program over the level SET
# --------------------------------------------------------------------------- #
def train_on_set(level_asciis, search_seed, n_iter, k_neighbors, max_calls,
                 w_on_target, w_progress, sigma):
    """Hill Climbing for a single program maximizing MEAN shaped reward over the set.

    Returns (program, train_mean_reward, n_program_evals, n_env_evals)."""
    _lazy_imports()
    task_envs = [BoxobanTask({"ascii": a, "max_calls": max_calls, "crashable": False,
                              "w_on_target": w_on_target, "w_progress": w_progress},
                             seed=search_seed) for a in level_asciis]
    space = ProgrammaticSpace(SokobanDSL(), sigma)
    space.set_seed(search_seed)
    hc = HillClimbing(k=k_neighbors, e=2)
    progs, rewards = hc.search(space, task_envs, seed=search_seed, n_iterations=n_iter)
    best_prog = progs[-1]
    n_prog_evals = task_envs[0].program_num
    n_env_evals = n_prog_evals * len(task_envs)
    return best_prog, float(rewards[-1]), int(n_prog_evals), int(n_env_evals)


def eval_program(program, eval_asciis, max_calls, w_on_target, w_progress):
    """Mean shaped reward + solve rate of one program on held-out levels."""
    _lazy_imports()
    rewards = []
    for a in eval_asciis:
        task = BoxobanTask({"ascii": a, "max_calls": max_calls, "crashable": False,
                            "w_on_target": w_on_target, "w_progress": w_progress}, seed=0)
        rewards.append(float(task.evaluate_program(program)))
    rewards = np.array(rewards)
    return float(rewards.mean()), float((rewards >= 0.999).mean())


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_everything(args):
    """Deterministically (given args.seed) build the bank, dirty pool, clean test,
    and the per-condition subset ensembles. Every shard agrees on the result."""
    _lazy_imports()
    rng = random.Random(args.seed)
    nprng = np.random.default_rng(args.seed)

    pool_eligible, test_levels = load_level_bank(args, rng)
    print(f"\n[1] Scoring {len(pool_eligible)} pool-eligible + {len(test_levels)} "
          f"held-out levels (static feats + mini-HC probe) ...", flush=True)
    bank = score_levels(pool_eligible, args, verbose=True)
    test_df = score_levels(test_levels, args, verbose=False)
    print(f"    d_probe range pool [{bank.d_probe.min():.2f},{bank.d_probe.max():.2f}] "
          f"test [{test_df.d_probe.min():.2f},{test_df.d_probe.max():.2f}]", flush=True)

    pool, info = compose_dirty_pool(bank, args, nprng)
    print(f"[2] Contaminated pool N={len(pool)}: trivial<= {info['lo_q']:.2f} "
          f"({info['n_trivial']}), informative ({info['n_info']}), "
          f"hard>= {info['hi_q']:.2f} ({info['n_hard']}), dup ({info['n_dup']}); "
          f"distinct levels {info['distinct_levels']}/{len(pool)}", flush=True)

    # Per-condition ensembles (matched n via args.draws). Distinct rng per draw.
    subsets = {c: [] for c in _CONDITIONS}
    for d in range(args.draws):
        drng = np.random.default_rng(args.seed * 7919 + 131 * d + 1)
        rstart = (d > 0)
        subsets["random"].append(random_subset(pool, args.k, drng))
        subsets["band"].append(band_only_subset(pool, args.k, args.band_lo, args.band_hi, drng))
        subsets["coverage"].append(coverage_only_subset(pool, args.k, drng, rstart))
        subsets["both"].append(both_subset(pool, args.k, args.band_lo, args.band_hi, drng, rstart))

    test_asciis = test_df["ascii"].tolist()
    return bank, pool, test_asciis, subsets, info


def _asciis_for(pool, pool_ids):
    """Map a list of pool_ids back to their ASCII grids."""
    by_id = dict(zip(pool["pool_id"], pool["ascii"]))
    return [by_id[i] for i in pool_ids]


def build_jobs(args, pool, subsets):
    """Ordered job list: full pool baseline, then each condition x draw x train-seed."""
    jobs = []
    full_ids = [int(i) for i in pool["pool_id"].to_numpy()]
    full_train_seeds = args.full_train_seeds if args.full_train_seeds > 0 else args.train_seeds
    if args.full_train_seeds < 0:
        full_train_seeds = 0  # skip the expensive full-pool baseline
    for si in range(full_train_seeds):
        jobs.append(("full", 0, full_ids, si))
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

    print(f"== Sokoban dirty-pool curation (Path 1) ==", flush=True)
    print(f"pool N={args.pool_size}  k={args.k}  held-out M={args.heldout}  "
          f"shard {args.shard}/{args.num_shards}", flush=True)

    bank, pool, test_asciis, subsets, info = build_everything(args)
    if args.shard == 0:
        pool.drop(columns=["ascii"]).to_csv(f"{args.out}_pool.csv", index=False)
        with open(f"{args.out}_meta.json", "w") as f:
            json.dump({"args": vars(args), "pool_info": info,
                       "subsets": {c: [list(map(int, s)) for s in subsets[c]]
                                   for c in _CONDITIONS}}, f, indent=2)

    jobs = build_jobs(args, pool, subsets)
    my_jobs = [(i, j) for i, j in enumerate(jobs) if i % args.num_shards == args.shard]
    print(f"\n[3] Running {len(my_jobs)}/{len(jobs)} training jobs for this shard\n", flush=True)

    results = []
    for idx, (cond, di, subset, si) in my_jobs:
        search_seed = 7000 + 101 * si + 13 * di
        asciis = _asciis_for(pool, subset)
        t0 = time.time()
        prog, train_r, n_prog, n_env = train_on_set(
            asciis, search_seed, args.n_iter, args.k_neighbors, args.max_calls,
            args.w_on_target, args.w_progress, args.sigma)
        ho_r, ho_solve = eval_program(prog, test_asciis, args.max_calls,
                                      args.w_on_target, args.w_progress)
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
    out = f"{args.out}_shard{args.shard}of{args.num_shards}.csv"
    res.to_csv(out, index=False)
    if args.num_shards == 1:
        summ = summarize(res)
        print(f"\n== SUMMARY (mean +/- std over runs) ==", flush=True)
        print(summ.to_string(), flush=True)
    print(f"\nwrote {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    # bank composition (spans full difficulty range)
    ap.add_argument("--n-unfiltered", type=int, default=240)
    ap.add_argument("--n-medium", type=int, default=240)
    ap.add_argument("--n-hard", type=int, default=120)
    ap.add_argument("--pool-size", type=int, default=300)
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--heldout", type=int, default=80)
    ap.add_argument("--heldout-tier", default="mix",
                    choices=["mix", "unfiltered", "medium", "hard"],
                    help="held-out difficulty: 'mix' = natural (Step 2); a tier "
                         "= distribution-shift target (Step 3).")
    # mini-HC probe (cheap difficulty estimate)
    ap.add_argument("--probe-budget", type=int, default=80)
    ap.add_argument("--probe-k", type=int, default=32)
    ap.add_argument("--probe-seeds", type=int, default=1)
    # learner
    ap.add_argument("--n-iter", type=int, default=250)
    ap.add_argument("--k-neighbors", type=int, default=128)
    ap.add_argument("--sigma", type=float, default=0.25)
    ap.add_argument("--max-calls", type=int, default=2000)
    ap.add_argument("--w-on-target", type=float, default=0.8)
    ap.add_argument("--w-progress", type=float, default=0.2)
    ap.add_argument("--train-seeds", type=int, default=6)
    ap.add_argument("--full-train-seeds", type=int, default=0,
                    help="train-seeds for the (expensive) full-pool baseline; "
                         "0 = use --train-seeds, <0 = skip full entirely.")
    ap.add_argument("--draws", type=int, default=6,
                    help="ensemble subsets per condition (matched n for a fair ablation)")
    ap.add_argument("--band-lo", type=float, default=0.15)
    ap.add_argument("--band-hi", type=float, default=0.85)
    # contamination recipe
    ap.add_argument("--frac-trivial", type=float, default=0.55)
    ap.add_argument("--frac-hard", type=float, default=0.15)
    ap.add_argument("--frac-dup", type=float, default=0.15)
    ap.add_argument("--dup-clusters", type=int, default=3)
    ap.add_argument("--trivial-q", type=float, default=0.33)
    ap.add_argument("--hard-q", type=float, default=0.67)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sokoban_dirty_pool")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny smoke config")
    args = ap.parse_args()

    if args.quick:
        args.n_unfiltered = 40
        args.n_medium = 40
        args.n_hard = 20
        args.pool_size = 60
        args.k = 12
        args.heldout = 16
        args.probe_budget = 30
        args.probe_k = 16
        args.probe_seeds = 1
        args.n_iter = 40
        args.k_neighbors = 24
        args.train_seeds = 1
        args.draws = 2

    run(args)


if __name__ == "__main__":
    main()
