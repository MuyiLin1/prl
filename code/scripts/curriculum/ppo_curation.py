#!/usr/bin/env python3
"""Option B: does cheap-signal CURATION beat random SELECTION for a NEURAL (PPO) learner?

Companion to sokoban_dirty_pool.py (Option A / single-program learner). The Option-A learner
(one hill-climbed DSL program) is too low-capacity to exploit better training data, so curation
tied random there even in a graded domain and under distribution shift. PPO has the capacity to
learn transferable skill from harder training layouts, so this is the FAIR test of the selection
claim -- the same learner that already benefits from difficulty ORDERING in the paper's §6.7.

Design (mirrors the dirty-pool ablation, PPO learner, LavaCrossing domain):
  * Pool: real LavaCrossing layouts over (size x rivers), scored cheaply (geometry + random-walk
    probe; NO training). Composed EASY-HEAVY (trivial-heavy by the cheap score) with a thin
    informative/hard tail + duplicate clusters -- the realistic "dirty pool".
  * Subsets (matched size k, matched #draws): random / band / coverage / both.
  * Learner: PPO trained on ONLY the k selected layouts (fixed active set, no unlocking), so the
    CHOICE of training tasks is the only thing that varies.
  * Held-out: 'mix' natural target (Step 2) or 'hard' distribution-shift target (Step 3), labelled
    by the environment's own (size, rivers) ground truth -- zero reference to the cheap score or
    the curation rule (breaks circularity).

Headline: held-out success (fraction of held-out layouts solved), curated vs random, over
draws x seeds. Bar: SUBSTANTIAL (>=~0.05, non-overlapping CIs), not merely significant.

NO LLM. Standard gymnasium MiniGrid CrossingEnv.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from difficulty_scoring import (
    Layout, enumerate_pool, geometric_score, random_walk_probe, feature_vector,
)
from curriculum_harness import CurriculumPoolEnv, make_ppo, eval_layouts

_HERE = Path(__file__).resolve().parent
_CONDITIONS = ["random", "band", "coverage", "both"]


# --------------------------------------------------------------------------- #
# Cheap scoring: normalized difficulty (geometry + probe) used by the curation rule
# --------------------------------------------------------------------------- #
def _rank01(v):
    v = np.asarray(v, dtype=float)
    r = pd.Series(v).rank().to_numpy() - 1.0
    return r / max(1.0, (len(v) - 1))


def score_pool(pool, probe_k, seed, verbose=True):
    """Cheap difficulty d_cheap in [0,1] (higher=harder) from geometry + random-walk probe,
    plus the coverage feature matrix. NO training, NO ground-truth tier used."""
    t0 = time.time()
    g = np.array([geometric_score(lo) for lo in pool])
    p = np.array([random_walk_probe(lo, k=probe_k, seed=seed) for lo in pool])
    d_cheap = 0.5 * (_rank01(g) + _rank01(p))          # rank-average of the two cheap signals
    if verbose:
        print(f"    scored {len(pool)} layouts in {time.time()-t0:.0f}s "
              f"(d_cheap {d_cheap.min():.2f}-{d_cheap.max():.2f})", flush=True)
    rows = []
    for lo, dc, gg, pp in zip(pool, d_cheap, g, p):
        rows.append({"sig": lo.sig, "size": lo.size, "n": lo.n, "seed": lo.seed,
                     "d_cheap": float(dc), "g": float(gg), "probe": float(pp)})
    df = pd.DataFrame(rows)
    df["_idx"] = np.arange(len(df))                     # positional handle into `pool`
    return df


# --------------------------------------------------------------------------- #
# Dirty-pool composition: easy-heavy + informative/hard tail + duplicate clusters
# --------------------------------------------------------------------------- #
def compose_dirty_pool(bank, args, rng):
    lo_q, hi_q = np.quantile(bank["d_cheap"], [args.trivial_q, args.hard_q])
    triv = bank[bank["d_cheap"] <= lo_q]
    hard = bank[bank["d_cheap"] >= hi_q]
    info = bank[(bank["d_cheap"] > lo_q) & (bank["d_cheap"] < hi_q)]

    n_triv = int(round(args.frac_trivial * args.pool_size))
    n_hard = int(round(args.frac_hard * args.pool_size))
    n_dup = int(round(args.frac_dup * args.pool_size))
    n_info = max(0, args.pool_size - n_triv - n_hard - n_dup)

    def draw(sub, m, replace):
        if len(sub) == 0 or m == 0:
            return sub.iloc[[]]
        idx = rng.choice(sub["_idx"].to_numpy(), size=m, replace=replace or m > len(sub))
        return bank.set_index("_idx").loc[idx].reset_index()

    parts = [draw(triv, n_triv, False), draw(info, n_info, False), draw(hard, n_hard, False)]
    # duplicate clusters: a few informative layouts repeated many times
    if n_dup > 0 and len(info) > 0:
        clusters = rng.choice(info["_idx"].to_numpy(),
                              size=min(args.dup_clusters, len(info)), replace=False)
        reps = rng.choice(clusters, size=n_dup, replace=True)
        parts.append(bank.set_index("_idx").loc[reps].reset_index())
    pool = pd.concat(parts, ignore_index=True)
    pool["pool_id"] = np.arange(len(pool))
    info_str = dict(lo_q=float(lo_q), hi_q=float(hi_q), n_triv=int(n_triv),
                    n_info=int(n_info), n_hard=int(n_hard), n_dup=int(n_dup),
                    distinct=int(pool["_idx"].nunique()))
    return pool, info_str


# --------------------------------------------------------------------------- #
# Subset selectors (matched size k)
# --------------------------------------------------------------------------- #
def _standardize(X):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    return (X - mu) / sd


def _coverage(df, pool_vecs, k, rng, random_start):
    X = _standardize(np.array([pool_vecs[i] for i in df["_idx"].to_numpy()], dtype=float))
    start = int(rng.integers(len(df))) if random_start else \
        int((df["d_cheap"] - df["d_cheap"].median()).abs().to_numpy().argmin())
    chosen = [start]
    mind = np.linalg.norm(X - X[start], axis=1)
    while len(chosen) < k:
        nxt = int(mind.argmax())
        if nxt in chosen:
            rem = [j for j in range(len(df)) if j not in chosen]
            if not rem:
                break
            nxt = int(rng.choice(rem))
        chosen.append(nxt)
        mind = np.minimum(mind, np.linalg.norm(X - X[nxt], axis=1))
    return df.iloc[chosen]["pool_id"].tolist()


def select_subset(cond, pool, pool_vecs, k, band_lo, band_hi, rng, random_start):
    if cond == "random":
        idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
        return pool.iloc[idx]["pool_id"].tolist()
    lo, hi = np.quantile(pool["d_cheap"], [band_lo, band_hi])
    band = pool[(pool["d_cheap"] >= lo) & (pool["d_cheap"] <= hi)]
    if cond == "band":
        b = band if len(band) > k else pool
        idx = rng.choice(len(b), size=min(k, len(b)), replace=False)
        return b.iloc[idx]["pool_id"].tolist()
    if cond == "coverage":
        return _coverage(pool.reset_index(drop=True), pool_vecs, k, rng, random_start)
    if cond == "both":
        b = (band if len(band) > k else pool).reset_index(drop=True)
        return _coverage(b, pool_vecs, k, rng, random_start)
    raise ValueError(cond)


# --------------------------------------------------------------------------- #
# Learner (Option B): PPO on the selected subset, then held-out success.
# --order trains an easy->hard curriculum WITHIN the subset, using our difficulty
# score to bin+unlock (random selection cannot self-order -> that ordering is the
# signal's contribution). Flat mode trains on the whole subset at once.
# --------------------------------------------------------------------------- #
def _make_bins(layouts, scores, n_bins):
    order = np.argsort(np.asarray(scores, dtype=float))     # easy (low d_cheap) first
    ordered = [layouts[i] for i in order]
    bins = [list(c) for c in np.array_split(ordered, n_bins)]
    return [b for b in bins if len(b) > 0] or [list(layouts)]


def train_and_eval(subset_layouts, subset_scores, held_out, args, seed, restrict):
    env = CurriculumPoolEnv(restrict, rng_seed=seed)
    model = make_ppo(env, seed=seed)
    if not args.order:
        env.set_active(subset_layouts)          # flat: whole subset from the start
        model.learn(total_timesteps=args.total_steps, reset_num_timesteps=True)
        return float(np.mean(eval_layouts(model, held_out, args.eval_episodes, restrict)))

    # easy->hard curriculum within the subset, ordered by our difficulty score
    bins = _make_bins(subset_layouts, subset_scores, args.n_bins)
    frontier, frontier_start = 0, 0
    env.set_active(bins[0])
    n_ck = max(1, args.total_steps // args.eval_every)
    bin_deadline = int(args.bin_deadline_frac * args.total_steps)
    for ck in range(1, n_ck + 1):
        model.learn(total_timesteps=args.eval_every, reset_num_timesteps=(ck == 1))
        env_steps = ck * args.eval_every
        if frontier < len(bins) - 1:
            active = [lo for b in bins[:frontier + 1] for lo in b]
            gate = float(np.mean(eval_layouts(model, active, args.gate_episodes, restrict)))
            if gate >= args.gate or (env_steps - frontier_start) >= bin_deadline:
                frontier += 1
                frontier_start = env_steps
                env.set_active([lo for b in bins[:frontier + 1] for lo in b])
    return float(np.mean(eval_layouts(model, held_out, args.eval_episodes, restrict)))


# --------------------------------------------------------------------------- #
# Held-out construction (ground-truth difficulty = the (size, rivers) cell)
# --------------------------------------------------------------------------- #
def build_heldout(args, train_sigs):
    ev = enumerate_pool(args.sizes, args.tiers, args.eval_n_seeds, args.eval_cap_per_cell,
                        seed_start=args.eval_seed_start, exclude_sigs=train_sigs)
    if args.heldout_n > 0:                     # explicit rivers-tier target (learnable medium)
        ev = [lo for lo in ev if lo.n == args.heldout_n]
    elif args.heldout_tier == "hard":
        nmax, smax = max(args.tiers), max(args.sizes)
        # hard cell = most rivers, and at least the median grid size
        smed = sorted(args.sizes)[len(args.sizes) // 2]
        hard = [lo for lo in ev if lo.n == nmax and lo.size >= smed]
        ev = hard if len(hard) >= args.heldout else [lo for lo in ev if lo.n == nmax]
    rng = np.random.default_rng(12345)
    if len(ev) > args.heldout:
        ev = [ev[i] for i in rng.choice(len(ev), size=args.heldout, replace=False)]
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[9, 13, 17])
    ap.add_argument("--tiers", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n-seeds", type=int, default=120)
    ap.add_argument("--cap-per-cell", type=int, default=40)
    ap.add_argument("--pool-size", type=int, default=120)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--heldout", type=int, default=60)
    ap.add_argument("--heldout-tier", default="hard", choices=["mix", "hard"],
                    help="'mix' natural target (Step 2) or 'hard' distribution-shift (Step 3)")
    ap.add_argument("--heldout-n", type=int, default=0,
                    help="if >0, held-out is exactly this rivers tier across all sizes "
                         "(overrides --heldout-tier; use a learnable medium tier)")
    ap.add_argument("--eval-seed-start", type=int, default=100_000)
    ap.add_argument("--eval-n-seeds", type=int, default=300)
    ap.add_argument("--eval-cap-per-cell", type=int, default=30)
    ap.add_argument("--eval-episodes", type=int, default=4)
    ap.add_argument("--probe-k", type=int, default=8)
    ap.add_argument("--total-steps", type=int, default=250_000)
    ap.add_argument("--order", action="store_true",
                    help="train each subset as an easy->hard curriculum ordered by our "
                         "difficulty score (random selection cannot self-order)")
    ap.add_argument("--n-bins", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=25_000)
    ap.add_argument("--gate", type=float, default=0.5)
    ap.add_argument("--gate-episodes", type=int, default=3)
    ap.add_argument("--bin-deadline-frac", type=float, default=0.34)
    ap.add_argument("--draws", type=int, default=6)
    ap.add_argument("--train-seeds", type=int, default=2)
    ap.add_argument("--band-lo", type=float, default=0.15)
    ap.add_argument("--band-hi", type=float, default=0.85)
    ap.add_argument("--frac-trivial", type=float, default=0.55)
    ap.add_argument("--frac-hard", type=float, default=0.15)
    ap.add_argument("--frac-dup", type=float, default=0.15)
    ap.add_argument("--dup-clusters", type=int, default=3)
    ap.add_argument("--trivial-q", type=float, default=0.33)
    ap.add_argument("--hard-q", type=float, default=0.67)
    ap.add_argument("--conditions", nargs="+", default=_CONDITIONS)
    ap.add_argument("--no-restrict-actions", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ppo_curation")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.sizes, args.tiers = [9, 13], [1, 2, 3]
        args.n_seeds, args.cap_per_cell = 40, 12
        args.pool_size, args.k, args.heldout = 48, 10, 20
        args.eval_n_seeds, args.eval_cap_per_cell = 100, 8
        args.total_steps = 30_000
        args.draws, args.train_seeds = 3, 1

    import warnings
    warnings.filterwarnings("ignore")
    restrict = not args.no_restrict_actions

    print(f"== PPO curation (Option B): heldout={args.heldout_tier}  "
          f"pool N={args.pool_size}  k={args.k}  order={args.order} ==", flush=True)

    # 1. build + cheap-score the layout bank
    bank_pool = enumerate_pool(args.sizes, args.tiers, args.n_seeds, args.cap_per_cell)
    print(f"[1] bank {len(bank_pool)} distinct layouts; cheap scoring ...", flush=True)
    bank = score_pool(bank_pool, args.probe_k, args.seed)
    pool_vecs = {row["_idx"]: feature_vector(bank_pool[row["_idx"]]) for _, row in bank.iterrows()}

    # 2. contaminated pool + held-out (disjoint, ground-truth-tier target)
    nprng = np.random.default_rng(args.seed)
    pool, info = compose_dirty_pool(bank, args, nprng)
    print(f"[2] pool N={len(pool)}: trivial<= {info['lo_q']:.2f} ({info['n_triv']}), "
          f"informative ({info['n_info']}), hard>= {info['hi_q']:.2f} ({info['n_hard']}), "
          f"dup ({info['n_dup']}); distinct {info['distinct']}", flush=True)
    train_sigs = {bank_pool[i].sig for i in pool["_idx"].unique()}
    held_out = build_heldout(args, train_sigs)
    hc = {}
    for lo in held_out:
        hc[(lo.size, lo.n)] = hc.get((lo.size, lo.n), 0) + 1
    print(f"    held-out {len(held_out)} layouts ({args.heldout_tier}); cells "
          f"{ {f's{k[0]}N{k[1]}': v for k, v in sorted(hc.items())} }", flush=True)

    # id -> Layout and id -> difficulty score for building PPO active sets / ordering
    id2layout = {int(r["pool_id"]): bank_pool[int(r["_idx"])] for _, r in pool.iterrows()}
    id2score = {int(r["pool_id"]): float(r["d_cheap"]) for _, r in pool.iterrows()}

    # 3. subsets (matched n via draws; distinct rng per draw)
    subsets = {c: [] for c in args.conditions}
    for d in range(args.draws):
        drng = np.random.default_rng(args.seed * 7919 + 131 * d + 1)
        for c in args.conditions:
            subsets[c].append(select_subset(c, pool, pool_vecs, args.k,
                                             args.band_lo, args.band_hi, drng, random_start=(d > 0)))

    # 4. train PPO on each subset, eval on held-out (draw-major so early jobs
    #    cover every condition -> fast read on whether the learner discriminates)
    rows, job = [], 0
    total = len(args.conditions) * args.draws * args.train_seeds
    for d in range(args.draws):
        for c in args.conditions:
            ids = subsets[c][d]
            layouts = [id2layout[i] for i in ids]
            scores = [id2score[i] for i in ids]
            for s in range(args.train_seeds):
                t0 = time.time()
                held = train_and_eval(layouts, scores, held_out, args, seed=1000 * s + d, restrict=restrict)
                job += 1
                print(f"  [{job:>3}/{total}] {c:>8} draw{d} seed{s}: held={held:.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                rows.append({"condition": c, "draw": d, "train_seed": s, "held": held})

    df = pd.DataFrame(rows)
    out = _HERE / f"{args.out}_{args.heldout_tier}.csv"
    df.to_csv(out, index=False)

    print("\n== SUMMARY (held-out success, mean +/- std) ==")
    g = df.groupby("condition")["held"].agg(["mean", "std", "count"])
    print(g.to_string())
    if "random" in g.index:
        rmean = g.loc["random", "mean"]
        print(f"\n  vs random (mean {rmean:.3f}):")
        for c in args.conditions:
            if c != "random":
                print(f"    {c:>8}: {g.loc[c,'mean']-rmean:+.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
