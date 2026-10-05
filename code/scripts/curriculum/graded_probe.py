#!/usr/bin/env python3
"""Step 1: make the cheap curation probe GRADED instead of near-binary.

Diagnosis (handoff/README.md): on LavaGap the current selection signal
(`truncated_search_difficulty_lavagap`) is `1 - max(shaped_reward)` over ~11
random programs near one base. That collapses to a few discrete values
(0.5/0.667/0.75/0.833/1.0) and leaves the medium difficulty tier EMPTY, so the
curation difficulty-band step has nothing to select. This script explores two
independent fixes and their combination, on identical rollouts so they are
directly comparable:

    Way A  (budget):    evaluate more programs via a real hill-climb WALK, so the
                        search reaches finer progress depths.
    Way B  (aggregate): score with 1 - mean(reward) (smooth) instead of
                        1 - max(reward) (snaps to discrete depths).
    Combined:           big budget + mean aggregate.

For each variant we report, against a full-HC "true" difficulty label:
  * Spearman rho   (is the graded signal still MEANINGFUL, not just smooth?)
  * n_unique       (how many distinct values -- de-clumping)
  * mid-band frac  (fraction of the pool landing in the MIDDLE THIRD of the score
                    range -- i.e. does a "medium" tier finally exist?)

This is a DIAGNOSTIC on LavaGap. LavaGap difficulty is intrinsically ~flat, so the
goal here is NOT to beat random; it is to confirm the graded-probe machinery
de-clumps and still tracks the true label, before carrying the best recipe to a
genuinely graded environment (Step 2).

NO LLM anywhere.
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

_imports_done = False


def _lazy_imports():
    global _imports_done, MinigridDSL, MinigridProgrammaticSpace, HillClimbing
    global make_shaped_lavagap, run_hc_shaped_lavagap
    if _imports_done:
        return
    from prog_policies.utils import get_env_name  # noqa: F401 -- circular-import fix
    from prog_policies.minigrid.dsl import MinigridDSL as _DSL
    from prog_policies.search_space import MinigridProgrammaticSpace as _Space
    from prog_policies.search_methods import HillClimbing as _HC
    from minigrid_shaped import make_shaped_lavagap as _make, run_hc_shaped_lavagap as _fullhc
    MinigridDSL = _DSL
    MinigridProgrammaticSpace = _Space
    HillClimbing = _HC
    make_shaped_lavagap = _make
    run_hc_shaped_lavagap = _fullhc
    _imports_done = True


# --------------------------------------------------------------------------- #
# One truncated hill-climb walk that records EVERY evaluation.
# --------------------------------------------------------------------------- #
def probe_run(env_seed, size, budget, search_seed, batch=8):
    """First-improvement hill-climb on one shaped-LavaGap task, capped at `budget`
    program evaluations. Returns (all_rewards, best_curve):
        all_rewards : shaped reward of every program evaluated (len <= budget)
        best_curve  : best-so-far shaped reward after each evaluation (monotone)
    """
    _lazy_imports()
    dsl = MinigridDSL()
    task_envs = [make_shaped_lavagap(env_seed, size=size)]
    space = MinigridProgrammaticSpace(dsl, sigma=0.25)
    space.set_seed(search_seed + 7919)
    hc = HillClimbing(k=batch, e=2)

    best_ind, best_prog = space.initialize_individual()
    all_rewards, best_curve = [], []
    r = hc.evaluate_program(best_prog, task_envs)
    best_r = r
    all_rewards.append(r)
    best_curve.append(best_r)

    while len(all_rewards) < budget:
        improved = False
        for nind, nprog in space.get_neighbors(best_ind, k=batch):
            rr = hc.evaluate_program(nprog, task_envs)
            all_rewards.append(rr)
            if rr > best_r:                 # first-improvement: walk to it
                best_ind, best_r = nind, rr
                improved = True
            best_curve.append(best_r)
            if len(all_rewards) >= budget:
                break
            if improved:
                break
        if best_r >= 1.0:                   # solved -> nothing left to grade
            break
        if not improved and len(all_rewards) >= budget:
            break
    return all_rewards, best_curve


def probe_scores(env_seed, size, budget, probe_seeds, topk=3):
    """Average the graded scores over a few probe search seeds for stability.
    Returns dict of candidate difficulty scores in [0, 1] (higher = harder)."""
    d_max, d_mean, d_auc, d_top = [], [], [], []
    for j in range(probe_seeds):
        allr, curve = probe_run(env_seed, size, budget, search_seed=1000 * (j + 1) + env_seed)
        top = np.sort(allr)[::-1][:topk]              # best few programs
        d_max.append(1.0 - max(allr))                 # current-style (Way baseline)
        d_mean.append(1.0 - float(np.mean(allr)))     # Way B: mean over all evals
        d_auc.append(1.0 - float(np.mean(curve)))     # Way B': area under best-so-far
        d_top.append(1.0 - float(np.mean(top)))       # Way B'': mean of best few
    return {
        "d_max": float(np.mean(d_max)),
        "d_mean": float(np.mean(d_mean)),
        "d_auc": float(np.mean(d_auc)),
        "d_top": float(np.mean(d_top)),
    }


def true_label(env_seed, size, R, base_seed=7000):
    """Full-HC difficulty label = 1 - mean best shaped reward over R search seeds."""
    _lazy_imports()
    vals = [run_hc_shaped_lavagap(env_seed, search_seed=base_seed + r, n_iterations=2000, size=size)
            for r in range(R)]
    return 1.0 - float(np.mean(vals))


# --------------------------------------------------------------------------- #
# Grading quality metrics
# --------------------------------------------------------------------------- #
def _spearman(a, b):
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if np.std(ra) < 1e-9 or np.std(rb) < 1e-9:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _mid_band_frac(scores):
    """Fraction of the pool whose score lies in the MIDDLE THIRD of the score
    range -- a direct read on whether a 'medium' tier exists."""
    s = np.asarray(scores, dtype=float)
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9:
        return 0.0
    a, b = lo + (hi - lo) / 3.0, lo + 2.0 * (hi - lo) / 3.0
    return float(np.mean((s >= a) & (s <= b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-size", type=int, default=50)
    ap.add_argument("--size", type=int, default=9)
    ap.add_argument("--budgets", type=int, nargs="+", default=[10, 30, 50])
    ap.add_argument("--probe-seeds", type=int, default=3)
    ap.add_argument("--true-R", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", type=str, default="graded_probe_lavagap.csv")
    args = ap.parse_args()

    pool = list(range(args.seed0, args.seed0 + args.pool_size))

    print(f"== Step 1: graded probe on LavaGap (size {args.size}, N={len(pool)}) ==", flush=True)
    print(f"[true label] full HC, R={args.true_R} seeds ...", flush=True)
    t0 = time.time()
    truth = {s: true_label(s, args.size, args.true_R) for s in pool}
    print(f"    done in {time.time() - t0:.1f}s", flush=True)

    rows = []
    for budget in args.budgets:
        print(f"[probe] budget={budget} evals, probe_seeds={args.probe_seeds} ...", flush=True)
        t0 = time.time()
        scored = {s: probe_scores(s, args.size, budget, args.probe_seeds) for s in pool}
        for s in pool:
            rows.append({"env_seed": s, "budget": budget, "true": truth[s], **scored[s]})
        print(f"    done in {time.time() - t0:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    tvec = np.array([truth[s] for s in pool])
    print("\n=== grading quality (vs full-HC true label) ===")
    print(f"true label: n_unique={len(np.unique(np.round(tvec,4)))}  "
          f"range=[{tvec.min():.3f},{tvec.max():.3f}]  mid-band={_mid_band_frac(tvec):.2f}")
    header = f"{'budget':>7} {'aggregate':>10} {'rho_true':>9} {'n_uniq':>7} {'mid-band':>9} {'range':>15}"
    print(header)
    print("-" * len(header))
    for budget in args.budgets:
        sub = df[df.budget == budget]
        for agg in ["d_max", "d_mean", "d_auc", "d_top"]:
            sc = sub[agg].to_numpy()
            rho = _spearman(sc, sub["true"].to_numpy())
            nun = len(np.unique(np.round(sc, 4)))
            mid = _mid_band_frac(sc)
            rng = f"[{sc.min():.3f},{sc.max():.3f}]"
            tag = "  <- current" if (budget == args.budgets[0] and agg == "d_max") else ""
            print(f"{budget:>7} {agg:>10} {rho:>9.3f} {nun:>7} {mid:>9.2f} {rng:>15}{tag}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
