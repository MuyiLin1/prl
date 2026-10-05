#!/usr/bin/env python3
"""
Sample a stratified Boxoban level pool, compute the budgeted-solver difficulty label
and the cheap geometric descriptor for each level, and report:

  * split-half reliability of the label (Eq. reliability),
  * per-tier mean difficulty (external validity vs. Boxoban's unfiltered/medium/hard
    difficulty tiers),
  * the strongest single geometric-descriptor correlations,

then write a CSV (features + label) and a JSON (per-seed rewards) for the predictor
step. This is the fast-path de-risk check from the section-6.5 plan: it answers
whether a budgeted search yields a graded, reliable Sokoban difficulty label before
any DSL/Hill-Climbing solver is built.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr

from sokoban_core import SokobanLevel, extract_features, iter_levels_in_file
from solver import SolverConfig, difficulty_label

_ROOT = Path(__file__).resolve().parents[2]
_DATA = _ROOT / "data" / "boxoban"

# tier -> (subdir holding .txt level files)
TIER_DIRS = {
    "unfiltered": _DATA / "unfiltered" / "train",
    "medium": _DATA / "medium" / "train",
    "hard": _DATA / "hard",
}


def sample_levels(tier: str, n: int, rng: random.Random) -> List[SokobanLevel]:
    """Sample n levels from a tier by drawing from a few random files."""
    d = TIER_DIRS[tier]
    files = sorted(d.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no level files in {d}")
    rng.shuffle(files)
    out: List[SokobanLevel] = []
    for f in files:
        levels = iter_levels_in_file(f, tier=tier)
        rng.shuffle(levels)
        out.extend(levels)
        if len(out) >= n:
            break
    rng.shuffle(out)
    return out[:n]


def split_half_reliability(per_seed: np.ndarray, n_partitions: int = 20,
                           rng: random.Random | None = None) -> float:
    """Spearman-Brown-corrected split-half reliability over random seed partitions.
    per_seed: (n_levels, n_seeds) array of best-rewards."""
    rng = rng or random.Random(0)
    n_levels, n_seeds = per_seed.shape
    if n_seeds < 2 or n_levels < 3:
        return float("nan")
    idx = list(range(n_seeds))
    corrs = []
    for _ in range(n_partitions):
        rng.shuffle(idx)
        a, b = idx[: n_seeds // 2], idx[n_seeds // 2:]
        da = 1.0 - per_seed[:, a].mean(axis=1)
        db = 1.0 - per_seed[:, b].mean(axis=1)
        if np.std(da) < 1e-9 or np.std(db) < 1e-9:
            continue
        rho = spearmanr(da, db).correlation
        if not np.isnan(rho):
            corrs.append(rho)
    if not corrs:
        return float("nan")
    r = float(np.mean(corrs))
    return (2 * r) / (1 + r) if (1 + r) != 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-unfiltered", type=int, default=70)
    ap.add_argument("--n-medium", type=int, default=70)
    ap.add_argument("--n-hard", type=int, default=60)
    ap.add_argument("--n-seeds", type=int, default=12)
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", default=str(_ROOT / "data" / "sokoban_fastpath"))
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cfg = SolverConfig(budget=args.budget)

    plan = {"unfiltered": args.n_unfiltered, "medium": args.n_medium, "hard": args.n_hard}
    levels: List[SokobanLevel] = []
    for tier, n in plan.items():
        levels.extend(sample_levels(tier, n, rng))
    print(f"sampled {len(levels)} levels: "
          + ", ".join(f"{t}={n}" for t, n in plan.items()))

    rows: List[Dict[str, float]] = []
    per_seed_mat: List[List[float]] = []
    tiers: List[str] = []
    t0 = time.time()
    for i, lvl in enumerate(levels):
        feats = extract_features(lvl)
        lab = difficulty_label(lvl, n_seeds=args.n_seeds, cfg=cfg)
        row: Dict[str, float] = {
            "level_id": lvl.level_id,
            "tier": lvl.tier,
            "difficulty": lab["difficulty"],
            "mean_reward": lab["mean_reward"],
            "solved_frac": lab["solved_frac"],
        }
        row.update(feats)
        rows.append(row)
        per_seed_mat.append(lab["per_seed_reward"])
        tiers.append(lvl.tier)
        if not args.no_progress and (i + 1) % 25 == 0:
            print(f"  labeled {i+1}/{len(levels)}  ({time.time()-t0:.0f}s)")

    per_seed = np.array(per_seed_mat, dtype=float)
    difficulty = np.array([r["difficulty"] for r in rows], dtype=float)
    tiers_arr = np.array(tiers)

    # ---- reliability ----
    rel = split_half_reliability(per_seed, rng=random.Random(args.seed))

    # ---- tier separation (external validity) ----
    tier_means = {t: float(difficulty[tiers_arr == t].mean()) for t in plan}
    tier_order = {"unfiltered": 0, "medium": 1, "hard": 2}
    tier_rank = np.array([tier_order[t] for t in tiers])
    tier_spearman = spearmanr(tier_rank, difficulty).correlation

    # ---- geometry single-feature correlations ----
    feat_keys = [k for k in rows[0] if k not in
                 ("level_id", "tier", "difficulty", "mean_reward", "solved_frac")]
    geo_corrs = []
    for k in feat_keys:
        col = np.array([r[k] for r in rows], dtype=float)
        if np.std(col) < 1e-9:
            continue
        rho = spearmanr(col, difficulty).correlation
        if not np.isnan(rho):
            geo_corrs.append((k, rho))
    geo_corrs.sort(key=lambda kv: abs(kv[1]), reverse=True)

    # ---- write outputs ----
    csv_path = Path(args.out_prefix + "_labels.csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = Path(args.out_prefix + "_perseed.json")
    json_path.write_text(json.dumps(
        {"level_id": [r["level_id"] for r in rows],
         "tier": tiers,
         "per_seed_reward": per_seed.tolist()}, indent=0))

    # ---- report ----
    print("\n" + "=" * 60)
    print(f"SOKOBAN FAST-PATH RESULT  ({len(levels)} levels, "
          f"R={args.n_seeds} seeds, budget={args.budget})")
    print("=" * 60)
    print(f"label split-half reliability  rho_rel = {rel:.3f}")
    print(f"difficulty variance           Var(D)  = {difficulty.var():.4f} "
          f"[range {difficulty.min():.3f}..{difficulty.max():.3f}]")
    print(f"overall solved fraction               = "
          f"{np.mean([r['solved_frac'] for r in rows]):.3f}")
    print("\nper-tier mean difficulty (external validity):")
    for t in plan:
        print(f"  {t:11s} mean D = {tier_means[t]:.3f}")
    print(f"  tier-vs-D Spearman = {tier_spearman:.3f}  "
          f"(expect >0: harder tier -> higher D)")
    print("\ntop-8 single geometric/reward feature correlations vs D:")
    for k, rho in geo_corrs[:8]:
        print(f"  {k:28s} rho_s = {rho:+.3f}")
    print(f"\nwrote {csv_path.name}, {json_path.name}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
