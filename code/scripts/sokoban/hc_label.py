#!/usr/bin/env python
"""Difficulty labelling for Sokoban via the real DSL + Hill Climbing reference
solver (apples-to-apples with Karel / MiniGrid).

For each Boxoban level we run R independent HC seeds over
``ProgrammaticSpace(SokobanDSL, sigma)``. Each seed searches under a fixed
program-evaluation budget, restarting from a fresh random program whenever it
hits a local maximum (bounded random-restart hill climbing). The difficulty
label is:

    D_HC(level) = 1 - (1/R) * sum_seeds( best shaped reward )

identical in form to the Karel / MiniGrid label. Reliability is the split-half
Spearman-Brown coefficient over the per-seed rewards.

Usage (local pilot):
    python scripts/sokoban/hc_label.py --limit 12 --num-seeds 4 \
        --budget 800 --k 64 --out data/sokoban_hc_pilot

Usage (full cluster run):
    python scripts/sokoban/hc_label.py --labels data/sokoban_fastpath_labels.csv \
        --num-seeds 12 --budget 4000 --k 250 --sigma 0.25 \
        --out data/sokoban_hc_labels
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
# Make both the repo root and ./leaps importable, mirroring scripts/baseline.py.
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "leaps"))
sys.path.insert(0, str(_HERE))

from sokoban_core import iter_levels_in_file  # noqa: E402

# Import the utils package first to establish prog_policies' safe init order
# (its __init__ pulls base fully before the task packages), mirroring baseline.py.
import prog_policies.utils  # noqa: E402, F401

from prog_policies.sokoban.dsl import SokobanDSL  # noqa: E402
from prog_policies.sokoban_tasks.boxoban import BoxobanTask  # noqa: E402
from prog_policies.search_space import ProgrammaticSpace  # noqa: E402

TIER_DIRS = {
    "unfiltered": _ROOT / "data/boxoban/unfiltered/train",
    "medium": _ROOT / "data/boxoban/medium/train",
    "hard": _ROOT / "data/boxoban/hard",
}


def load_level_index(labels_csv: Path) -> list[tuple[str, str]]:
    """Return [(level_id, tier), ...] in CSV order so we label the same levels."""
    out = []
    with open(labels_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            out.append((row["level_id"], row["tier"]))
    return out


def build_ascii_cache(level_index: list[tuple[str, str]]) -> dict[str, str]:
    """Reconstruct each level's ASCII grid by reading each source file once."""
    by_file: dict[tuple[str, str], list[str]] = {}
    for level_id, tier in level_index:
        parts = level_id.split("/")
        filestem = parts[1] if len(parts) >= 3 else parts[0]
        by_file.setdefault((tier, filestem), []).append(level_id)

    ascii_cache: dict[str, str] = {}
    for (tier, filestem), lids in by_file.items():
        fpath = TIER_DIRS[tier] / f"{filestem}.txt"
        wanted = set(lids)
        for lvl in iter_levels_in_file(fpath, tier=tier):
            if lvl.level_id in wanted:
                ascii_cache[lvl.level_id] = "\n".join(lvl.grid)
    return ascii_cache


def run_hc_seed(ascii_str: str, dsl: SokobanDSL, sigma: float, seed: int,
                budget: int, k: int, max_calls: int,
                w_on_target: float, w_progress: float) -> float:
    """Bounded random-restart hill climbing on one level; returns best reward."""
    search_space = ProgrammaticSpace(dsl, sigma)
    search_space.set_seed(seed)
    task = BoxobanTask({"ascii": ascii_str, "max_calls": max_calls,
                        "crashable": False,
                        "w_on_target": w_on_target,
                        "w_progress": w_progress}, seed=seed)

    best_ind, best_prog = search_space.initialize_individual()
    best_reward = task.evaluate_program(best_prog)
    global_best = best_reward

    while task.program_num < budget and global_best < 1.0:
        candidates = search_space.get_neighbors(best_ind, k=k)
        improved = False
        for ind, prog in candidates:
            r = task.evaluate_program(prog)
            if r > best_reward:
                best_ind, best_prog, best_reward = ind, prog, r
                global_best = max(global_best, r)
                improved = True
                break
            if task.program_num >= budget:
                break
        if not improved:
            # Local maximum -> random restart, keeping the global best.
            best_ind, best_prog = search_space.initialize_individual()
            best_reward = task.evaluate_program(best_prog)
            global_best = max(global_best, best_reward)
    return float(global_best)


def split_half_reliability(per_seed: dict[str, list[float]], n_partitions: int,
                           rng: np.random.RandomState) -> float:
    """Spearman-Brown corrected split-half reliability over per-seed rewards."""
    from scipy.stats import spearmanr

    level_ids = list(per_seed.keys())
    R = len(next(iter(per_seed.values())))
    if R < 2:
        return float("nan")
    half = R // 2
    rs = []
    for _ in range(n_partitions):
        perm = rng.permutation(R)
        a_idx, b_idx = perm[:half], perm[half:2 * half]
        a = [float(np.mean([per_seed[lid][i] for i in a_idx])) for lid in level_ids]
        b = [float(np.mean([per_seed[lid][i] for i in b_idx])) for lid in level_ids]
        rho, _ = spearmanr(a, b)
        if not np.isnan(rho):
            rs.append(rho)
    if not rs:
        return float("nan")
    r = float(np.mean(rs))
    return (2 * r) / (1 + r) if (1 + r) != 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(_ROOT / "data/sokoban_fastpath_labels.csv"),
                    help="CSV listing the levels to label (uses level_id,tier).")
    ap.add_argument("--num-seeds", type=int, default=12)
    ap.add_argument("--budget", type=int, default=4000,
                    help="Program-evaluation budget per seed.")
    ap.add_argument("--k", type=int, default=250, help="HC neighbours per step.")
    ap.add_argument("--sigma", type=float, default=0.25)
    ap.add_argument("--max-calls", type=int, default=2000,
                    help="Max env steps per program rollout.")
    ap.add_argument("--w-on-target", type=float, default=0.8,
                    help="Shaping weight on fraction of boxes actually on target.")
    ap.add_argument("--w-progress", type=float, default=0.2,
                    help="Shaping weight on the matching-distance progress term.")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, only label the first N levels (pilot).")
    ap.add_argument("--shard", type=int, default=0,
                    help="Shard index for parallel array jobs (0-based).")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="Total number of shards; level i goes to shard i %% num_shards.")
    ap.add_argument("--n-partitions", type=int, default=20)
    ap.add_argument("--out", default=str(_ROOT / "data/sokoban_hc_labels"),
                    help="Output prefix (writes <prefix>_labels.csv and _perseed.json).")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    level_index = load_level_index(Path(args.labels))
    if args.limit > 0:
        level_index = level_index[:args.limit]
    if args.num_shards > 1:
        level_index = [lv for i, lv in enumerate(level_index)
                       if i % args.num_shards == args.shard]
        print(f"[shard {args.shard}/{args.num_shards}] "
              f"labelling {len(level_index)} levels", flush=True)
    ascii_cache = build_ascii_cache(level_index)

    dsl = SokobanDSL()
    per_seed: dict[str, list[float]] = {}
    rows = []
    t0 = time.time()
    n = len(level_index)
    for idx, (level_id, tier) in enumerate(level_index):
        ascii_str = ascii_cache.get(level_id)
        if ascii_str is None:
            print(f"[warn] missing ascii for {level_id}, skipping", flush=True)
            continue
        seed_rewards = []
        for s in range(args.num_seeds):
            r = run_hc_seed(ascii_str, dsl, args.sigma, seed=s,
                            budget=args.budget, k=args.k, max_calls=args.max_calls,
                            w_on_target=args.w_on_target, w_progress=args.w_progress)
            seed_rewards.append(r)
        per_seed[level_id] = seed_rewards
        mean_reward = float(np.mean(seed_rewards))
        difficulty = 1.0 - mean_reward
        solved_frac = float(np.mean([1.0 if x >= 1.0 else 0.0 for x in seed_rewards]))
        rows.append({
            "level_id": level_id, "tier": tier,
            "difficulty": difficulty, "mean_reward": mean_reward,
            "solved_frac": solved_frac,
        })
        if not args.no_progress:
            elapsed = time.time() - t0
            print(f"[{idx + 1}/{n}] {level_id} D={difficulty:.3f} "
                  f"mean={mean_reward:.3f} solved={solved_frac:.2f} "
                  f"({elapsed:.0f}s)", flush=True)

    # --- reliability + summary ---
    rng = np.random.RandomState(0)
    rho_rel = split_half_reliability(per_seed, args.n_partitions, rng)
    diffs = np.array([r["difficulty"] for r in rows], dtype=float)

    out_prefix = args.out
    if args.num_shards > 1:
        out_prefix = f"{args.out}_shard{args.shard}of{args.num_shards}"
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    labels_path = out_prefix + "_labels.csv"
    perseed_path = out_prefix + "_perseed.json"
    with open(labels_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["level_id", "tier", "difficulty",
                                           "mean_reward", "solved_frac"])
        w.writeheader()
        w.writerows(rows)
    with open(perseed_path, "w") as fh:
        json.dump({"config": vars(args), "per_seed": per_seed,
                   "rho_reliability": rho_rel}, fh, indent=2)

    print("\n=== Sokoban DSL+HC label summary ===", flush=True)
    print(f"levels labelled : {len(rows)}", flush=True)
    print(f"difficulty mean : {diffs.mean():.3f}  var: {diffs.var():.4f}  "
          f"range: [{diffs.min():.3f}, {diffs.max():.3f}]", flush=True)
    print(f"solved fraction : {np.mean([r['solved_frac'] for r in rows]):.3f}", flush=True)
    print(f"rho_reliability : {rho_rel:.3f}", flush=True)
    print(f"wrote: {labels_path}", flush=True)
    print(f"wrote: {perseed_path}", flush=True)


if __name__ == "__main__":
    main()
