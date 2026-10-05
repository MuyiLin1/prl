"""Option-1 regime probe: bigger LavaCrossing.

Curriculum learning can only help when the HARD tasks are beyond the exploration horizon
(uniform PPO ~never solves them from scratch) while the EASY tasks are still learnable
(so they can act as stepping stones). The first directional run showed that at sizes
{7,9,11} the hardest cell was still ~47% from scratch -> "more of the same skill" -> no
room for curriculum.

This probe trains ONE uniform PPO on a bigger pool and reports per-cell success, to check
whether a size-enriched pool creates that gap:
    EASY end (small grid, 1 river)  should reach  >~0.5   (learnable stepping stone)
    HARD end (big grid, 3 rivers)   should stay   <~0.1   (beyond the exploration horizon)

Verdict:
  * gap present (easy high, hard ~0)   -> regime is right, run the full curriculum comparison.
  * hard end still learnable (>0.3)    -> go bigger (raise sizes).
  * easy end also collapses (<0.3)     -> too hard, back off (smaller sizes / more steps).

NO LLM. Reuses the harness env/model so the probe and the real run are identical.
"""
from __future__ import annotations

import argparse
import time
import warnings
from collections import defaultdict

import numpy as np

from curriculum_harness import CurriculumPoolEnv, eval_layouts, make_ppo
from difficulty_scoring import enumerate_pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[9, 13, 17])
    ap.add_argument("--tiers", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n-seeds", type=int, default=80)
    ap.add_argument("--cap-per-cell", type=int, default=10)
    ap.add_argument("--total-steps", type=int, default=300_000)
    ap.add_argument("--eval-every", type=int, default=75_000)
    ap.add_argument("--eval-layouts-per-cell", type=int, default=6)
    ap.add_argument("--eval-episodes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-restrict-actions", action="store_true")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    restrict = not args.no_restrict_actions

    pool = enumerate_pool(args.sizes, args.tiers, args.n_seeds, args.cap_per_cell)
    by_cell = defaultdict(list)
    for lo in pool:
        by_cell[(lo.size, lo.n)].append(lo)
    cells = sorted(by_cell)
    print(f"[probe] pool={len(pool)} layouts over {len(cells)} cells "
          f"(sizes={args.sizes} tiers={args.tiers})", flush=True)
    for c in cells:
        print(f"        cell size{c[0]} N{c[1]}: {len(by_cell[c])} distinct layouts", flush=True)

    # uniform PPO over the WHOLE pool (this is exactly the "random"/no-curriculum condition)
    env = CurriculumPoolEnv(restrict, rng_seed=args.seed)
    env.set_active(pool)
    model = make_ppo(env, seed=args.seed)

    t0 = time.time()
    done = 0
    ck = 0
    while done < args.total_steps:
        step = min(args.eval_every, args.total_steps - done)
        model.learn(total_timesteps=step, reset_num_timesteps=(ck == 0))
        done += step
        ck += 1
        parts = []
        for c in cells:
            subset = by_cell[c][: args.eval_layouts_per_cell]
            r = float(np.mean(eval_layouts(model, subset, args.eval_episodes, restrict)))
            parts.append(f"s{c[0]}N{c[1]}={r:.2f}")
        print(f"[{done:>7d}] " + "  ".join(parts) + f"   ({time.time() - t0:.0f}s)", flush=True)

    # final verdict helper (easy = smallest size + tier 1; hard = largest size + max tier)
    easy = (min(args.sizes), min(args.tiers))
    hard = (max(args.sizes), max(args.tiers))
    er = float(np.mean(eval_layouts(model, by_cell[easy][: args.eval_layouts_per_cell],
                                    args.eval_episodes, restrict)))
    hr = float(np.mean(eval_layouts(model, by_cell[hard][: args.eval_layouts_per_cell],
                                    args.eval_episodes, restrict)))
    print(f"\n[verdict] easy {easy}={er:.2f}  hard {hard}={hr:.2f}", flush=True)
    if er >= 0.5 and hr <= 0.1:
        print("[verdict] GAP PRESENT -> regime is right; run the full curriculum comparison.", flush=True)
    elif hr > 0.3:
        print("[verdict] hard end still learnable -> go BIGGER (raise --sizes).", flush=True)
    elif er < 0.3:
        print("[verdict] easy end collapsed -> too hard; back off (smaller sizes / more steps).", flush=True)
    else:
        print("[verdict] borderline -> inspect per-cell curve above.", flush=True)


if __name__ == "__main__":
    main()
