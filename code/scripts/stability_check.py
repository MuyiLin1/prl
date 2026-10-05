#!/usr/bin/env python3
"""Difficulty stability check (Direction B gate).

Question: is HC difficulty a stable property of the (task, grid), or just HC
random noise? We fix the GRID (env seed) and vary only the HC SEARCH seed,
then measure how reproducible difficulty is.

For each task: take the first N_GRIDS env seeds, run HC R times each with
different search seeds (same grid every time), and report:
  - mean within-grid std of difficulty (how much it wobbles per grid)
  - pairwise Spearman of difficulty between search-seed runs (reproducibility)

Interpretation:
  self-rho >= ~0.6 -> difficulty is stable -> "geometry-blind" is a real finding
  self-rho low      -> ground truth is noisy -> must average over HC seeds
"""
import sys
import os
import numpy as np
from itertools import combinations
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'leaps'))

from prog_policies.utils import get_env_name  # noqa: F401 - resolves circular import
from prog_policies.karel_tasks import get_task_cls
from prog_policies.karel import KarelDSL
from prog_policies.search_space import ProgrammaticSpace
from prog_policies.search_methods import HillClimbing

# Reuse env args from the generation script
from generate_multiseed_dataset import get_env_args

TASKS = ["Seeder", "Harvester", "Snake", "TopOff"]
N_GRIDS = 60        # env seeds per task (fixed grids)
N_SEARCH = 3        # HC search seeds per grid
MAX_PROGRAMS = 5000


def run_hc(task_name, env_seed, search_seed, max_programs=MAX_PROGRAMS):
    """Run HC on a fixed grid (env_seed) with a given HC search seed."""
    dsl = KarelDSL()
    task_cls = get_task_cls(task_name)
    env_args = get_env_args(task_name)

    task_envs = [task_cls(env_args, env_seed)]   # GRID fixed by env_seed

    search_space = ProgrammaticSpace(dsl, sigma=0.1)
    search_space.set_seed(search_seed)           # only the SEARCH varies

    search_method = HillClimbing(k=250, e=2)
    _, rewards = search_method.search(
        search_space, task_envs, seed=search_seed, n_iterations=max_programs
    )
    best_reward = max(rewards) if rewards else 0.0
    return 1.0 - best_reward  # difficulty


print(f"Stability check: {N_GRIDS} grids x {N_SEARCH} HC seeds per task\n")

overall = []
for task in TASKS:
    # diff[grid_idx][search_idx]
    diff = np.zeros((N_GRIDS, N_SEARCH))
    for g in range(N_GRIDS):
        for s in range(N_SEARCH):
            search_seed = 10_000 * (s + 1) + g  # distinct, deterministic
            diff[g, s] = run_hc(task, env_seed=g, search_seed=search_seed)

    within_std = diff.std(axis=1)               # per-grid wobble
    mean_within_std = float(within_std.mean())

    # pairwise reproducibility across the N_SEARCH columns
    rhos = []
    for a, b in combinations(range(N_SEARCH), 2):
        if diff[:, a].std() > 1e-9 and diff[:, b].std() > 1e-9:
            r, _ = spearmanr(diff[:, a], diff[:, b])
            if not np.isnan(r):
                rhos.append(r)
    self_rho = float(np.mean(rhos)) if rhos else float("nan")

    mean_diff = diff.mean(axis=1)
    print(f"{task:<12} self_rho={self_rho:>6.3f}  "
          f"mean_within_std={mean_within_std:.3f}  "
          f"difficulty_range=[{mean_diff.min():.2f},{mean_diff.max():.2f}]  "
          f"std_across_grids={mean_diff.std():.3f}")
    overall.append(self_rho)

valid = [r for r in overall if not np.isnan(r)]
mean_self = float(np.mean(valid)) if valid else float("nan")
print(f"\nMean self_rho across tasks: {mean_self:.3f}")
if mean_self >= 0.6:
    print("VERDICT: STABLE -> difficulty is a real (task,grid) property; "
          "geometry-blind finding holds. Proceed to Direction C.")
elif mean_self >= 0.3:
    print("VERDICT: PARTIALLY STABLE -> averaging over a few HC seeds advisable "
          "before predictor claims.")
else:
    print("VERDICT: UNSTABLE -> ground truth is HC noise; must average difficulty "
          "over multiple HC seeds before any predictor conclusion.")
