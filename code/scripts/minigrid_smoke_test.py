#!/usr/bin/env python3
"""Direction C smoke test: verify the HC pipeline runs on MiniGrid tasks.

Mirrors the Karel run_hc_single, but uses the MiniGrid API:
  - dsl = MinigridDSL()
  - task_cls = get_task_cls(name); env = task_cls(seed, crashable, crash_penalty, max_calls)
  - search_space = MinigridProgrammaticSpace(dsl, sigma=0.25)
  - HillClimbing(k=250, e=2)

Runs a few seeds per task with a small budget to confirm rewards come back and vary.
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'leaps'))

from prog_policies.utils import get_env_name  # noqa: F401 - resolves circular import
from prog_policies.minigrid_tasks import get_task_cls, TASK_NAME_LIST
from prog_policies.minigrid.dsl import MinigridDSL
from prog_policies.search_space import MinigridProgrammaticSpace
from prog_policies.search_methods import HillClimbing

CRASHABLE = False
CRASH_PENALTY = 0.0
MAX_CALLS = 1000


def run_hc_minigrid(task_name, env_seed, search_seed, n_iterations=200):
    dsl = MinigridDSL()
    task_cls = get_task_cls(task_name)
    # MiniGrid task constructor: (seed, crashable, crash_penalty, max_calls)
    task_envs = [task_cls(env_seed, CRASHABLE, CRASH_PENALTY, MAX_CALLS)]
    search_space = MinigridProgrammaticSpace(dsl, sigma=0.25)
    search_space.set_seed(search_seed)
    search_method = HillClimbing(k=250, e=2)
    _, rewards = search_method.search(
        search_space, task_envs, seed=search_seed, n_iterations=n_iterations
    )
    return max(rewards) if rewards else 0.0


if __name__ == "__main__":
    print(f"MiniGrid tasks: {TASK_NAME_LIST}\n")
    for task in TASK_NAME_LIST:
        print(f"=== {task} ===")
        for seed in range(3):
            t0 = time.time()
            r = run_hc_minigrid(task, env_seed=seed, search_seed=1000 + seed, n_iterations=200)
            print(f"  seed {seed}: best_reward={r:.4f} difficulty={1.0 - r:.4f} "
                  f"time={time.time() - t0:.1f}s")
    print("\nSmoke test complete.")
