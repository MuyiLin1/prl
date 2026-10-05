#!/usr/bin/env python3
"""
Run Hill Climbing on all Karel tasks and collect ground-truth difficulty data.

For each task, runs HC with multiple seeds (default 8) and a fixed program budget
(default 100k). Records the best reward per seed, then computes mean/median
best reward across seeds as the ground-truth "solvability" signal.

Output: ground_truth_difficulty.csv with columns:
  task_name, mean_best_reward, median_best_reward, solve_rate, difficulty

Usage:
  conda activate llm_gs
  python scripts/run_all_hc.py --num_seeds 8 --max_program_nums 100000
  python scripts/run_all_hc.py --num_seeds 4 --max_program_nums 50000  # faster
"""

import sys
import os
import time
import json
import csv
import pathlib
from argparse import ArgumentParser

sys.path.append(".")
sys.path.append("./leaps")

from prog_policies.utils import get_env_name
from prog_policies.karel import KarelDSL
from prog_policies.karel_tasks import get_task_cls as get_karel_task_cls
from prog_policies.search_space import get_search_space_cls
from prog_policies.search_methods import get_search_method_cls

KAREL_TASKS = [
    "StairClimberSparse",
    "MazeSparse",
    "FourCorners",
    "TopOff",
    "Harvester",
    "CleanHouse",
    "DoorKey",
    "OneStroke",
    "Seeder",
    "Snake",
    "WallAvoider",
    "PathFollow",
]


def make_karel_envs(task_name, num_envs=32):
    dsl = KarelDSL()
    env_args = {
        "env_height": 8,
        "env_width": 8,
        "crashable": False,
        "leaps_behaviour": True,
        "max_calls": 10000,
    }
    if task_name in ("StairClimber", "StairClimberSparse", "TopOff", "FourCorners"):
        env_args["env_height"] = 12
        env_args["env_width"] = 12
    if task_name == "CleanHouse":
        env_args["env_height"] = 14
        env_args["env_width"] = 22
    if task_name == "WallAvoider":
        env_args["env_height"] = 8
        env_args["env_width"] = 5

    task_cls = get_karel_task_cls(task_name)
    task_envs = [task_cls(env_args, i) for i in range(num_envs)]
    return task_envs, dsl


def run_hc_single(task_name, seed, max_program_nums, k=250):
    """Run HC for one task+seed. Returns best_reward found."""
    task_envs, dsl = make_karel_envs(task_name)

    search_space_cls = get_search_space_cls("ProgrammaticSpace")
    search_space = search_space_cls(dsl, 0.1)
    search_space.set_seed(seed)

    search_method_cls = get_search_method_cls("HillClimbing")
    search_method = search_method_cls(k, 2)

    best_reward = -float("inf")
    best_prog = None

    while task_envs[0].program_num < max_program_nums:
        # Generate candidates and evaluate
        candidates = search_space.get_candidates(search_method, best_prog)
        for prog in candidates:
            rewards = []
            for env in task_envs:
                env.evaluate_program(prog, dsl)
                rewards.append(env.get_reward_value())
            mean_reward = sum(rewards) / len(rewards)
            if mean_reward > best_reward:
                best_reward = mean_reward
                best_prog = prog

        if best_reward >= 1.0:
            break

    return best_reward, task_envs[0].program_num


def run_hc_single_via_baseline(task_name, seed, max_program_nums, k=250):
    """
    Run HC using the same logic as scripts/baseline.py (record_search).
    This ensures results match the original codebase exactly.
    """
    from prog_policies.utils.evaluate_and_search import record_search

    task_envs, dsl = make_karel_envs(task_name)

    search_space_cls = get_search_space_cls("ProgrammaticSpace")
    search_space = search_space_cls(dsl, 0.1)
    search_space.set_seed(seed)

    search_method_cls = get_search_method_cls("HillClimbing")
    search_method = search_method_cls(k, 2)

    best_reward = -float("inf")
    best_prog = None

    # Dummy output dir (we won't save intermediate files)
    output_dir_seed = f"/tmp/hc_run/{task_name}/{seed}"
    pathlib.Path(output_dir_seed).mkdir(parents=True, exist_ok=True)
    output_dir = f"/tmp/hc_run/{task_name}"
    log = {"args": {}, "seed": seed}
    init_time = time.time()

    while task_envs[0].program_num < max_program_nums and best_reward < 1.0:
        best_prog, best_reward, _ = record_search(
            best_prog, best_reward, search_method, search_space,
            task_envs, dsl, output_dir_seed, log, init_time, output_dir, task_name, seed
        )

    return best_reward, task_envs[0].program_num


def main():
    parser = ArgumentParser()
    parser.add_argument("--num_seeds", type=int, default=8)
    parser.add_argument("--max_program_nums", type=int, default=100000)
    parser.add_argument("--k", type=int, default=250)
    parser.add_argument("--tasks", nargs="+", default=KAREL_TASKS)
    parser.add_argument("--output", default="ground_truth_difficulty.csv")
    args = parser.parse_args()

    results = []

    for task in args.tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}")

        seed_rewards = []
        seed_programs = []
        for seed in range(args.num_seeds):
            t0 = time.time()
            best_reward, num_programs = run_hc_single_via_baseline(
                task, seed, args.max_program_nums, args.k
            )
            elapsed = time.time() - t0
            seed_rewards.append(best_reward)
            seed_programs.append(num_programs)
            print(f"  Seed {seed}: best_reward={best_reward:.4f}, "
                  f"programs={num_programs}, time={elapsed:.1f}s")

        import numpy as np
        mean_reward = float(np.mean(seed_rewards))
        median_reward = float(np.median(seed_rewards))
        solve_rate = float(np.mean([1.0 if r >= 1.0 else 0.0 for r in seed_rewards]))
        difficulty = 1.0 - mean_reward  # higher = harder

        # Also record programs-to-solve as alternate difficulty metric
        # Use log(programs) for solved seeds, max_budget for unsolved
        programs_list = [p for _, p in zip(seed_rewards, seed_programs)]
        mean_programs = float(np.mean(programs_list))
        log_programs = float(np.mean([np.log10(max(1, p)) for p in programs_list]))

        results.append({
            "task_name": task,
            "mean_best_reward": round(mean_reward, 6),
            "median_best_reward": round(median_reward, 6),
            "solve_rate": round(solve_rate, 4),
            "difficulty": round(difficulty, 6),
            "mean_programs": round(mean_programs, 1),
            "log_mean_programs": round(log_programs, 4),
            "num_seeds": args.num_seeds,
            "max_program_nums": args.max_program_nums,
            "seed_rewards": seed_rewards,
            "seed_programs": seed_programs,
        })

        print(f"  => mean={mean_reward:.4f}, solve_rate={solve_rate:.2f}, "
              f"difficulty={difficulty:.4f}")

    # Write CSV
    fieldnames = ["task_name", "mean_best_reward", "median_best_reward",
                  "solve_rate", "difficulty", "mean_programs", "log_mean_programs"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})

    # Also write full JSON for detailed analysis
    json_output = args.output.replace(".csv", ".json")
    with open(json_output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote {args.output} and {json_output}")
    print("\nDifficulty ranking (hardest first):")
    for r in sorted(results, key=lambda x: x["difficulty"], reverse=True):
        print(f"  {r['task_name']:20s}  difficulty={r['difficulty']:.4f}  "
              f"solve_rate={r['solve_rate']:.2f}")


if __name__ == "__main__":
    main()
