#!/usr/bin/env python3
"""Direction C / Option A: shaped-reward MiniGrid difficulty.

MiniGrid rewards are sparse (1 at goal, -1 lava, 0 else), so HC sees a flat
landscape and terminates immediately. Here we add DISTANCE-TO-GOAL shaping so HC
has a gradient: a program is rewarded for how close to the goal the agent gets.

shaped_reward = (d_start - d_min_reached) / d_start
  = 0 if the agent never gets closer, 1 if it reaches the goal.

This makes LavaGap a clean geometry test: difficulty should depend on the gap
position in the lava wall (where the single opening is). We then test whether
geometric features predict the shaped difficulty.

Focus task: LavaGap (canonical "structure gates progress").
"""
import sys
import os
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'leaps'))

from prog_policies.utils import get_env_name  # noqa: F401
from prog_policies.minigrid.dsl import MinigridDSL
from prog_policies.minigrid.wrapper import ProgramWrapper
from prog_policies.minigrid_tasks.lavagap import LavaGapEnv_modified
from prog_policies.search_space import MinigridProgrammaticSpace
from prog_policies.search_methods import HillClimbing


def _passable(cell):
    """A cell is passable if empty, goal, or anything the agent can overlap.
    Walls and lava are impassable for the distance potential."""
    if cell is None:
        return True
    t = cell.type
    if t in ("wall", "lava"):
        return False
    return True


def bfs_dist_to_goal(grid, goal_pos, width, height):
    """BFS distances (in steps) from every passable cell to the goal."""
    gx, gy = int(goal_pos[0]), int(goal_pos[1])
    dist = {(gx, gy): 0}
    q = deque([(gx, gy)])
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in dist:
                if _passable(grid.get(nx, ny)):
                    dist[(nx, ny)] = dist[(x, y)] + 1
                    q.append((nx, ny))
    return dist


class ShapedLavaGapWrapper(ProgramWrapper):
    """ProgramWrapper that returns distance-to-goal shaped reward."""

    def evaluate_program(self, program, record_video=False, record_dir=None):
        self.reset()
        self.program_num += 1
        u = self.unwrapped
        dist = bfs_dist_to_goal(u.grid, u.goal_pos, u.width, u.height)
        start_pos = tuple(int(v) for v in u.agent_pos)
        d_start = dist.get(start_pos, None)
        if d_start is None or d_start == 0:
            return 1.0  # already at goal / degenerate
        d_min = d_start
        reached_goal = False
        for _ in program.run_generator(self):
            pos = tuple(int(v) for v in u.agent_pos)
            # If agent stepped onto lava the episode terminates; pos is lava cell
            cell = u.grid.get(*pos)
            if cell is not None and cell.type == "lava":
                break
            d = dist.get(pos, None)
            if d is not None:
                d_min = min(d_min, d)
                if d == 0:
                    reached_goal = True
                    break
            terminated, _ = self.get_reward()
            if terminated:
                break
        if reached_goal:
            return 1.0
        return float((d_start - d_min) / d_start)


def make_shaped_lavagap(env_seed, size=6, max_calls=1000):
    env = LavaGapEnv_modified(size=size)
    return ShapedLavaGapWrapper(env, env_seed, False, 0.0, max_calls)


def run_hc_shaped_lavagap(env_seed, search_seed, n_iterations=2000, size=6):
    dsl = MinigridDSL()
    task_envs = [make_shaped_lavagap(env_seed, size=size)]
    search_space = MinigridProgrammaticSpace(dsl, sigma=0.25)
    search_space.set_seed(search_seed)
    search_method = HillClimbing(k=250, e=2)
    _, rewards = search_method.search(
        search_space, task_envs, seed=search_seed, n_iterations=n_iterations
    )
    return max(rewards) if rewards else 0.0


if __name__ == "__main__":
    import time
    print("Shaped LavaGap HC smoke test (does shaping give a gradient?)\n")
    for seed in range(6):
        t0 = time.time()
        r = run_hc_shaped_lavagap(seed, search_seed=1000 + seed, n_iterations=2000)
        print(f"  env_seed {seed}: shaped_best={r:.3f} difficulty={1 - r:.3f} "
              f"time={time.time() - t0:.1f}s")
