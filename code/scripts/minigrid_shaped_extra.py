#!/usr/bin/env python3
"""Shaped-reward difficulty for the two remaining MiniGrid tasks: PutNear and
RedBlueDoor.

MiniGrid rewards are sparse / staged (pickup +0.5, drop +0.5, etc.), so raw HC
sees a flat landscape and terminates immediately (difficulty collapses to 1.0
for every grid). Mirroring the LavaGap treatment (scripts/minigrid_shaped.py),
we add a graded potential so HC has a gradient to climb:

  PutNear   : distance-to-ball shaping (phase 1) then distance-of-carried-ball
              -to-target shaping (phase 2), with subgoal credit for a correct
              pickup. shaped in [0,1]; 1.0 on a successful drop near the target.
  RedBlueDoor: subgoal-progress shaping. Stage 1 = approach + open the red door
              (0 -> 0.5), stage 2 = approach + open the blue door after the red
              (0.5 -> 1.0). No natural distance metric between the goals, so the
              within-stage gradient is BFS distance to the current target door.

We then run the same Direction-B pipeline as the other domains: per-grid
difficulty = 1 - mean_R(shaped_best), split-half Spearman-Brown reliability, and
geometry RF 5-fold CV rho (does structure predict the stabilized label?).
"""
import sys
import os
import argparse
from collections import deque

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'leaps'))

from prog_policies.utils import get_env_name  # noqa: F401 - resolves circular import
from prog_policies.minigrid.dsl import MinigridDSL
from prog_policies.minigrid.wrapper import ProgramWrapper
from prog_policies.minigrid_tasks.putnear import PutNearEnv_modified
from prog_policies.minigrid_tasks.redbluedoor import RedBlueDoorEnv_modified
from prog_policies.search_space import MinigridProgrammaticSpace
from prog_policies.search_methods import HillClimbing


# --------------------------------------------------------------------------- #
# BFS distance helpers (walls-only passability; objects/doors do not block the
# potential field, which keeps the shaping cheap and dense).
# --------------------------------------------------------------------------- #
def _wall_blocks(cell):
    return cell is not None and cell.type == "wall"


def multi_source_bfs(grid, sources, width, height):
    """BFS distance from the nearest source cell to every passable cell."""
    dist = {}
    q = deque()
    for (sx, sy) in sources:
        if 0 <= sx < width and 0 <= sy < height and not _wall_blocks(grid.get(sx, sy)):
            dist[(sx, sy)] = 0
            q.append((sx, sy))
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in dist:
                if not _wall_blocks(grid.get(nx, ny)):
                    dist[(nx, ny)] = dist[(x, y)] + 1
                    q.append((nx, ny))
    return dist


def _adjacent_passable(grid, pos, width, height):
    """Passable cells orthogonally adjacent to a target cell (+ the cell itself
    if passable). These are the cells from which the agent can act on it."""
    x, y = int(pos[0]), int(pos[1])
    cells = []
    for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height and not _wall_blocks(grid.get(nx, ny)):
            cells.append((nx, ny))
    return cells


def _clamp01(v):
    return float(max(0.0, min(1.0, v)))


# --------------------------------------------------------------------------- #
# PutNear: distance + subgoal shaping.
# --------------------------------------------------------------------------- #
class ShapedPutNearWrapper(ProgramWrapper):
    def evaluate_program(self, program, record_video=False, record_dir=None):
        self.reset()
        self.program_num += 1
        u = self.unwrapped
        W, H = u.width, u.height

        move_pos = tuple(int(v) for v in u.move_pos)
        target_pos = tuple(int(v) for v in u.target_pos)
        move_type, move_color = u.move_type, u.moveColor

        field_ball = multi_source_bfs(
            u.grid, _adjacent_passable(u.grid, move_pos, W, H), W, H)
        field_target = multi_source_bfs(
            u.grid, _adjacent_passable(u.grid, target_pos, W, H), W, H)

        start = tuple(int(v) for v in u.agent_pos)
        d_ball_start = field_ball.get(start, None)
        # distance the ball would have to travel to the target (from its cell)
        d_target_start = field_target.get(move_pos, None)
        if d_ball_start is None or d_ball_start == 0:
            d_ball_start = 1
        if d_target_start is None or d_target_start == 0:
            d_target_start = 1

        picked_up = False
        best = 0.0
        for _ in program.run_generator(self):
            pos = tuple(int(v) for v in u.agent_pos)
            carrying = u.carrying
            correct_carry = (
                carrying is not None
                and getattr(carrying, "type", None) == move_type
                and getattr(carrying, "color", None) == move_color
            )
            if correct_carry:
                picked_up = True
            wrong_carry = carrying is not None and not correct_carry

            if not picked_up:
                d = field_ball.get(pos, d_ball_start)
                approach = _clamp01((d_ball_start - d) / d_ball_start)
                val = 0.45 * approach
            else:
                d = field_target.get(pos, d_target_start)
                approach = _clamp01((d_target_start - d) / d_target_start)
                val = 0.5 + 0.5 * approach
            best = max(best, val)

            terminated, _ = self.get_reward()
            if wrong_carry:
                break
            if terminated:
                # successful drop near target (only positive terminal for us)
                if picked_up:
                    best = max(best, 1.0)
                break
        return best


def make_shaped_putnear(env_seed, size=6, max_calls=1000):
    env = PutNearEnv_modified(size=size)
    return ShapedPutNearWrapper(env, env_seed, False, 0.0, max_calls)


def putnear_features(env_seed, size):
    env = PutNearEnv_modified(size=size)
    env.reset(seed=env_seed)
    u = env
    W, H = u.width, u.height
    start = (int(u.agent_pos[0]), int(u.agent_pos[1]))
    ball = (int(u.move_pos[0]), int(u.move_pos[1]))
    target = (int(u.target_pos[0]), int(u.target_pos[1]))

    field_ball = multi_source_bfs(u.grid, _adjacent_passable(u.grid, ball, W, H), W, H)
    field_target = multi_source_bfs(u.grid, _adjacent_passable(u.grid, target, W, H), W, H)

    d_agent_ball = field_ball.get(start, -1)
    d_ball_target = field_target.get(ball, -1)
    man_agent_ball = abs(start[0] - ball[0]) + abs(start[1] - ball[1])
    man_ball_target = abs(ball[0] - target[0]) + abs(ball[1] - target[1])

    return {
        "agent_x": start[0], "agent_y": start[1],
        "ball_x": ball[0], "ball_y": ball[1],
        "target_x": target[0], "target_y": target[1],
        "d_agent_ball": d_agent_ball,
        "d_ball_target": d_ball_target,
        "total_path": (d_agent_ball + d_ball_target)
        if (d_agent_ball >= 0 and d_ball_target >= 0) else -1,
        "man_agent_ball": man_agent_ball,
        "man_ball_target": man_ball_target,
        "size": size,
    }


# --------------------------------------------------------------------------- #
# RedBlueDoor: subgoal-progress shaping (red then blue), distance gradient
# within each stage.
# --------------------------------------------------------------------------- #
class ShapedRedBlueDoorWrapper(ProgramWrapper):
    def evaluate_program(self, program, record_video=False, record_dir=None):
        self.reset()
        self.program_num += 1
        u = self.unwrapped
        W, H = u.width, u.height

        red_cell = None
        blue_cell = None
        for x in range(W):
            for y in range(H):
                c = u.grid.get(x, y)
                if c is not None and getattr(c, "type", None) == "door":
                    if c.color == "red":
                        red_cell = (x, y)
                    elif c.color == "blue":
                        blue_cell = (x, y)

        field_red = multi_source_bfs(
            u.grid, _adjacent_passable(u.grid, red_cell, W, H), W, H) if red_cell else {}
        field_blue = multi_source_bfs(
            u.grid, _adjacent_passable(u.grid, blue_cell, W, H), W, H) if blue_cell else {}

        start = tuple(int(v) for v in u.agent_pos)
        d_red_start = field_red.get(start, None)
        if d_red_start is None or d_red_start == 0:
            d_red_start = 1
        d_blue_start = None  # fixed when the red door opens

        best = 0.0
        for _ in program.run_generator(self):
            pos = tuple(int(v) for v in u.agent_pos)
            red_open = u.red_door.is_open
            blue_open = u.blue_door.is_open

            if not red_open:
                d = field_red.get(pos, d_red_start)
                approach = _clamp01((d_red_start - d) / d_red_start)
                val = 0.4 * approach
            else:
                if d_blue_start is None:
                    d0 = field_blue.get(pos, None)
                    d_blue_start = d0 if (d0 is not None and d0 > 0) else 1
                d = field_blue.get(pos, d_blue_start)
                approach = _clamp01((d_blue_start - d) / d_blue_start)
                val = 0.5 + 0.4 * approach
            best = max(best, val)

            terminated, _ = self.get_reward()
            if blue_open:
                # terminal: 1.0 only if opened in the correct order (red first)
                if red_open:
                    best = max(best, 1.0)
                break
            if terminated:
                break
        return best


def make_shaped_redbluedoor(env_seed, size=6, max_calls=1000):
    env = RedBlueDoorEnv_modified(size=size)
    return ShapedRedBlueDoorWrapper(env, env_seed, False, 0.0, max_calls)


def redbluedoor_features(env_seed, size):
    env = RedBlueDoorEnv_modified(size=size)
    env.reset(seed=env_seed)
    u = env
    W, H = u.width, u.height
    start = (int(u.agent_pos[0]), int(u.agent_pos[1]))

    red_cell = blue_cell = None
    for x in range(W):
        for y in range(H):
            c = u.grid.get(x, y)
            if c is not None and getattr(c, "type", None) == "door":
                if c.color == "red":
                    red_cell = (x, y)
                elif c.color == "blue":
                    blue_cell = (x, y)

    field_red = multi_source_bfs(u.grid, _adjacent_passable(u.grid, red_cell, W, H), W, H)
    field_blue = multi_source_bfs(u.grid, _adjacent_passable(u.grid, blue_cell, W, H), W, H)
    d_agent_red = field_red.get(start, -1)
    d_red_blue = field_blue.get(red_cell, -1)

    return {
        "agent_x": start[0], "agent_y": start[1],
        "red_y": red_cell[1] if red_cell else -1,
        "blue_y": blue_cell[1] if blue_cell else -1,
        "door_row_gap": abs((red_cell[1] if red_cell else 0)
                            - (blue_cell[1] if blue_cell else 0)),
        "d_agent_red": d_agent_red,
        "d_red_blue": d_red_blue,
        "total_path": (d_agent_red + d_red_blue)
        if (d_agent_red >= 0 and d_red_blue >= 0) else -1,
        "size": size,
    }


# --------------------------------------------------------------------------- #
# Shared HC + analysis pipeline.
# --------------------------------------------------------------------------- #
TASKS = {
    "PutNear": (make_shaped_putnear, putnear_features),
    "RedBlueDoor": (make_shaped_redbluedoor, redbluedoor_features),
}


def run_hc(make_env, env_seed, search_seed, n_iterations, size):
    dsl = MinigridDSL()
    task_envs = [make_env(env_seed, size=size)]
    search_space = MinigridProgrammaticSpace(dsl, sigma=0.25)
    search_space.set_seed(search_seed)
    hc = HillClimbing(k=250, e=2)
    _, rewards = hc.search(
        search_space, task_envs, seed=search_seed, n_iterations=n_iterations)
    return max(rewards) if rewards else 0.0


def split_half_reliability(mat):
    g, r = mat.shape
    if r < 2:
        return float("nan")
    cols = np.arange(r)
    rng = np.random.default_rng(0)
    rhos = []
    for _ in range(20):
        perm = rng.permutation(cols)
        a, b = perm[: r // 2], perm[r // 2:]
        ma, mb = mat[:, a].mean(axis=1), mat[:, b].mean(axis=1)
        if ma.std() > 1e-9 and mb.std() > 1e-9:
            rho, _ = spearmanr(ma, mb)
            if not np.isnan(rho):
                rhos.append(2 * rho / (1 + rho))
    return float(np.mean(rhos)) if rhos else float("nan")


def rf_cv(fdf, y):
    use = [c for c in fdf.columns if fdf[c].std() > 1e-9]
    if len(use) == 0 or np.std(y) < 1e-9:
        return float("nan"), use
    X = fdf[use].values
    n_splits = min(5, len(y))
    if n_splits < 2:
        return float("nan"), use
    oof = np.zeros(len(y))
    for tr, te in KFold(n_splits=n_splits, shuffle=True, random_state=0).split(X):
        rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0)
        rf.fit(X[tr], y[tr])
        oof[te] = rf.predict(X[te])
    rho, _ = spearmanr(oof, y)
    return float(rho), use


def best_single(fdf, y):
    bf, br = None, 0.0
    for c in fdf.columns:
        if fdf[c].std() < 1e-9:
            continue
        rho, _ = spearmanr(fdf[c].values, y)
        if not np.isnan(rho) and abs(rho) > abs(br):
            br, bf = rho, c
    return bf, br


def _summary_row(task, shaped, fdf):
    """Compute the one-row summary (reliability + geometry CV) for a task."""
    diff = 1.0 - shaped.mean(axis=1)
    rel = split_half_reliability(shaped)
    bf, br = best_single(fdf, diff)
    rho, _ = rf_cv(fdf, diff)
    return {
        "task": task,
        "across_grid_std": float(diff.std()),
        "within_grid_std": float(shaped.std(axis=1).mean()),
        "reliability": rel,
        "best_feature": bf,
        "best_single_|rho|": br,
        "rf_cv_rho": rho,
    }


def merge_raw(shard_paths, task, output):
    """Reassemble per-grid raw shards into the full (G x R) matrix and write the
    task summary. Raw shards have columns: grid_seed, shaped_0..shaped_{R-1},
    <feature columns>. Grids are de-duplicated and sorted by grid_seed so the
    label/reliability/geometry stats are computed over the complete grid set."""
    frames = [pd.read_csv(p) for p in shard_paths]
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.drop_duplicates(subset="grid_seed").sort_values("grid_seed")
    shaped_cols = sorted(
        [c for c in raw.columns if c.startswith("shaped_")],
        key=lambda c: int(c.split("_")[1]))
    shaped = raw[shaped_cols].to_numpy()
    feat_cols = [c for c in raw.columns
                 if c != "grid_seed" and not c.startswith("shaped_")]
    fdf = raw[feat_cols].reset_index(drop=True)
    row = _summary_row(task, shaped, fdf)
    summary = pd.DataFrame([row])
    summary.to_csv(output, index=False)
    print(f"merged {len(shard_paths)} shards -> {len(raw)} grids")
    print(summary.to_string(index=False))
    print(f"wrote {output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(TASKS.keys()))
    ap.add_argument("--grids", type=int, default=60)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--size", type=int, default=6)
    ap.add_argument("--n-iter", type=int, default=2000)
    ap.add_argument("--output", default="minigrid_shaped_extra.csv")
    # Grid sharding: process only [grid-start, grid-end) and dump the raw
    # per-grid per-seed matrix so many workers can split one task across cores.
    ap.add_argument("--grid-start", type=int, default=0)
    ap.add_argument("--grid-end", type=int, default=None)
    ap.add_argument("--raw-output", default=None,
                    help="write per-grid raw shard (skips inline summary)")
    # Merge mode: combine raw shards from one task into a summary and exit.
    ap.add_argument("--merge-raw", nargs="+", default=None,
                    help="raw shard CSVs to merge into a summary")
    ap.add_argument("--merge-task", default=None,
                    help="task name for the merged summary row")
    args = ap.parse_args()

    if args.merge_raw:
        merge_raw(args.merge_raw, args.merge_task, args.output)
        return

    g_start = args.grid_start
    g_end = args.grid_end if args.grid_end is not None else args.grids

    rows = []
    for task in args.tasks:
        make_env, feat_fn = TASKS[task]
        grid_ids = list(range(g_start, g_end))
        print("=" * 70)
        print(f"Task: {task}  (grids [{g_start},{g_end}) x {args.reps} HC "
              f"seeds, size={args.size})")
        print("=" * 70, flush=True)

        feats = []
        shaped = np.zeros((len(grid_ids), args.reps))
        for i, g in enumerate(grid_ids):
            feats.append(feat_fn(g, args.size))
            for s in range(args.reps):
                ss = 100_000 * (s + 1) + g
                shaped[i, s] = run_hc(make_env, g, ss, args.n_iter, args.size)
            print(f"  grid {g:>3}: mean_shaped={shaped[i].mean():.3f} "
                  f"std={shaped[i].std():.3f}", flush=True)

        fdf = pd.DataFrame(feats)

        if args.raw_output:
            raw = fdf.copy()
            raw.insert(0, "grid_seed", grid_ids)
            for s in range(args.reps):
                raw[f"shaped_{s}"] = shaped[:, s]
            raw.to_csv(args.raw_output, index=False)
            print(f"wrote raw shard {args.raw_output} "
                  f"({len(grid_ids)} grids)", flush=True)
            continue

        rows.append(_summary_row(task, shaped, fdf))
        out = fdf.copy()
        out.insert(0, "grid_seed", grid_ids)
        out["difficulty"] = 1.0 - shaped.mean(axis=1)
        out.to_csv(f"/tmp/minigrid_{task}.csv", index=False)

    if not rows:
        return
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output, index=False)
    print("\n" + "=" * 78)
    print("MINIGRID SHAPED EXTRA — SUMMARY")
    print("=" * 78)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
