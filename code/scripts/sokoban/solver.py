#!/usr/bin/env python3
"""
Budgeted stochastic Sokoban solver with reward shaping --- the fast-path reference
solver for the third-domain difficulty label (paper section 6.5).

This is the Sokoban analogue of the Hill-Climbing reference solver used for
Karel/MiniGrid. Like HC, it runs a *bounded-budget* search from random restarts and
records the *best shaped reward reached* within the budget; on an easy level the
search quickly reaches reward 1 (a solved board), on a hard level it stalls at a
low partial reward. The per-run difficulty is 1 - best_shaped_reward, and the
difficulty label averages this over R search seeds (Eq. difficulty-label).

The search is a randomized weighted-A* / greedy best-first over Sokoban game states
with:
  * shaped reward = box-on-target fraction + normalized progress on the
    box->target min-cost matching potential (the distance-to-goal shaping of
    Eq. shaping, generalized to multi-box Sokoban),
  * corner-deadlock pruning (a box pushed to a non-target corner is unrecoverable),
  * an expansion budget T mirroring the HC evaluation budget, and
  * random restarts + random tie-breaking controlled by the seed.

No DSL and no learned model are involved; this only establishes whether a budgeted
search yields a graded, reliable difficulty label before any DSL/HC is built.
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from sokoban_core import SokobanLevel, hungarian_min_cost, is_dead_corner

Cell = Tuple[int, int]
State = Tuple[Cell, FrozenSet[Cell]]  # (player, boxes)

_MOVES = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


@dataclass
class SolverConfig:
    budget: int = 4000          # max state expansions per run (mirrors HC T)
    n_restarts: int = 8         # random restarts within the budget
    w_target: float = 0.6       # weight on box-on-target fraction
    w_potential: float = 0.4    # weight on matching-distance progress
    greedy_eps: float = 0.25    # prob of random expansion (exploration)


class SokobanModel:
    """Precomputes static level structure: walls, targets, dead cells, and
    all-pairs floor distances used by the shaping potential."""

    def __init__(self, level: SokobanLevel):
        self.walls = level.walls
        self.floor = level.floor
        self.targets = frozenset(level.targets)
        self.num_boxes = level.num_boxes
        self.start_player = level.player
        self.start_boxes = frozenset(level.boxes)
        self.dead_cells = frozenset(
            c for c in level.floor if is_dead_corner(c, level.walls, level.targets)
        )
        self._dist = self._all_pairs_floor_dist()
        self._init_potential = self._matching_potential(self.start_boxes)

    # ---- static geometry ----
    def _all_pairs_floor_dist(self) -> Dict[Cell, Dict[Cell, int]]:
        dist: Dict[Cell, Dict[Cell, int]] = {}
        for src in self.targets:  # only need distances *to* targets
            d = {src: 0}
            frontier = [src]
            while frontier:
                nxt = []
                for u in frontier:
                    uy, ux = u
                    for dy, dx in _MOVES.values():
                        v = (uy + dy, ux + dx)
                        if v in self.floor and v not in d:
                            d[v] = d[u] + 1
                            nxt.append(v)
                frontier = nxt
            dist[src] = d
        return dist

    def _box_target_cost(self, boxes: FrozenSet[Cell]) -> np.ndarray:
        tlist = list(self.targets)
        blist = list(boxes)
        n = max(len(blist), len(tlist))
        cost = np.full((n, n), 1e6, dtype=float)
        for i, b in enumerate(blist):
            for j, t in enumerate(tlist):
                cost[i, j] = self._dist[t].get(b, 1e6)
        return cost

    def _matching_potential(self, boxes: FrozenSet[Cell]) -> float:
        if not boxes or not self.targets:
            return 0.0
        return hungarian_min_cost(self._box_target_cost(boxes))

    # ---- shaped reward ----
    def shaped_reward(self, state: State) -> float:
        _, boxes = state
        on_target = len(boxes & self.targets)
        frac = on_target / max(1, self.num_boxes)
        if on_target == self.num_boxes:
            return 1.0
        if self._init_potential <= 0:
            progress = 0.0
        else:
            pot = self._matching_potential(boxes)
            progress = max(0.0, (self._init_potential - pot) / self._init_potential)
        cfg_w_t, cfg_w_p = 0.6, 0.4
        return min(0.999, cfg_w_t * frac + cfg_w_p * progress)

    def is_solved(self, state: State) -> bool:
        return state[1] <= self.targets

    # ---- transitions ----
    def successors(self, state: State) -> List[State]:
        player, boxes = state
        out: List[State] = []
        py, px = player
        for dy, dx in _MOVES.values():
            np_ = (py + dy, px + dx)
            if np_ in self.walls:
                continue
            if np_ in boxes:
                # push the box
                nb = (np_[0] + dy, np_[1] + dx)
                if nb in self.walls or nb in boxes:
                    continue
                if nb not in self.floor:
                    continue
                if nb in self.dead_cells:
                    continue  # deadlock prune
                new_boxes = frozenset((boxes - {np_}) | {nb})
                out.append((np_, new_boxes))
            else:
                # plain move
                if np_ in self.floor:
                    out.append((np_, boxes))
        return out


def run_search(level: SokobanLevel, seed: int, cfg: SolverConfig) -> float:
    """One budgeted stochastic search run. Returns the best shaped reward reached."""
    model = SokobanModel(level)
    if model.start_player is None or model.num_boxes == 0:
        return 0.0
    rng = random.Random(seed)

    start: State = (model.start_player, model.start_boxes)
    best = model.shaped_reward(start)
    expansions = 0
    restarts = max(1, cfg.n_restarts)
    budget_per_restart = max(1, cfg.budget // restarts)

    for _ in range(restarts):
        # priority queue: (-reward, tie, state); randomized tie-break
        visited = {start}
        counter = 0
        h0 = -model.shaped_reward(start)
        frontier: List[Tuple[float, int, State]] = [(h0, rng.random(), start)]
        local_budget = budget_per_restart
        while frontier and local_budget > 0:
            local_budget -= 1
            expansions += 1
            if rng.random() < cfg.greedy_eps:
                idx = rng.randrange(len(frontier))
                _, _, state = frontier.pop(idx)
            else:
                _, _, state = heapq.heappop(frontier)

            r = model.shaped_reward(state)
            if r > best:
                best = r
            if r >= 1.0 or model.is_solved(state):
                return 1.0
            for ns in model.successors(state):
                if ns in visited:
                    continue
                visited.add(ns)
                counter += 1
                heapq.heappush(
                    frontier, (-model.shaped_reward(ns), rng.random(), ns)
                )
        if best >= 1.0:
            return 1.0
    return best


def difficulty_label(
    level: SokobanLevel, n_seeds: int = 12, cfg: Optional[SolverConfig] = None
) -> Dict[str, object]:
    """Difficulty label D = 1 - mean_r best_shaped_reward over n_seeds runs, plus
    the per-seed rewards (needed for split-half reliability)."""
    cfg = cfg or SolverConfig()
    rewards = [run_search(level, seed=s, cfg=cfg) for s in range(n_seeds)]
    rewards = np.array(rewards, dtype=float)
    return {
        "level_id": level.level_id,
        "tier": level.tier,
        "per_seed_reward": rewards.tolist(),
        "mean_reward": float(rewards.mean()),
        "difficulty": float(1.0 - rewards.mean()),
        "solved_frac": float((rewards >= 1.0).mean()),
    }


if __name__ == "__main__":
    import argparse
    import time
    from pathlib import Path

    from sokoban_core import iter_levels_in_file

    _ROOT = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(_ROOT / "data/boxoban/hard/000.txt"))
    ap.add_argument("--tier", default="hard")
    ap.add_argument("--n-levels", type=int, default=5)
    ap.add_argument("--n-seeds", type=int, default=12)
    ap.add_argument("--budget", type=int, default=4000)
    args = ap.parse_args()

    cfg = SolverConfig(budget=args.budget)
    levels = iter_levels_in_file(args.file, tier=args.tier)[: args.n_levels]
    t0 = time.time()
    for lvl in levels:
        res = difficulty_label(lvl, n_seeds=args.n_seeds, cfg=cfg)
        print(
            f"{res['level_id']:20s} D={res['difficulty']:.3f} "
            f"mean_r={res['mean_reward']:.3f} solved={res['solved_frac']:.2f} "
            f"per_seed={[round(x,2) for x in res['per_seed_reward']]}"
        )
    print(f"\n{len(levels)} levels x {args.n_seeds} seeds in {time.time()-t0:.1f}s")
