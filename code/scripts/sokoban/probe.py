#!/usr/bin/env python3
"""
Cheap dynamics-aware probe for Sokoban (paper section: cheap probe, Eq. mini-hc).

The probe is the Sokoban analogue of the Karel fixed-library probe: it executes a
small, fixed set of short action programs in the level --- a handful of greedy
"push the nearest box toward its nearest target" macros plus several short random
action sequences --- each run once under a tiny budget, and records the shaped
reward attained. These cheap rollouts sense the interaction between layout and the
box/target reward structure (which the static geometric descriptor cannot observe)
at roughly two orders of magnitude less cost than the full budgeted solver.

No DSL search and no learned model: just fixed/random short rollouts.
"""

from __future__ import annotations

import random
from typing import Dict, List

import numpy as np

from sokoban_core import SokobanLevel
from solver import SokobanModel, State, _MOVES

# probe budget: a handful of steps, vs thousands for the full solver
PROBE_STEPS = 30
N_RANDOM_PROGRAMS = 8


def _greedy_push_rollout(model: SokobanModel, max_steps: int = PROBE_STEPS) -> float:
    """Deterministic greedy: at each step take the successor with the highest shaped
    reward (one-ply lookahead). A short, fixed 'always push toward goal' program."""
    state: State = (model.start_player, model.start_boxes)
    best = model.shaped_reward(state)
    for _ in range(max_steps):
        succ = model.successors(state)
        if not succ:
            break
        rewards = [model.shaped_reward(s) for s in succ]
        j = int(np.argmax(rewards))
        if rewards[j] <= model.shaped_reward(state):
            # no improving one-ply move; greedy program stalls
            break
        state = succ[j]
        best = max(best, rewards[j])
        if best >= 1.0:
            return 1.0
    return best


def _random_rollout(model: SokobanModel, seed: int, max_steps: int = PROBE_STEPS) -> float:
    rng = random.Random(seed)
    state: State = (model.start_player, model.start_boxes)
    best = model.shaped_reward(state)
    for _ in range(max_steps):
        succ = model.successors(state)
        if not succ:
            break
        state = rng.choice(succ)
        best = max(best, model.shaped_reward(state))
        if best >= 1.0:
            return 1.0
    return best


def probe_features(level: SokobanLevel) -> Dict[str, float]:
    """Cheap probe feature vector for one level."""
    model = SokobanModel(level)
    if model.start_player is None or model.num_boxes == 0:
        return {
            "probe_greedy_reward": 0.0,
            "probe_random_mean": 0.0,
            "probe_random_max": 0.0,
            "probe_random_min": 0.0,
            "probe_best": 0.0,
            "probe_init_reward": 0.0,
        }
    greedy = _greedy_push_rollout(model)
    randoms = [_random_rollout(model, seed=s) for s in range(N_RANDOM_PROGRAMS)]
    randoms = np.array(randoms, dtype=float)
    init_r = model.shaped_reward((model.start_player, model.start_boxes))
    return {
        "probe_greedy_reward": float(greedy),
        "probe_random_mean": float(randoms.mean()),
        "probe_random_max": float(randoms.max()),
        "probe_random_min": float(randoms.min()),
        "probe_best": float(max(greedy, randoms.max())),
        "probe_init_reward": float(init_r),
    }


if __name__ == "__main__":
    from pathlib import Path
    from sokoban_core import iter_levels_in_file

    _ROOT = Path(__file__).resolve().parents[2]
    levels = iter_levels_in_file(_ROOT / "data/boxoban/hard/000.txt", tier="hard")[:5]
    for lvl in levels:
        print(lvl.level_id, probe_features(lvl))
