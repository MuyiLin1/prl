from __future__ import annotations

from collections import deque
from typing import Union

import numpy as np

from prog_policies.base import BaseTask, BaseEnvironment
from prog_policies.sokoban.environment import SokobanEnvironment, ascii_to_state

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - scipy is expected on the cluster
    _HAS_SCIPY = False


# Default weights for the shaped potential. ``W_ON_TARGET`` rewards boxes that
# are *actually* parked on a target (hard to fake); ``W_PROGRESS`` is the
# smoother matching-distance term that only supplies gradient. Emphasising the
# former keeps the label tied to genuine solving rather than box-nudging.
W_ON_TARGET = 0.8
W_PROGRESS = 0.2
_UNREACHABLE = 10_000


def _bfs_from(target: tuple[int, int], walls: np.ndarray) -> np.ndarray:
    """4-connected BFS distance from a single target over non-wall cells."""
    h, w = walls.shape
    dist = np.full((h, w), _UNREACHABLE, dtype=np.int32)
    tr, tc = target
    dist[tr, tc] = 0
    q = deque([(tr, tc)])
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not walls[nr, nc] \
                    and dist[nr, nc] == _UNREACHABLE:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


class BoxobanTask(BaseTask):
    """A single Boxoban level treated as one task instance.

    ``get_reward`` returns the positive increment of the best shaped potential
    reached so far, so the cumulative reward over a program rollout equals the
    best shaped value in [0, 1]. The difficulty label is then
    ``1 - mean_over_seeds(best_reward)`` exactly as for Karel / MiniGrid."""

    def __init__(self, env_args: dict = {}, seed: Union[int, None] = None):
        super().__init__(env_args, seed)

    def generate_initial_environment(self, env_args: dict) -> BaseEnvironment:
        ascii_str = env_args["ascii"]
        crashable = env_args.get("crashable", False)
        max_calls = env_args.get("max_calls", 10000)
        # Configurable shaping weights (Option 1: tune partial credit).
        self._w_on_target = float(env_args.get("w_on_target", W_ON_TARGET))
        self._w_progress = float(env_args.get("w_progress", W_PROGRESS))
        rows = [r for r in ascii_str.replace('|', '').split('\n') if len(r) > 0]
        state = ascii_to_state(rows)
        env = SokobanEnvironment(initial_state=state, crashable=crashable,
                                 max_calls=max_calls)

        # --- precompute static potential fields from the initial layout ---
        walls = state[4].copy()
        targets = [tuple(t) for t in np.argwhere(state[6])]
        boxes0 = [tuple(b) for b in np.argwhere(state[5])]
        self._targets = targets
        self._num_boxes = max(len(boxes0), 1)
        # BFS distance grid from each target over the floor graph.
        self._target_dist = [_bfs_from(t, walls) for t in targets]
        self._init_matching = self._matching_distance(boxes0)
        self._phi_best = 0.0
        return env

    def reset_environment(self) -> None:
        super().reset_environment()
        # Baseline potential of the initial layout (typically 0 for Boxoban).
        self._phi_best = self._compute_phi(self.environment)[0]

    def _matching_distance(self, boxes: list[tuple[int, int]]) -> float:
        """Minimum-cost assignment of boxes to targets over BFS floor distance."""
        n = len(self._targets)
        if n == 0 or len(boxes) == 0:
            return 0.0
        cost = np.full((len(boxes), n), float(_UNREACHABLE), dtype=float)
        for i, (br, bc) in enumerate(boxes):
            for j in range(n):
                cost[i, j] = float(self._target_dist[j][br, bc])
        if _HAS_SCIPY:
            rows, cols = linear_sum_assignment(cost)
            return float(cost[rows, cols].sum())
        # Greedy fallback if scipy is unavailable.
        used = set()
        total = 0.0
        for i in range(cost.shape[0]):
            order = np.argsort(cost[i])
            for j in order:
                if j not in used:
                    used.add(j)
                    total += cost[i, j]
                    break
        return total

    def _compute_phi(self, environment: SokobanEnvironment) -> tuple[float, bool]:
        state = environment.get_state()
        boxes = [tuple(b) for b in np.argwhere(state[5])]
        on_target = sum(1 for (r, c) in boxes if state[6, r, c])
        frac = on_target / self._num_boxes
        terminated = (on_target == len(self._targets) and len(self._targets) > 0)
        if self._init_matching <= 0:
            progress = 1.0
        else:
            d = self._matching_distance(boxes)
            progress = (self._init_matching - d) / self._init_matching
            progress = float(np.clip(progress, 0.0, 1.0))
        phi = self._w_on_target * frac + self._w_progress * progress
        phi = float(np.clip(phi, 0.0, 1.0))
        return phi, terminated

    def get_reward(self, environment: BaseEnvironment) -> tuple[bool, float]:
        phi, terminated = self._compute_phi(environment)
        increment = max(0.0, phi - self._phi_best)
        self._phi_best = max(self._phi_best, phi)
        return terminated, increment
