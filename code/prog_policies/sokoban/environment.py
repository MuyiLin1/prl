from __future__ import annotations

import copy
import hashlib
from collections import deque
from typing import Union

import numpy as np

from prog_policies.base import BaseEnvironment

# Channel layout (boolean, shape (NUM_CHANNELS, H, W)):
#   0: agent facing North
#   1: agent facing East
#   2: agent facing South
#   3: agent facing West
#   4: wall
#   5: box
#   6: target
NUM_CHANNELS = 7

# Boxoban ASCII legend:
#   '#' wall, '@' player, '$' box, '.' target, '*' box-on-target,
#   '+' player-on-target, ' ' floor
DIRECTION_MAP = {
    0: (-1, 0),  # North
    1: (0, 1),   # East
    2: (1, 0),   # South
    3: (0, -1),  # West
}


def ascii_to_state(rows: list[str]) -> np.ndarray:
    """Parse a list of Boxoban ASCII rows into the channel-encoded state array.

    The agent always starts facing North (channel 0)."""
    # Normalise to a rectangular grid (pad short rows with floor).
    height = len(rows)
    width = max(len(r) for r in rows)
    state = np.zeros((NUM_CHANNELS, height, width), dtype=bool)
    for r in range(height):
        line = rows[r]
        for c in range(width):
            ch = line[c] if c < len(line) else ' '
            if ch == '#':
                state[4, r, c] = True
            elif ch == '$':
                state[5, r, c] = True
            elif ch == '.':
                state[6, r, c] = True
            elif ch == '*':
                state[5, r, c] = True
                state[6, r, c] = True
            elif ch == '@':
                state[0, r, c] = True
            elif ch == '+':
                state[0, r, c] = True
                state[6, r, c] = True
            # ' ' -> floor, nothing set
    return state


class SokobanEnvironment(BaseEnvironment):
    """Sokoban world that mirrors the Karel environment contract so it can be
    driven by the same DSL / Hill Climbing machinery.

    Actions are Karel-style: ``move`` advances one cell in the facing direction
    and pushes a box if one is directly ahead and the cell beyond it is free.
    ``turnLeft`` / ``turnRight`` rotate the agent. Pushing is therefore implicit
    in ``move`` (there is no separate push action), exactly mirroring how Karel
    folds interaction into movement."""

    def __init__(self, env_height: int = 10, env_width: int = 10,
                 crashable: bool = False, leaps_behaviour: bool = False,
                 max_calls: int = 10000,
                 initial_state: Union[np.ndarray, None] = None):
        self.crashable = crashable
        self.leaps_behaviour = leaps_behaviour
        actions = {
            "move": self.move,
            "turnLeft": self.turn_left,
            "turnRight": self.turn_right,
            "solveStep": self.solve_step,
        }
        # Cache of solve_step decisions keyed by board signature so repeated
        # boards (e.g. no-op loops, restored resets) cost O(1).
        self._solve_cache: dict = {}
        bool_features = {
            "frontIsClear": self.front_is_clear,
            "leftIsClear": self.left_is_clear,
            "rightIsClear": self.right_is_clear,
            "frontIsBox": self.front_is_box,
            "frontIsTarget": self.front_is_target,
            "boxIsPushable": self.box_is_pushable,
        }
        int_features = {}
        if initial_state is not None:
            state_shape = initial_state.shape
        else:
            state_shape = (NUM_CHANNELS, env_height, env_width)
        super().__init__(actions, bool_features, int_features, state_shape,
                         initial_state, max_calls=max_calls)

    # ------------------------------------------------------------------ state
    def default_state(self) -> np.ndarray:
        state = np.zeros(self.state_shape, dtype=bool)
        # Border walls + agent facing North at (1, 1).
        state[4, 0, :] = True
        state[4, -1, :] = True
        state[4, :, 0] = True
        state[4, :, -1] = True
        state[0, 1, 1] = True
        return state

    def set_state(self, state: np.ndarray) -> None:
        self.state = copy.deepcopy(state)
        d, r, c = np.where(self.state[:4, :, :] > 0)
        self.agent_pos = [int(r[0]), int(c[0]), int(d[0])]

    def get_state(self) -> np.ndarray:
        return self.state

    @classmethod
    def from_string(cls, state_str: str, crashable: bool = False,
                    max_calls: int = 10000):
        rows = state_str.replace('|', '').split('\n')
        rows = [r for r in rows if len(r) > 0]
        state = ascii_to_state(rows)
        return cls(initial_state=state, crashable=crashable, max_calls=max_calls)

    def __eq__(self, other: "SokobanEnvironment") -> bool:
        if self.agent_pos[:2] != other.agent_pos[:2]:
            return False
        return np.array_equal(self.state[5], other.state[5])

    def hash(self, size: int = 16) -> str:
        sample_hash = hashlib.sha256()
        boxes = np.argwhere(self.state[5]).flatten().tolist()
        state_str = ''.join(map(str, self.agent_pos + boxes))
        sample_hash.update(state_str.encode("utf8"))
        return sample_hash.hexdigest()[:size]

    # -------------------------------------------------------------- geometry
    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.state_shape[1] and 0 <= c < self.state_shape[2]

    def is_wall(self, r: int, c: int) -> bool:
        if not self.in_bounds(r, c):
            return True
        return bool(self.state[4, r, c])

    def is_box(self, r: int, c: int) -> bool:
        if not self.in_bounds(r, c):
            return False
        return bool(self.state[5, r, c])

    def is_free(self, r: int, c: int) -> bool:
        return self.in_bounds(r, c) and not self.is_wall(r, c) and not self.is_box(r, c)

    def _front_cell(self) -> tuple[int, int]:
        r, c, d = self.agent_pos
        dr, dc = DIRECTION_MAP[d]
        return r + dr, c + dc

    # ------------------------------------------------------------ perceptions
    def front_is_clear(self) -> bool:
        nr, nc = self._front_cell()
        return self.is_free(nr, nc)

    def left_is_clear(self) -> bool:
        r, c, d = self.agent_pos
        dr, dc = DIRECTION_MAP[(d - 1) % 4]
        return self.is_free(r + dr, c + dc)

    def right_is_clear(self) -> bool:
        r, c, d = self.agent_pos
        dr, dc = DIRECTION_MAP[(d + 1) % 4]
        return self.is_free(r + dr, c + dc)

    def front_is_box(self) -> bool:
        nr, nc = self._front_cell()
        return self.is_box(nr, nc)

    def front_is_target(self) -> bool:
        nr, nc = self._front_cell()
        if not self.in_bounds(nr, nc):
            return False
        return bool(self.state[6, nr, nc])

    def box_is_pushable(self) -> bool:
        r, c, d = self.agent_pos
        dr, dc = DIRECTION_MAP[d]
        nr, nc = r + dr, c + dc
        if not self.is_box(nr, nc):
            return False
        return self.is_free(nr + dr, nc + dc)

    # ---------------------------------------------------------------- actions
    def move(self) -> None:
        r, c, d = self.agent_pos
        dr, dc = DIRECTION_MAP[d]
        nr, nc = r + dr, c + dc

        if self.is_box(nr, nc):
            br, bc = nr + dr, nc + dc
            if self.is_free(br, bc):
                # Push the box one cell forward, agent follows.
                self.state[5, nr, nc] = False
                self.state[5, br, bc] = True
                self.state[d, r, c] = False
                self.state[d, nr, nc] = True
                self.agent_pos = [nr, nc, d]
            elif self.crashable:
                self.crashed = True
            elif self.leaps_behaviour:
                self.turn_left()
                self.turn_left()
            return

        if self.is_free(nr, nc):
            self.state[d, r, c] = False
            self.state[d, nr, nc] = True
            self.agent_pos = [nr, nc, d]
        elif self.crashable:
            self.crashed = True
        elif self.leaps_behaviour:
            self.turn_left()
            self.turn_left()

    def turn_left(self) -> None:
        r, c, d = self.agent_pos
        new_d = (d - 1) % 4
        self.state[d, r, c] = False
        self.state[new_d, r, c] = True
        self.agent_pos = [r, c, new_d]

    def turn_right(self) -> None:
        r, c, d = self.agent_pos
        new_d = (d + 1) % 4
        self.state[d, r, c] = False
        self.state[new_d, r, c] = True
        self.agent_pos = [r, c, new_d]

    # ------------------------------------------------------------ macro action
    def _plan_box_to_target(self, box: tuple[int, int], tgt: tuple[int, int],
                            other_boxes: frozenset, walls: np.ndarray):
        """Single-box Sokoban push planner.

        BFS over joint (box position, agent position) states, treating every
        other box and every wall as an immovable obstacle. Returns
        ``(plan_len, (agent_r, agent_c, last_push_dir))`` for the shortest plan
        that lands ``box`` on ``tgt``, or ``None`` if no feasible plan exists
        from the agent's current position."""
        h, w = walls.shape
        ar, ac, _ = self.agent_pos

        def agent_free(cell: tuple[int, int], boxpos: tuple[int, int]) -> bool:
            r, c = cell
            if not (0 <= r < h and 0 <= c < w):
                return False
            if walls[r, c] or cell == boxpos or cell in other_boxes:
                return False
            return True

        start = (box, (ar, ac))
        dist = {start: 0}
        q = deque([start])
        while q:
            boxpos, agentpos = q.popleft()
            base = dist[(boxpos, agentpos)]
            # (1) agent walks one cell (free, never through the pushed box).
            for dr, dc in DIRECTION_MAP.values():
                ncell = (agentpos[0] + dr, agentpos[1] + dc)
                if agent_free(ncell, boxpos):
                    ns = (boxpos, ncell)
                    if ns not in dist:
                        dist[ns] = base + 1
                        q.append(ns)
            # (2) agent pushes the box: agent must stand just behind it.
            for d, (dr, dc) in DIRECTION_MAP.items():
                if agentpos != (boxpos[0] - dr, boxpos[1] - dc):
                    continue
                nbox = (boxpos[0] + dr, boxpos[1] + dc)
                r, c = nbox
                if not (0 <= r < h and 0 <= c < w):
                    continue
                if walls[r, c] or nbox in other_boxes:
                    continue
                if nbox == tgt:
                    return base + 1, (boxpos[0], boxpos[1], d)
                ns = (nbox, boxpos)
                if ns not in dist:
                    dist[ns] = base + 1
                    q.append(ns)
        return None

    def solve_step(self) -> None:
        """Macro action: pick the cheapest box that can be pushed all the way to
        a free target (with every other box held fixed) and place it there,
        leaving the agent in the final pushing cell. No-op if no box can be
        completed from the current configuration. Counts as a single action."""
        state = self.state
        boxes = [tuple(b) for b in np.argwhere(state[5])]
        key = (self.agent_pos[0], self.agent_pos[1], frozenset(boxes))
        if key in self._solve_cache:
            best = self._solve_cache[key]
        else:
            walls = state[4]
            box_set = set(boxes)
            targets = [tuple(t) for t in np.argwhere(state[6])]
            free_targets = [t for t in targets if t not in box_set]
            cand_boxes = [b for b in boxes if not state[6, b[0], b[1]]]
            best = None
            for b in cand_boxes:
                others = frozenset(x for x in boxes if x != b)
                for t in free_targets:
                    res = self._plan_box_to_target(b, t, others, walls)
                    if res is not None:
                        plen, final_agent = res
                        if best is None or plen < best[0]:
                            best = (plen, b, t, final_agent)
            if len(self._solve_cache) < 200_000:
                self._solve_cache[key] = best
        if best is None:
            return
        _, b, t, final_agent = best
        ar, ac, ad = self.agent_pos
        self.state[5, b[0], b[1]] = False
        self.state[5, t[0], t[1]] = True
        self.state[ad, ar, ac] = False
        fr, fc, fd = final_agent
        self.state[fd, fr, fc] = True
        self.agent_pos = [fr, fc, fd]

    # ------------------------------------------------------------ rendering
    _GLYPH = {'wall': '#', 'box': '$', 'target': '.', 'box_on_target': '*',
              'floor': ' '}
    _AGENT = {0: '^', 1: '>', 2: 'v', 3: '<'}

    def to_string_worldcoder_style(self, state: np.ndarray = None) -> str:
        s = self.state if state is None else state
        h, w = s.shape[1], s.shape[2]
        lines = []
        for r in range(h):
            row = []
            for c in range(w):
                agent_dir = np.where(s[:4, r, c])[0]
                if s[4, r, c]:
                    row.append('#')
                elif len(agent_dir) > 0:
                    row.append(self._AGENT[int(agent_dir[0])])
                elif s[5, r, c] and s[6, r, c]:
                    row.append('*')
                elif s[5, r, c]:
                    row.append('$')
                elif s[6, r, c]:
                    row.append('.')
                else:
                    row.append(' ')
            lines.append(''.join(row))
        return '\n'.join(lines)

    def record_partial_state(self) -> str:
        return self.to_string_worldcoder_style()

    def to_image(self, grid_size: int = 100, root_dir: str = "./") -> np.ndarray:
        # Lightweight renderer (no external textures): colour-code channels.
        h, w = self.state_shape[1], self.state_shape[2]
        img = np.zeros((h * grid_size, w * grid_size, 3), dtype=np.uint8)
        colours = {
            'floor': (235, 235, 235),
            'wall': (60, 60, 60),
            'box': (190, 130, 60),
            'box_on_target': (90, 170, 90),
            'target': (210, 200, 120),
            'agent': (60, 110, 200),
        }
        for r in range(h):
            for c in range(w):
                if self.state[4, r, c]:
                    col = colours['wall']
                elif self.state[5, r, c] and self.state[6, r, c]:
                    col = colours['box_on_target']
                elif self.state[5, r, c]:
                    col = colours['box']
                elif self.state[6, r, c]:
                    col = colours['target']
                else:
                    col = colours['floor']
                if self.state[:4, r, c].any():
                    col = colours['agent']
                img[r * grid_size:(r + 1) * grid_size,
                    c * grid_size:(c + 1) * grid_size] = col
        return img
