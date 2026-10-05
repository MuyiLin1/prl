#!/usr/bin/env python3
"""
Sokoban (Boxoban) core: ASCII level parsing and cheap geometric / reward-structure
descriptors for the third-domain difficulty experiments (paper section 6.5).

This is the Sokoban analogue of `karel_curriculum_difficulty_predictor.py`. It reads
a Boxoban ASCII level, builds the free-cell grid graph, and extracts the same three
descriptor layers used for Karel/MiniGrid (local, reachability, spectral) plus a set
of Sokoban-specific reward-structure features (box/target counts, box->target
distances, corner deadlocks, tunnels). No solver and no rollout are run here: every
feature is read statically from the level, exactly as in the geometric predictor of
the paper.

Boxoban ASCII encoding (10x10 grids):
    '#'  wall
    ' '  floor
    '@'  player
    '$'  box
    '.'  target (goal)
    '*'  box on target
    '+'  player on target
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Reuse the validated grid-graph helpers from the Karel predictor so that the
# geometric descriptor is computed identically across domains.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from karel_curriculum_difficulty_predictor import (  # noqa: E402
    bfs_distances,
    bridges_and_articulations,
    build_grid_graph,
    connected_components,
    graph_laplacian_eigs,
)

Cell = Tuple[int, int]


# ------------------------------- level parsing -------------------------------


@dataclass
class SokobanLevel:
    """A parsed Boxoban level."""

    grid: List[str]                 # raw rows, walls intact
    height: int
    width: int
    walls: set = field(default_factory=set)
    floor: set = field(default_factory=set)       # all non-wall cells (traversable)
    boxes: set = field(default_factory=set)
    targets: set = field(default_factory=set)
    player: Optional[Cell] = None
    level_id: str = ""
    tier: str = ""

    @property
    def num_boxes(self) -> int:
        return len(self.boxes)

    @property
    def num_targets(self) -> int:
        return len(self.targets)


def parse_level(rows: Sequence[str], level_id: str = "", tier: str = "") -> SokobanLevel:
    """Parse a single Boxoban level (a list of ASCII rows) into a SokobanLevel."""
    grid = [row.rstrip("\n") for row in rows]
    height = len(grid)
    width = max((len(r) for r in grid), default=0)
    # pad ragged rows with walls so indexing is safe
    grid = [r.ljust(width, "#") for r in grid]

    walls: set = set()
    floor: set = set()
    boxes: set = set()
    targets: set = set()
    player: Optional[Cell] = None

    for y, row in enumerate(grid):
        for x, ch in enumerate(row):
            cell = (y, x)
            if ch == "#":
                walls.add(cell)
                continue
            # everything below is traversable floor
            floor.add(cell)
            if ch == "$":
                boxes.add(cell)
            elif ch == "*":
                boxes.add(cell)
                targets.add(cell)
            elif ch == ".":
                targets.add(cell)
            elif ch == "@":
                player = cell
            elif ch == "+":
                player = cell
                targets.add(cell)
            # ' ' is plain floor
    return SokobanLevel(
        grid=grid,
        height=height,
        width=width,
        walls=walls,
        floor=floor,
        boxes=boxes,
        targets=targets,
        player=player,
        level_id=level_id,
        tier=tier,
    )


def iter_levels_in_file(path: str | Path, tier: str = "") -> List[SokobanLevel]:
    """Parse all levels in a Boxoban file. Levels are separated by '; <id>' headers."""
    text = Path(path).read_text(encoding="utf-8")
    levels: List[SokobanLevel] = []
    cur_rows: List[str] = []
    cur_id = ""
    file_stem = Path(path).stem

    def flush() -> None:
        nonlocal cur_rows, cur_id
        if cur_rows:
            lid = f"{tier}/{file_stem}/{cur_id}" if tier else f"{file_stem}/{cur_id}"
            levels.append(parse_level(cur_rows, level_id=lid, tier=tier))
        cur_rows = []

    for line in text.splitlines():
        if line.startswith(";"):
            flush()
            cur_id = line[1:].strip()
        elif line.strip() == "":
            continue
        else:
            cur_rows.append(line)
    flush()
    return levels


# ----------------------- Sokoban-specific helpers ----------------------------


def _neighbors4(cell: Cell) -> List[Cell]:
    y, x = cell
    return [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]


def is_dead_corner(cell: Cell, walls: set, targets: set) -> bool:
    """A non-target floor cell wedged into a wall corner: a box pushed here can
    never be moved again, an unrecoverable Sokoban deadlock."""
    if cell in targets:
        return False
    y, x = cell
    up, down = (y - 1, x) in walls, (y + 1, x) in walls
    left, right = (y, x - 1) in walls, (y, x + 1) in walls
    return (up or down) and (left or right)


def hungarian_min_cost(cost: np.ndarray) -> float:
    """Minimum-cost assignment (Hungarian) total cost; falls back to a greedy
    estimate if scipy is unavailable. cost is square (n boxes x n targets)."""
    if cost.size == 0:
        return 0.0
    try:
        from scipy.optimize import linear_sum_assignment

        r, c = linear_sum_assignment(cost)
        return float(cost[r, c].sum())
    except Exception:
        # greedy fallback
        cost = cost.copy()
        total = 0.0
        n = cost.shape[0]
        for _ in range(n):
            i, j = np.unravel_index(np.argmin(cost), cost.shape)
            total += float(cost[i, j])
            cost[i, :] = np.inf
            cost[:, j] = np.inf
        return total


# ------------------------------ feature builder ------------------------------


def extract_features(level: SokobanLevel, n_eigs: int = 6) -> Dict[str, float]:
    """Cheap static descriptor for a Sokoban level: geometry (local/reachability/
    spectral, reused from the Karel extractor) plus Sokoban reward structure."""
    free_cells = sorted(level.floor)
    graph = build_grid_graph(free_cells)
    comps = connected_components(graph)
    largest = max((len(c) for c in comps), default=0)
    degrees = (
        np.array([len(nbrs) for nbrs in graph.values()], dtype=float)
        if graph
        else np.array([0.0])
    )
    bridges, arts = bridges_and_articulations(graph)

    # player reachability / eccentricity
    start = level.player
    start_dist = bfs_distances(graph, start) if start in graph else {}
    start_reachable = len(start_dist)
    start_ecc = max(start_dist.values()) if start_dist else 0

    # diameter + mean pairwise distance (grids are 10x10, exact is cheap)
    diameter = 0
    mpd_accum = 0.0
    mpd_count = 0
    for node in graph:
        d = bfs_distances(graph, node)
        if d:
            diameter = max(diameter, max(d.values()))
            mpd_accum += sum(d.values())
            mpd_count += len(d)
    mean_pair_dist = mpd_accum / max(1, mpd_count)

    eigs = graph_laplacian_eigs(graph, k=n_eigs)

    # ---- Sokoban reward-structure features ----
    boxes = sorted(level.boxes)
    targets = sorted(level.targets)
    nb, nt = len(boxes), len(targets)

    # box <-> target distances on the floor graph (ignoring other boxes: lower bound)
    target_dists = {t: bfs_distances(graph, t) for t in targets}
    bt_min_dists: List[float] = []
    cost = np.full((nb, max(nb, nt)), 1e6, dtype=float)
    for i, b in enumerate(boxes):
        best = 1e6
        for j, t in enumerate(targets):
            dd = target_dists[t].get(b, 1e6)
            cost[i, j] = dd
            best = min(best, dd)
        bt_min_dists.append(best if best < 1e6 else 0.0)
    # square the cost matrix for assignment
    n = max(nb, nt)
    sq = np.full((n, n), 1e6, dtype=float)
    sq[:nb, :nt] = cost[:nb, :nt]
    matching_lb = hungarian_min_cost(sq) if nb and nt else 0.0
    matching_lb = matching_lb if matching_lb < 1e6 * n else 0.0

    boxes_on_target = len(level.boxes & level.targets)
    dead_cells = sum(1 for c in level.floor if is_dead_corner(c, level.walls, level.targets))
    boxes_in_corner = sum(
        1 for b in boxes if is_dead_corner(b, level.walls, level.targets)
    )
    player_to_box = (
        min((start_dist.get(b, 1e6) for b in boxes), default=0.0) if start_dist else 0.0
    )
    player_to_box = player_to_box if player_to_box < 1e6 else 0.0

    h, w = level.height, level.width
    feats: Dict[str, float] = {
        # local
        "grid_height": float(h),
        "grid_width": float(w),
        "grid_area": float(h * w),
        "num_free_cells": float(len(free_cells)),
        "num_wall_cells": float(len(level.walls)),
        "wall_fraction": float(len(level.walls) / max(1, h * w)),
        "num_edges": float(sum(len(v) for v in graph.values()) / 2.0),
        "mean_degree": float(np.mean(degrees)),
        "min_degree": float(np.min(degrees)),
        "max_degree": float(np.max(degrees)),
        "num_deadends": float(np.sum(degrees <= 1)),
        "num_corridor_cells": float(np.sum(degrees == 2)),
        "num_branch_cells": float(np.sum(degrees >= 3)),
        # reachability
        "num_components": float(len(comps)),
        "largest_component_frac": float(largest / max(1, len(free_cells))),
        "graph_diameter": float(diameter),
        "mean_pair_distance": float(mean_pair_dist),
        "player_reachable_frac": float(start_reachable / max(1, len(free_cells))),
        "player_eccentricity": float(start_ecc),
        "num_bridges": float(bridges),
        "num_articulation_points": float(arts),
        # reward structure (Sokoban-specific)
        "num_boxes": float(nb),
        "num_targets": float(nt),
        "boxes_on_target_init": float(boxes_on_target),
        "box_target_dist_sum": float(np.sum(bt_min_dists)),
        "box_target_dist_mean": float(np.mean(bt_min_dists)) if bt_min_dists else 0.0,
        "box_target_dist_max": float(np.max(bt_min_dists)) if bt_min_dists else 0.0,
        "box_target_matching_lb": float(matching_lb),
        "num_dead_cells": float(dead_cells),
        "boxes_in_corner_init": float(boxes_in_corner),
        "player_to_nearest_box": float(player_to_box),
    }
    for i, ev in enumerate(eigs):
        feats[f"laplacian_eig_{i}"] = float(ev)
    return feats


FEATURE_ORDER: Optional[List[str]] = None


def features_to_vector(feats: Dict[str, float]) -> Tuple[List[str], np.ndarray]:
    keys = sorted(feats.keys())
    return keys, np.array([feats[k] for k in keys], dtype=float)


if __name__ == "__main__":
    # quick self-test against the cloned data
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(_ROOT / "data/boxoban/hard/000.txt"))
    ap.add_argument("--n", type=int, default=2)
    args = ap.parse_args()

    levels = iter_levels_in_file(args.file, tier="hard")
    print(f"parsed {len(levels)} levels from {args.file}")
    for lvl in levels[: args.n]:
        print(f"\n--- level {lvl.level_id} "
              f"(boxes={lvl.num_boxes}, targets={lvl.num_targets}, player={lvl.player}) ---")
        print("\n".join(lvl.grid))
        feats = extract_features(lvl)
        for k in sorted(feats):
            print(f"  {k:28s} {feats[k]:.4f}")
