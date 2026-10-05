"""Difficulty scoring for the LavaCrossing curriculum experiment.

Pure (no torch / no SB3) so it can be imported cheaply by the harness and by
analysis scripts. Provides:

  * A `Layout` = one concrete LavaCrossing instance, identified by (tier N, seed).
  * `geometric_features(...)`  -> cheap features readable from the grid alone
        (NO policy, NO training): river count, BFS detour, lava mass, etc.
  * `enumerate_pool(...)`      -> a de-duplicated pool of distinct layouts.
  * `geometric_score(...)`     -> a cheap scalar difficulty estimate (the
        "Geometry" curriculum ordering key).
  * `bin_layouts(...)`         -> split an ordered pool into easy/med/hard bins.

Difficulty structure (from the de-risk): the DOMINANT axis is the number of lava
rivers (the tier N), which is itself a geometric property -- a "river" is a full
interior row or column of lava with a single crossing gap. The BFS detour adds a
weak within-tier refinement. So `geometric_score` = river_count (dominant) with
detour as a normalized tiebreaker.

NO LLM. Standard gymnasium MiniGrid CrossingEnv (Version B).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

WALL, LAVA, GOAL = 2, 9, 8


def make_raw_env(size: int, n: int):
    from minigrid.envs import CrossingEnv

    return CrossingEnv(size=size, num_crossings=n, max_steps=4 * size * size)


def _grid_objects(size: int, n: int, seed: int):
    """Return (img[x,y]=object-id, start(x,y), goal(x,y)) for a layout, or None if unsolvable shape."""
    from minigrid.wrappers import FullyObsWrapper

    env = FullyObsWrapper(make_raw_env(size, n))
    obs, _ = env.reset(seed=seed)
    img = obs["image"][:, :, 0]  # [x, y] -> object id
    start = tuple(int(v) for v in env.unwrapped.agent_pos)
    goal_cells = list(zip(*np.where(img == GOAL)))
    env.close()
    if not goal_cells:
        return None
    return img, start, (int(goal_cells[0][0]), int(goal_cells[0][1]))


def _bfs(img: np.ndarray, start, goal, block_lava: bool) -> int:
    """Shortest 4-connected path length start->goal. Walls always block; lava blocks iff block_lava."""
    w, h = img.shape
    blocked = (img == WALL)
    if block_lava:
        blocked = blocked | (img == LAVA)
    blocked = blocked.copy()
    sx, sy = start
    gx, gy = goal
    blocked[sx, sy] = False
    blocked[gx, gy] = False
    dist = -np.ones((w, h), dtype=int)
    dist[sx, sy] = 0
    q = deque([(sx, sy)])
    while q:
        x, y = q.popleft()
        if (x, y) == (gx, gy):
            return int(dist[x, y])
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < h and not blocked[nx, ny] and dist[nx, ny] < 0:
                dist[nx, ny] = dist[x, y] + 1
                q.append((nx, ny))
    return int(dist[gx, gy])  # -1 if unreachable


def _estimate_rivers(img: np.ndarray) -> int:
    """Recover the number of lava rivers from the grid alone (NOT from the tier label).

    A river is a full interior row or column that is mostly lava (with one crossing
    gap). Count interior rows and columns whose lava fraction exceeds 0.5.
    """
    w, h = img.shape
    rivers = 0
    # interior columns (vertical rivers run along y for a fixed x)
    for x in range(1, w - 1):
        col = img[x, 1:h - 1]
        if (col == LAVA).mean() > 0.5:
            rivers += 1
    # interior rows (horizontal rivers run along x for a fixed y)
    for y in range(1, h - 1):
        row = img[1:w - 1, y]
        if (row == LAVA).mean() > 0.5:
            rivers += 1
    return rivers


@dataclass
class Layout:
    n: int                       # tier (num_crossings) used to construct it
    seed: int                    # gym reset seed
    size: int
    sig: tuple                   # canonical signature (for de-dup)
    features: dict = field(default_factory=dict)

    @property
    def detour(self) -> int:
        return self.features["detour"]


def geometric_features(size: int, n: int, seed: int):
    """Cheap, policy-free geometric descriptor of a layout. Returns (features, sig) or None."""
    got = _grid_objects(size, n, seed)
    if got is None:
        return None
    img, start, goal = got
    detour = _bfs(img, start, goal, block_lava=True)
    if detour < 0:
        return None  # unsolvable layout; skip
    free = _bfs(img, start, goal, block_lava=False)
    n_lava = int((img == LAVA).sum())
    est_rivers = _estimate_rivers(img)
    sg_manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    feats = {
        "est_rivers": est_rivers,        # one difficulty axis (free readout of tier N)
        "detour": detour,                # BFS shortest path avoiding lava
        "free_path": free,               # BFS ignoring lava (lower bound)
        "detour_excess": detour - free,  # extra steps forced by lava
        "n_lava": n_lava,                # total lava mass
        "sg_manhattan": sg_manhattan,    # start->goal manhattan distance
        "start": start,                  # agent start cell (for the behavioral probe)
        "goal": goal,                    # goal cell
    }
    # signature: lava layout + start + goal uniquely identifies the instance (pure python ints)
    lava_cells = tuple(sorted((int(x), int(y)) for x, y in np.argwhere(img == LAVA)))
    sig = (int(n), lava_cells, start, goal)
    return feats, sig


def enumerate_pool(sizes, tiers, n_seeds: int, cap_per_cell=None, seed_start: int = 0,
                   exclude_sigs=None):
    """De-duplicated pool of distinct, solvable layouts over the (size x tier) grid.

    sizes:         int or list of grid sizes (the 2nd difficulty axis).
    cap_per_cell:  if set, keep at most this many distinct layouts per (size, tier) cell.
    seed_start:    first reset seed (use a high value to build a DISJOINT held-out set).
    exclude_sigs:  optional iterable of signatures to skip (keeps held-out disjoint from train).
    """
    if isinstance(sizes, int):
        sizes = [sizes]
    pool: list[Layout] = []
    seen: set = set(exclude_sigs) if exclude_sigs else set()
    for size in sizes:
        for n in tiers:
            kept = 0
            for s in range(seed_start, seed_start + n_seeds):
                if cap_per_cell is not None and kept >= cap_per_cell:
                    break
                res = geometric_features(size, n, s)
                if res is None:
                    continue
                feats, sig = res
                if sig in seen:
                    continue
                seen.add(sig)
                kept += 1
                pool.append(Layout(n=n, seed=s, size=size, sig=sig, features=feats))
    return pool


def geometric_score(layout: Layout) -> float:
    """Cheap scalar difficulty estimate from grid shape alone (NO policy, NO training).

    Two COMPARABLE axes deliberately trade off, so difficulty is a 2-D surface rather than
    a single tier stamp: the number of lava rivers and the grid size. A bigger maze with
    fewer rivers can match a small maze with more rivers -- which is exactly what makes the
    cheap-vs-expensive ordering comparison non-trivial. Detour adds a fine within-cell tiebreak.
        est_rivers in {1,2,3};  (size-7)/2 in {0,1,2};  detour/100 << 1.
    """
    f = layout.features
    size_term = (layout.size - 7) / 2.0
    return f["est_rivers"] + size_term + f["detour"] / 100.0


def random_walk_probe(layout: "Layout", k: int = 8, max_steps=None, seed: int = 0) -> float:
    """Cheap BEHAVIORAL difficulty signal (NO network, NO training).

    Run k random-action rollouts (restricted to the 3 nav actions) on the fixed layout and
    return the mean over rollouts of the closest Manhattan distance to the goal reached.
    Lower => a blind walker gets near the goal easily (easier); higher => it never approaches
    (harder). Distinct from pure geometry because it reflects how reachable the goal is under
    actual movement (lava deaths cut rollouts short).
    """
    env = make_raw_env(layout.size, layout.n)
    gx, gy = layout.features["goal"]
    rng = np.random.default_rng(seed + 1009 * (layout.seed + 1))
    cap = max_steps if max_steps is not None else 4 * layout.size * layout.size
    mins = []
    for _ in range(k):
        env.reset(seed=layout.seed)
        ax, ay = (int(v) for v in env.unwrapped.agent_pos)
        best = abs(ax - gx) + abs(ay - gy)
        done, steps = False, 0
        while not done and steps < cap:
            _, _, term, trunc, _ = env.step(int(rng.integers(3)))  # nav actions 0/1/2
            ax, ay = (int(v) for v in env.unwrapped.agent_pos)
            best = min(best, abs(ax - gx) + abs(ay - gy))
            done = term or trunc
            steps += 1
        mins.append(best)
    env.close()
    return float(np.mean(mins))


GEOM_FEATURE_NAMES = ["size", "n", "est_rivers", "detour", "free_path",
                      "detour_excess", "n_lava", "sg_manhattan"]


def feature_vector(layout: "Layout"):
    """Policy-free geometric feature vector for the fitted difficulty predictor."""
    f = layout.features
    return [layout.size, layout.n, f["est_rivers"], f["detour"], f["free_path"],
            f["detour_excess"], f["n_lava"], f["sg_manhattan"]]


def fit_geometry_oof(pool, labels, seed: int = 0, n_splits: int = 5, n_estimators: int = 200):
    """Out-of-fold RandomForest difficulty predictions from geometric features alone.

    labels: dict sig -> measured difficulty (e.g. oracle 1 - success). Returns dict sig -> OOF pred.

    This is the paper's fitted-predictor method (Part B): rather than a hand-tuned formula, learn the
    mapping geometry-features -> difficulty from data, so the right feature weighting (rivers >> size)
    is discovered automatically. K-fold cross-fitting means every layout's predicted difficulty comes
    from a forest that never saw it (honest, out-of-sample). At predict time only policy-free geometric
    features are used; the measured labels only calibrate the mapping (amortized once in deployment).
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold

    X = np.array([feature_vector(lo) for lo in pool], dtype=float)
    y = np.array([labels[lo.sig] for lo in pool], dtype=float)
    pred = np.zeros(len(pool))
    n_splits = max(2, min(n_splits, len(pool)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=1)
        rf.fit(X[tr], y[tr])
        pred[te] = rf.predict(X[te])
    return {lo.sig: float(p) for lo, p in zip(pool, pred)}


def bin_layouts(pool, key, n_bins: int = 3):
    """Split `pool` into `n_bins` ordered bins (easy->hard) by ascending `key(layout)`.

    Returns a list of lists; bin 0 = easiest. Uses quantile cut points so bins are
    balanced in count even when the score distribution is lumpy (tiered).
    """
    ordered = sorted(pool, key=key)
    n = len(ordered)
    edges = [round(i * n / n_bins) for i in range(n_bins + 1)]
    return [ordered[edges[i]:edges[i + 1]] for i in range(n_bins)]


def _summary(pool, n_bins: int = 3):
    """Human-readable pool summary for sanity-checking (printed by __main__)."""
    by_cell: dict = {}
    for lo in pool:
        by_cell.setdefault((lo.size, lo.n), []).append(lo)
    print(f"pool: {len(pool)} distinct layouts")
    for (sz, n), los in sorted(by_cell.items()):
        det = sorted({lo.detour for lo in los})
        print(f"  size={sz} N={n}: {len(los):3d} layouts | detour {det}")
    bins = bin_layouts(pool, geometric_score, n_bins)
    print(f"\ngeometric bins (easy->hard), n_bins={n_bins}:")
    for i, b in enumerate(bins):
        cells: dict = {}
        for lo in b:
            cells[(lo.size, lo.n)] = cells.get((lo.size, lo.n), 0) + 1
        sc = [geometric_score(lo) for lo in b]
        print(f"  bin{i}: {len(b):3d} layouts | cells {dict(sorted(cells.items()))} | "
              f"score {min(sc):.2f}..{max(sc):.2f}")


if __name__ == "__main__":
    import argparse
    import warnings

    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[7, 9, 11])
    ap.add_argument("--tiers", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n-seeds", type=int, default=80)
    ap.add_argument("--cap-per-cell", type=int, default=10)
    ap.add_argument("--n-bins", type=int, default=3)
    args = ap.parse_args()

    pool = enumerate_pool(args.sizes, args.tiers, args.n_seeds, args.cap_per_cell)
    _summary(pool, args.n_bins)


# --------------------------------------------------------------------------- v2 grader (2026-10-02)
# Pre-registered in PREREGISTRATION.md. Three-layer descriptor read from the layout alone, with NO
# designer labels (no tier n, no est_rivers, no explicit grid size). Fitted once on a separate,
# cheaper CALIBRATION pool (sizes 7/9/11) and applied to a curriculum pool it has never seen.

GRADER_FEATURES = ["num_free", "lava_frac", "mean_degree", "deadends", "corridor", "branch",
                   "bridges", "articulation", "start_ecc", "sg_dist", "sg_excess",
                   "lam2", "lam3", "lam4", "lam5", "lam6"]


def descriptor_features(size: int, n: int, seed: int) -> dict:
    """Local + reachability + spectral features of the feasibility graph (free = not wall, not lava)."""
    import sys
    from pathlib import Path as _P
    root = str(_P(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    from karel_curriculum_difficulty_predictor import (bfs_distances, bridges_and_articulations,
                                                      build_grid_graph, graph_laplacian_eigs)
    img, start, goal = _grid_objects(size, n, seed)
    free = [(x, y) for x in range(img.shape[0]) for y in range(img.shape[1])
            if img[x, y] not in (WALL, LAVA)]
    g = build_grid_graph(free)
    deg = np.array([len(v) for v in g.values()], float)
    br, art = bridges_and_articulations(g)
    sd = bfs_distances(g, start)
    eig = graph_laplacian_eigs(g, k=6)
    f = dict(num_free=len(free), lava_frac=float((img == LAVA).sum()) / len(free),
             mean_degree=float(deg.mean()), deadends=int((deg <= 1).sum()),
             corridor=int((deg == 2).sum()), branch=int((deg >= 3).sum()), bridges=br,
             articulation=art, start_ecc=max(sd.values()), sg_dist=sd.get(goal, -1),
             sg_excess=sd.get(goal, -1) - (abs(start[0] - goal[0]) + abs(start[1] - goal[1])))
    f.update({f"lam{i + 2}": float(e) for i, e in enumerate(eig[1:6])})
    return f


def fit_grader(calib_pool, calib_labels, seed: int = 0):
    """Fit the RF grader on calibration layouts. Returns predict(layouts) -> np.ndarray."""
    from sklearn.ensemble import RandomForestRegressor
    X = np.array([[descriptor_features(lo.size, lo.n, lo.seed)[k] for k in GRADER_FEATURES]
                  for lo in calib_pool], float)
    y = np.array([calib_labels[lo.sig] for lo in calib_pool], float)
    rf = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=1).fit(X, y)

    def predict(layouts):
        Z = np.array([[descriptor_features(lo.size, lo.n, lo.seed)[k] for k in GRADER_FEATURES]
                      for lo in layouts], float)
        return rf.predict(Z)
    return predict
