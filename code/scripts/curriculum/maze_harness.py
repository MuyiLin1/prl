"""Knob-free curriculum test: random MiniGrid mazes (the standard UED testbed; generator places walls only).

Reuses curriculum_harness.run_condition unchanged (same PPO, bins, gate, deadline, held-out eval); only the
level family and the difficulty scores are swapped in. Conditions:
  flat               no curriculum
  random_curriculum  curriculum mechanics, uniform random order
  wall_count         order by number of interior walls (the generator's only parameter: the "designer knob")
  path_len           order by BFS start-goal distance (simplest hand heuristic)
  grader             RF on layout features, fit on a SEPARATE calibration pool labelled by reference PPOs
Heuristic scores get a seeded random tiebreak (as in the v2 LavaCrossing runs).

usage: maze_harness.py --condition C --seed S --sizes 13 --max-walls 60 --results-dir DIR [harness args]
       maze_harness.py --build-calib --sizes 13 --max-walls 60   (reference PPOs on the calibration pool)
"""
import json, sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
import curriculum_harness as ch  # noqa: E402

MAX_WALLS = 60
CORNERS = False   # True: start top-left, goal bottom-right (as in MiniGrid Crossing); else random
PERFECT, P_MAX = False, 0.5   # PERFECT: DFS-carved maze, then remove a uniform fraction in [0, P_MAX] of walls
CALIB_P = None   # p_max of the calibration pool if it differs from the main run's
CALIB_SIZE, CALIB_WALLS = None, None   # default: same as the main run
CALIB_SEED_START, EVAL_SEED_START = 500_000, 100_000


def _maze_cls():
    from minigrid.core.grid import Grid
    from minigrid.core.mission import MissionSpace
    from minigrid.core.world_object import Goal, Wall
    from minigrid.minigrid_env import MiniGridEnv

    class RandomMazeEnv(MiniGridEnv):
        """Domain-randomized maze: uniform number of walls in [0, max_walls], random start and goal,
        resampled until the goal is reachable. Fully determined by the reset seed."""

        def __init__(self, size=13, max_walls=MAX_WALLS, **kw):
            self.max_walls = max_walls
            super().__init__(mission_space=MissionSpace(mission_func=lambda: "get to the green goal square"),
                             grid_size=size, max_steps=4 * size * size, see_through_walls=False, **kw)

        def _gen_grid(self, width, height):
            rng = self.np_random
            cells = [(x, y) for x in range(1, width - 1) for y in range(1, height - 1)]
            if PERFECT:
                walls, start, goal = _perfect_maze(width, height, rng)
            while not PERFECT:
                if CORNERS:
                    start, goal = (1, 1), (width - 2, height - 2)
                    rest = [c for c in cells if c not in (start, goal)]
                    idx = rng.permutation(len(rest))
                    n = int(rng.integers(0, min(self.max_walls, len(rest)) + 1))
                    walls = {rest[i] for i in idx[:n]}
                else:
                    idx = rng.permutation(len(cells))
                    n = int(rng.integers(0, min(self.max_walls, len(cells) - 2) + 1))
                    walls = {cells[i] for i in idx[:n]}
                    start, goal = cells[idx[n]], cells[idx[n + 1]]
                if _reachable(set(cells) - walls, start, goal):
                    break
            self.grid = Grid(width, height)
            self.grid.wall_rect(0, 0, width, height)
            for x, y in walls:
                self.grid.set(x, y, Wall())
            self.put_obj(Goal(), *goal)
            self.agent_pos, self.agent_dir = start, int(rng.integers(4))
            self.mission = "get to the green goal square"

    return RandomMazeEnv


def _perfect_maze(width, height, rng):
    """Randomized depth-first maze on the odd lattice (one route between any two cells), start (1,1), goal at
    the far corner; then a uniform fraction in [0, P_MAX] of the removable walls is knocked out (adds loops)."""
    lattice = [(x, y) for x in range(1, width - 1, 2) for y in range(1, height - 1, 2)]
    open_ = set(lattice)
    stack, seen = [(1, 1)], {(1, 1)}
    while stack:
        x, y = stack[-1]
        nbrs = [(x + dx, y + dy) for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2))
                if (x + dx, y + dy) in open_ and (x + dx, y + dy) not in seen]
        if not nbrs:
            stack.pop()
            continue
        nx, ny = nbrs[int(rng.integers(len(nbrs)))]
        open_.add(((x + nx) // 2, (y + ny) // 2))
        seen.add((nx, ny))
        stack.append((nx, ny))
    interior = {(x, y) for x in range(1, width - 1) for y in range(1, height - 1)}
    removable = sorted(c for c in interior - open_ if (c[0] % 2) != (c[1] % 2))
    k = int(round(float(rng.uniform(0, P_MAX)) * len(removable)))
    for i in rng.permutation(len(removable))[:k]:
        open_.add(removable[i])
    return interior - open_, (1, 1), (width - 2 - (width % 2 == 0), height - 2 - (height % 2 == 0))


def _reachable(free, s, g):
    seen, stack = {s}, [s]
    while stack:
        x, y = stack.pop()
        if (x, y) == g:
            return True
        for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nb in free and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return False


_ENV_CLS = None


def make_maze(size, max_walls=None):
    global _ENV_CLS
    _ENV_CLS = _ENV_CLS or _maze_cls()
    return _ENV_CLS(size=size, max_walls=max_walls or MAX_WALLS)


@dataclass
class MazeLayout:
    n: int        # wall-count tercile, used only for per-cell reporting
    seed: int
    size: int
    sig: tuple
    features: dict = field(default_factory=dict)


def maze_objects(size, seed, max_walls=None):
    env = make_maze(size, max_walls)
    env.reset(seed=seed)
    walls = {(x, y) for x in range(1, size - 1) for y in range(1, size - 1)
             if env.grid.get(x, y) is not None and env.grid.get(x, y).type == "wall"}
    goal = next((x, y) for x in range(size) for y in range(size)
                if env.grid.get(x, y) is not None and env.grid.get(x, y).type == "goal")
    start = tuple(int(v) for v in env.agent_pos)
    env.close()
    return walls, start, goal


def maze_features(size, seed, max_walls=None):
    """Same three-layer descriptor as the LavaCrossing grader (no wall count given explicitly)."""
    from karel_curriculum_difficulty_predictor import (bfs_distances, bridges_and_articulations,
                                                      build_grid_graph, graph_laplacian_eigs)
    walls, start, goal = maze_objects(size, seed, max_walls)
    free = [(x, y) for x in range(1, size - 1) for y in range(1, size - 1) if (x, y) not in walls]
    g = build_grid_graph(free)
    deg = np.array([len(v) for v in g.values()], float)
    br, art = bridges_and_articulations(g)
    sd = bfs_distances(g, start)
    eig = graph_laplacian_eigs(g, k=6)
    f = dict(num_free=len(free), mean_degree=float(deg.mean()), deadends=int((deg <= 1).sum()),
             corridor=int((deg == 2).sum()), branch=int((deg >= 3).sum()), bridges=br, articulation=art,
             start_ecc=max(sd.values()), sg_dist=sd[goal],
             sg_excess=sd[goal] - (abs(start[0] - goal[0]) + abs(start[1] - goal[1])))
    f.update({f"lam{i + 2}": float(e) for i, e in enumerate(eig[1:6])})
    f["walls"] = len(walls)
    return f


GRADER_FEATS = ["num_free", "mean_degree", "deadends", "corridor", "branch", "bridges", "articulation",
                "start_ecc", "sg_dist", "sg_excess", "lam2", "lam3", "lam4", "lam5", "lam6"]


def maze_pool(sizes, tiers=None, n_seeds=90, cap_per_cell=None, seed_start=0, exclude_sigs=None, max_walls=None):
    """n_seeds distinct solvable mazes per size (tiers/cap unused; kept for harness compatibility)."""
    sizes = [sizes] if isinstance(sizes, int) else sizes
    mw = max_walls or MAX_WALLS
    seen, pool = set(exclude_sigs or ()), []
    for size in sizes:
        s, got = seed_start, 0
        while got < n_seeds:
            walls, start, goal = maze_objects(size, s, mw)
            sig = (size, tuple(sorted(walls)), start, goal)
            if sig not in seen:
                seen.add(sig)
                f = maze_features(size, s, mw)
                pool.append(MazeLayout(n=0 if PERFECT else min(2, 3 * f["walls"] // (mw + 1)), seed=s, size=size,
                                       sig=sig, features=f))
                got += 1
            s += 1
    return pool


def maze_wrap(size, n, restrict):
    from minigrid.wrappers import ImgObsWrapper
    return ch.maybe_restrict_actions(ImgObsWrapper(make_maze(size)), restrict)


def _calib_size(args):
    return CALIB_SIZE or args.sizes[0]


def _calib_walls():
    return CALIB_WALLS or MAX_WALLS


def _calib_path(args, ref=None):
    tag = f"maze_calib_s{_calib_size(args)}_" + (f"perfect_p{CALIB_P or P_MAX}" if PERFECT else f"w{_calib_walls()}")
    return ch.CACHE_DIR / (f"{tag}_ref{ref}.json" if ref is not None else f"{tag}.json")


def maze_scores(condition, train_pool, args, restrict):
    rng = np.random.default_rng(10_007 + args.seed)
    jitter = {lo.sig: 1e-6 * float(rng.random()) for lo in train_pool}   # seeded tiebreak
    if condition in ("flat", "random_curriculum"):
        r = np.random.default_rng(args.seed)
        return {lo.sig: float(r.random()) for lo in train_pool}
    if condition == "wall_count":
        return {lo.sig: lo.features["walls"] + jitter[lo.sig] for lo in train_pool}
    if condition == "path_len":
        return {lo.sig: lo.features["sg_dist"] + jitter[lo.sig] for lo in train_pool}
    if condition == "grader":
        from sklearn.ensemble import RandomForestRegressor
        global P_MAX
        calib = json.loads(_calib_path(args).read_text())
        main_p, P_MAX = P_MAX, (CALIB_P or P_MAX)   # regenerate the calibration pool exactly as it was labelled
        cpool = maze_pool([_calib_size(args)], n_seeds=calib["n"], seed_start=CALIB_SEED_START,
                          max_walls=_calib_walls())
        P_MAX = main_p
        X = np.array([[lo.features[k] for k in GRADER_FEATS] for lo in cpool], float)
        y = np.array(calib["label"], float)
        rf = RandomForestRegressor(300, random_state=0, n_jobs=1).fit(X, y)
        Z = np.array([[lo.features[k] for k in GRADER_FEATS] for lo in train_pool], float)
        return {lo.sig: float(p) + jitter[lo.sig] for lo, p in zip(train_pool, rf.predict(Z))}
    raise ValueError(condition)


def build_calib_ref(args, ref, n=90, steps=1_500_000, episodes=20):
    """One reference PPO trained flat on the separate calibration pool; writes 1 - success per maze."""
    restrict = True
    cpool = maze_pool([_calib_size(args)], n_seeds=n, seed_start=CALIB_SEED_START, max_walls=_calib_walls())
    env = ch.CurriculumPoolEnv(restrict, rng_seed=777 + ref)
    env.set_active(cpool)
    model = ch.make_ppo(env, seed=777 + ref)
    model.learn(total_timesteps=steps)
    lab = [1 - r for r in ch.eval_layouts(model, cpool, episodes, restrict)]
    _calib_path(args, ref).write_text(json.dumps(dict(n=n, label=lab)))
    print(f"[calib] ref {ref}: mean success {1 - np.mean(lab):.3f}", flush=True)


def merge_calib(args, refs=(0, 1, 2)):
    from scipy.stats import spearmanr
    acc = [json.loads(_calib_path(args, r).read_text())["label"] for r in refs]
    label = np.mean(acc, 0)
    out = dict(n=len(label), label=label.tolist(), per_ref=acc,
               ref0_vs_rest_spearman=float(spearmanr(acc[0], np.mean(acc[1:], 0))[0]),
               tied_at_1=float(np.mean(label >= 0.999)), mean_success=float(1 - label.mean()))
    _calib_path(args).write_text(json.dumps(out))
    print(f"[calib] merged -> {_calib_path(args).name}: tied_at_1={out['tied_at_1']:.2f} "
          f"ref0-vs-rest rho={out['ref0_vs_rest_spearman']:.2f} mean success={out['mean_success']:.3f}", flush=True)


CONDITIONS = ["flat", "random_curriculum", "wall_count", "path_len", "grader"]


def install():
    ch.wrap = maze_wrap
    ch.enumerate_pool = maze_pool
    ch.initial_scores = maze_scores
    ch.CONDITIONS = CONDITIONS
    ch.FLAT_CONDITIONS = {"flat"}
    ch.SCORING_METHOD.update({
        "wall_count": "interior wall count (+ seeded tiebreak)", "path_len": "BFS start-goal distance (+ tiebreak)",
        "grader": "RF(300) on maze GRADER_FEATS, fit on reference-PPO labels of a separate calibration pool"})


def _pop_flag(name, cast=int):
    if name not in sys.argv:
        return None
    i = sys.argv.index(name)
    v = cast(sys.argv[i + 1])
    del sys.argv[i:i + 2]
    return v


if __name__ == "__main__":
    install()
    MAX_WALLS = _pop_flag("--max-walls") or MAX_WALLS
    if "--corners" in sys.argv:
        sys.argv.remove("--corners")
        CORNERS = True
    if "--perfect" in sys.argv:
        sys.argv.remove("--perfect")
        PERFECT = True
        P_MAX = _pop_flag("--p-max", float) or P_MAX
        CALIB_P = _pop_flag("--calib-p", float)
    CALIB_SIZE = _pop_flag("--calib-size")
    CALIB_WALLS = _pop_flag("--calib-walls")
    calib_ref = _pop_flag("--calib-ref")
    if calib_ref is not None or "--merge-calib" in sys.argv:
        import argparse
        sizes = [_pop_flag("--sizes") or 13]
        ns = argparse.Namespace(sizes=sizes)
        if calib_ref is not None:
            build_calib_ref(ns, calib_ref)
        else:
            merge_calib(ns)
    else:
        ch.main()
