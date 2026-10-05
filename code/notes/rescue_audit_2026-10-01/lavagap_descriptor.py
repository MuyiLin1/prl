"""F7 check: does the paper's THREE-LAYER descriptor (not hand gap features) predict shaped
LavaGap difficulty? Labels: direction_c_AB.csv (60 grids, R=12). Same RF/CV protocol as
scripts/direction_c_AB.py. Descriptor helpers are the shared Karel/Sokoban ones."""
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = "/Users/linmuyi/code/temp-prl"
sys.path[:0] = [ROOT, ROOT + "/scripts", ROOT + "/leaps"]
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from prog_policies.utils import get_env_name  # noqa: F401 (circular-import order)
from prog_policies.minigrid_tasks.lavagap import LavaGapEnv_modified
from karel_curriculum_difficulty_predictor import (bfs_distances, bridges_and_articulations,
                                                  build_grid_graph, connected_components,
                                                  graph_laplacian_eigs)
from option_a_lavagap import lavagap_features


def descriptor(seed, size=9):
    env = LavaGapEnv_modified(size=size)
    env.reset(seed=seed)
    g = env.grid
    free, lava, walls = [], 0, 0
    for x in range(g.width):
        for y in range(g.height):
            c = g.get(x, y)
            t = None if c is None else c.type
            if t == "wall":
                walls += 1
            elif t == "lava":
                lava += 1
            else:
                free.append((x, y))
    graph = build_grid_graph(free)
    comps = connected_components(graph)
    deg = np.array([len(v) for v in graph.values()], float)
    br, art = bridges_and_articulations(graph)
    start = tuple(int(v) for v in env.agent_pos)
    goal = tuple(int(v) for v in env.goal_pos)
    sd = bfs_distances(graph, start)
    diam, acc, cnt = 0, 0.0, 0
    for n in graph:
        d = bfs_distances(graph, n)
        diam = max(diam, max(d.values()))
        acc += sum(d.values())
        cnt += len(d)
    eigs = graph_laplacian_eigs(graph, k=6)
    f = dict(num_free_cells=len(free), num_lava=lava, num_wall_cells=walls,
             num_edges=deg.sum() / 2, mean_degree=deg.mean(), min_degree=deg.min(), max_degree=deg.max(),
             num_deadends=(deg <= 1).sum(), num_corridor=(deg == 2).sum(), num_branch=(deg >= 3).sum(),
             num_components=len(comps), graph_diameter=diam, mean_pair_distance=acc / max(1, cnt),
             start_reachable=len(sd), start_eccentricity=max(sd.values()), num_bridges=br,
             num_articulation_points=art, start_goal_dist=sd.get(goal, -1))
    for i, e in enumerate(eigs):
        f[f"spectral_lambda_{i + 1}"] = e
    return f


def rf_cv(X, y):
    X = X[:, X.std(0) > 1e-9]
    oof = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0).fit(X[tr], y[tr])
        oof[te] = rf.predict(X[te])
    return spearmanr(oof, y)[0]


lab = pd.read_csv(ROOT + "/direction_c_AB.csv")
D = pd.DataFrame([descriptor(s) for s in lab.grid_seed])
H = pd.DataFrame([lavagap_features(s, 9) for s in lab.grid_seed])
spec = [c for c in D.columns if c.startswith("spectral")]
print("descriptor features varying across grids:", [c for c in D.columns if D[c].std() > 1e-9])
for y in ["diff_A_partial", "diff_B1_successrate", "diff_B2_evals"]:
    v = lab[y].values
    best = max(((c, spearmanr(D[c], v)[0]) for c in D.columns if D[c].std() > 1e-9), key=lambda t: abs(t[1]))
    print(f"{y:22s} hand-gap RF CV {rf_cv(H.values.astype(float), v):+.2f} | 3-layer RF CV "
          f"{rf_cv(D.values.astype(float), v):+.2f} | spectral-only {rf_cv(D[spec].values, v):+.2f} "
          f"| best single {best[0]} {best[1]:+.2f}")
