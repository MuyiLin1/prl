"""Analyze derisk_oracle outputs: reliability of reference-PPO difficulty labels on LavaCrossing,
overall and WITHIN (size, N) cells, and whether a knob-free three-layer descriptor predicts them."""
import glob
import json
import sys
import warnings
from itertools import combinations

warnings.filterwarnings("ignore")
ROOT = "/Users/linmuyi/code/temp-prl"
sys.path[:0] = [ROOT, ROOT + "/scripts/curriculum"]
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from difficulty_scoring import LAVA, WALL, _grid_objects
from karel_curriculum_difficulty_predictor import (bfs_distances, bridges_and_articulations,
                                                  build_grid_graph, graph_laplacian_eigs)

OUT = "/Users/linmuyi/code/temp-prl/notes/rescue_audit_2026-10-01/derisk_out"


def descriptor(size, n, seed):
    img, start, goal = _grid_objects(size, n, seed)
    free = [(x, y) for x in range(img.shape[0]) for y in range(img.shape[1]) if img[x, y] not in (WALL, LAVA)]
    g = build_grid_graph(free)
    deg = np.array([len(v) for v in g.values()], float)
    br, art = bridges_and_articulations(g)
    sd = bfs_distances(g, start)
    eig = graph_laplacian_eigs(g, k=6)
    f = dict(num_free=len(free), lava_frac=float((img == LAVA).sum()) / len(free), mean_degree=deg.mean(),
             deadends=(deg <= 1).sum(), corridor=(deg == 2).sum(), branch=(deg >= 3).sum(),
             bridges=br, articulation=art, start_ecc=max(sd.values()), sg_dist=sd.get(goal, -1),
             sg_excess=sd.get(goal, -1) - (abs(start[0] - goal[0]) + abs(start[1] - goal[1])))
    f.update({f"lam{i + 2}": e for i, e in enumerate(eig[1:])})
    return f


def rf_oof(X, y):
    p = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        p[te] = RandomForestRegressor(300, random_state=0, n_jobs=4).fit(X[tr], y[tr]).predict(X[te])
    return p


def sb(r, k):  # Spearman-Brown
    return k * r / (1 + (k - 1) * r)


for tag in ["p7911", "p91317"]:
    files = sorted(glob.glob(f"{OUT}/{tag}_ref*.json"))
    if not files:
        continue
    runs = [json.load(open(f)) for f in files]
    base = pd.DataFrame(runs[0]["rows"])[["size", "n", "seed", "sig", "detour", "n_lava"]]
    K = len(runs)
    S = np.array([[1 - np.mean(r["succ"]) for r in run["rows"]] for run in runs]).T   # layouts x seeds
    P = np.array([[1 - np.mean(r["prog"]) for r in run["rows"]] for run in runs]).T
    cell = base["size"].astype(str) + "N" + base["n"].astype(str)
    print(f"\n================ {tag}: {K} ref seeds x {len(base)} layouts, "
          f"train {np.mean([r['train_s'] for r in runs]):.0f}s/seed ================")
    for name, L in [("1-success", S), ("1-progress", P)]:
        m = L.mean(1)
        tied = np.mean(m >= 0.999)
        pair = [spearmanr(L[:, a], L[:, b])[0] for a, b in combinations(range(K), 2)]
        r_ = np.nanmean(pair)
        # within-cell: remove each seed's cell means, then correlate residuals across seeds
        R = L - pd.DataFrame(L).groupby(cell.values).transform("mean").values
        wpair = [spearmanr(R[:, a], R[:, b])[0] for a, b in combinations(range(K), 2)]
        wr = np.nanmean(wpair)
        eta = 1 - ((m - pd.Series(m).groupby(cell.values).transform("mean").values) ** 2).sum() / ((m - m.mean()) ** 2).sum()
        print(f"[{name:10s}] tied@1: {tied:.0%}  distinct: {len(np.unique(np.round(m, 3)))}  "
              f"pairwise rho {r_:+.2f} -> rel(K={K}) {sb(r_, K):+.2f} | WITHIN-cell pairwise {wr:+.2f} -> rel {sb(wr, K):+.2f}"
              f" | eta^2(cell) {eta:.2f}")
    if tag == "p7911":
        cache = json.load(open(ROOT + "/scripts/curriculum/cache/oracle_6ccc84561c_r3e20.json"))
        c = np.array([cache[s] for s in base.sig])
        m = S.mean(1)
        Rm = m - pd.Series(m).groupby(cell.values).transform("mean").values
        Rc = c - pd.Series(c).groupby(cell.values).transform("mean").values
        print(f"test-retest vs cached r3e20 oracle (independent 3-seed build): rho {spearmanr(m, c)[0]:+.2f}, "
              f"within-cell rho {spearmanr(Rm, Rc)[0]:+.2f}")
    # descriptor prediction (knob-free: no size, no n), label = mean 1-progress and 1-success
    D = pd.DataFrame([descriptor(*t) for t in zip(base["size"], base["n"], base.seed)])
    X = D.loc[:, D.std() > 1e-9].values.astype(float)
    Kn = base[["size", "n"]].values.astype(float)
    for name, L in [("1-success", S), ("1-progress", P)]:
        y = L.mean(1)
        yr = y - pd.Series(y).groupby(cell.values).transform("mean").values
        p = rf_oof(X, y)
        pk = rf_oof(Kn, y)
        pr = rf_oof(X, yr)
        print(f"  [{name:10s}] CV rho: knob-free descriptor {spearmanr(p, y)[0]:+.2f} | knobs (size,N) only "
              f"{spearmanr(pk, y)[0]:+.2f} | descriptor on WITHIN-cell residual {spearmanr(pr, yr)[0]:+.2f}")
