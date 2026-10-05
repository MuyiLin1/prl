"""Same-dataset benchmark on Boxoban with PUBLISHED difficulty labels.

Labels (no solver run by us):
  * A* search effort and optimal solution length per level, from the public
    AlignmentResearch/boxoban-astar-solutions dataset (heuristic = sum of box->nearest-target
    Manhattan distances; budget 1M nodes unfiltered / 5M medium).
  * Boxoban tier (unfiltered vs medium). Guez et al. (2019) built `medium` by keeping levels that
    partially-trained DRC agents FAILED, so tier membership is an RL-agent difficulty label.

Predictors scored on identical levels:
  * ours_geometry : three-layer layout descriptor (local, reachability, spectral) -> RF, 5-fold CV
  * ours_full     : descriptor + Sokoban reward-structure features -> RF, 5-fold CV
  * kartal2016    : published Sokoban difficulty function f = (5*P_b + 10*P_c + n)/50
                    (Kartal, Sohre & Guy, AIIDE 2016), unfitted; alpha=beta=gamma=1 (their tuned
                    values are not reported). Also its components refit with an RF (kartal_rf).
  * box_count / grid baselines are constant in Boxoban (always 4 boxes, 10x10) -> reported as such.

Writes data/benchmarks/boxoban_astar/benchmark_levels.csv (features + labels + ASCII) so the LLM
judge and CNN baselines score exactly the same levels.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "sokoban")]
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from sokoban_core import extract_features, hungarian_min_cost, iter_levels_in_file

OUT = ROOT / "data" / "benchmarks" / "boxoban_astar"
N_PER_TIER = 1500
REWARD_FEATS = {"num_boxes", "num_targets", "boxes_on_target_init", "box_target_dist_sum",
                "box_target_dist_mean", "box_target_dist_max", "box_target_matching_lb",
                "num_dead_cells", "boxes_in_corner_init", "player_to_nearest_box"}


def kartal(level):
    """Kartal et al. 2016: f = (w_b*P_b + w_c*P_c + w_n*n)/k with w_b=5, w_c=10, w_n=1, k=50."""
    rows = level.grid
    H, W = len(rows), len(rows[0])
    # P_b (3x3 Blocks): rewards heterogeneous terrain -> count 3x3 windows that are NOT uniform
    pb = 0
    for y in range(H - 2):
        for x in range(W - 2):
            cells = {("#" if rows[y + dy][x + dx] == "#" else ".") for dy in range(3) for dx in range(3)}
            pb += len(cells) > 1
    # P_c (Congestion v2): sum_i (a*b_i + b*g_i) / (c*(A_i - o_i)) over box->assigned-goal rectangles
    boxes, goals = sorted(level.boxes), sorted(level.targets)
    cost = np.array([[abs(b[0] - g[0]) + abs(b[1] - g[1]) for g in goals] for b in boxes], float)
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(cost)
    pc = 0.0
    for i, j in zip(r, c):
        (bx, by), (gx, gy) = boxes[i], goals[j]
        x0, x1, y0, y1 = min(bx, gx), max(bx, gx), min(by, gy), max(by, gy)
        rect = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]
        A = len(rect)
        o = sum((x, y) in level.walls for x, y in rect)
        bi = sum((x, y) in level.boxes for x, y in rect)
        gi = sum((x, y) in level.targets for x, y in rect)
        pc += (bi + gi) / max(1, (A - o))
    n = len(boxes)
    return dict(kartal_Pb=pb, kartal_Pc=pc, kartal_n=n, kartal2016=(5 * pb + 10 * pc + n) / 50)


def load(tier, astar_csv):
    a = pd.read_csv(OUT / astar_csv, dtype=str)
    a = a[a.Steps.str.isdigit() & a.SearchSteps.str.isdigit()].copy()  # drop unsolved / failed entries
    a["SearchSteps"] = a.SearchSteps.astype(int)
    a["Steps"] = a.Steps.astype(int)
    a = a.sample(N_PER_TIER, random_state=0)
    rows = []
    for f, grp in a.groupby("File"):
        lv = {l.level_id.split("/")[-1]: l for l in
              iter_levels_in_file(ROOT / "data" / "boxoban" / tier / "valid" / f"{f}.txt", tier)}
        for _, r in grp.iterrows():
            L = lv[str(int(r.Level))]
            feats = extract_features(L)
            feats.update(kartal(L))
            rows.append(dict(level_id=f"{tier}/valid/{f}/{int(r.Level)}", tier=tier,
                             astar_search_steps=r.SearchSteps, astar_solution_len=r.Steps,
                             ascii="\n".join(L.grid), **feats))
    return rows


def rf_cv(X, y, classify=False):
    p = np.zeros(len(y))
    split = (StratifiedKFold if classify else KFold)(5, shuffle=True, random_state=0)
    for tr, te in split.split(X, y):
        if classify:
            p[te] = RandomForestClassifier(300, random_state=0, n_jobs=2).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        else:
            p[te] = RandomForestRegressor(300, random_state=0, n_jobs=2).fit(X[tr], y[tr]).predict(X[te])
    return p


def boot_ci(f, y, p, B=2000):
    rng = np.random.default_rng(0)
    vals = [f(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(B))]
    return np.percentile(vals, [2.5, 97.5])


def main():
    path = OUT / "benchmark_levels.csv"
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(load("unfiltered", "unfiltered_valid.csv.gz") + load("medium", "medium_valid.csv.gz"))
        df.to_csv(path, index=False)
    meta = {"level_id", "tier", "astar_search_steps", "astar_solution_len", "ascii"}
    kart = [c for c in df.columns if c.startswith("kartal")]
    feat_all = [c for c in df.columns if c not in meta and c not in kart and df[c].std() > 1e-9]
    feat_geo = [c for c in feat_all if c not in REWARD_FEATS]
    print(f"{len(df)} levels ({(df.tier == 'medium').sum()} medium); constant features dropped: "
          f"{sorted(c for c in df.columns if c not in meta and df[c].std() <= 1e-9)}")
    labels = {"log A* nodes expanded": np.log10(df.astar_search_steps.values),
              "A* optimal solution length": df.astar_solution_len.values.astype(float)}
    tier = (df.tier == "medium").astype(int).values
    preds = {
        "ours_geometry (RF, CV)": lambda y, cl: rf_cv(df[feat_geo].values, y, cl),
        "ours_full (RF, CV)": lambda y, cl: rf_cv(df[feat_all].values, y, cl),
        "kartal2016 (published, unfitted)": lambda y, cl: df.kartal2016.values,
        "kartal2016 components (RF, CV)": lambda y, cl: rf_cv(df[["kartal_Pb", "kartal_Pc"]].values, y, cl),
    }
    res = []
    for lname, y in labels.items():
        for pname, fn in preds.items():
            p = fn(y, False)
            rho = spearmanr(p, y)[0]
            lo, hi = boot_ci(lambda a, b: spearmanr(a, b)[0], y, p)
            res.append(dict(label=lname, predictor=pname, metric="Spearman", value=rho, ci_lo=lo, ci_hi=hi))
    for pname, fn in preds.items():
        p = fn(tier, True)
        auc = roc_auc_score(tier, p)
        lo, hi = boot_ci(roc_auc_score, tier, p)
        res.append(dict(label="Boxoban tier (medium = DRC agents failed)", predictor=pname, metric="AUC",
                        value=auc, ci_lo=lo, ci_hi=hi))
    r = pd.DataFrame(res)
    r.to_csv(OUT / "benchmark_results_layout_methods.csv", index=False)
    print(r.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
