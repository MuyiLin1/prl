"""Feature-layer ablation: does each descriptor layer (local -> +reachability -> +spectral) add predictive power?
Same RF, repeated 5-fold CV over several split seeds, on every spatial domain plus gated Karel tasks.
Layers are defined the same way everywhere:
  local  = cell counts and per-cell degree statistics (free/wall/hazard counts, degree, dead-end/corridor/branch)
  reach  = global path quantities (components, diameter, pair distance, reachability, eccentricity,
           start-goal distance, bridges, articulation points)
  spec   = normalized-Laplacian eigenvalues"""
import glob, io, json, contextlib, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts/curriculum")]
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

REACH_KEYS = ("component", "diameter", "pair_distance", "reachable", "eccentricity", "ecc", "sg_dist",
              "sg_excess", "start_goal", "bridges", "articulation")
SPEC_KEYS = ("lambda", "laplacian_eig", "lam")


def layer(col):
    c = col.lower()
    if any(k in c for k in SPEC_KEYS):
        return "spec"
    if any(k in c for k in REACH_KEYS):
        return "reach"
    return "local"


def cv_score(X, y, reps, classify=False):
    out = []
    for s in range(reps):
        p = np.zeros(len(y))
        split = (StratifiedKFold if classify else KFold)(5, shuffle=True, random_state=s)
        for tr, te in split.split(X, y):
            M = RandomForestClassifier if classify else RandomForestRegressor
            m = M(300, random_state=s, n_jobs=4).fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1] if classify else m.predict(X[te])
        out.append(roc_auc_score(y, p) if classify else spearmanr(p, y)[0])
    return float(np.mean(out)), float(np.std(out))


def sets_for(cols, extra=None):
    g = {c: layer(c) for c in cols}
    L = [c for c in cols if g[c] == "local"]
    R = [c for c in cols if g[c] == "reach"]
    S = [c for c in cols if g[c] == "spec"]
    out = {"local": L, "local+reach": L + R, "local+reach+spec": L + R + S, "spec only": S}
    if extra:
        out["full + domain-specific"] = L + R + S + extra
    return {k: v for k, v in out.items() if v}


ROWS = []


def run(domain, label, df, cols, y, reps, classify=False, extra=None):
    cols = [c for c in cols if df[c].std() > 1e-9]
    extra = [c for c in (extra or []) if df[c].std() > 1e-9]
    for name, fs in sets_for(cols, extra).items():
        m, s = cv_score(df[fs].values.astype(float), y, reps, classify)
        ROWS.append(dict(domain=domain, label=label, features=name, n_feats=len(fs), mean=m, sd=s,
                         metric="AUC" if classify else "Spearman"))
        print(f"{domain:28s} {label:22s} {name:24s} ({len(fs):2d}) {m:+.3f} +- {s:.3f}", flush=True)


def sokoban():
    b = pd.read_csv(ROOT / "data/benchmarks/boxoban_astar/benchmark_levels.csv")
    from benchmarks.sokoban_astar_benchmark import REWARD_FEATS
    meta = {"level_id", "tier", "astar_search_steps", "astar_solution_len", "ascii"}
    feats = [c for c in b.columns if c not in meta and not c.startswith("kartal")]
    geo = [c for c in feats if c not in REWARD_FEATS]
    rew = [c for c in feats if c in REWARD_FEATS]
    run("Sokoban public (3000)", "log A* effort", b, geo, np.log10(b.astar_search_steps.values), 3, extra=rew)
    run("Sokoban public (3000)", "tier", b, geo, (b.tier == "medium").astype(int).values, 3, True, extra=rew)


def lavagap():
    lg = pd.read_csv(ROOT / "direction_c_AB.csv")
    sys.argv = ["x"]
    from importlib.machinery import SourceFileLoader
    with contextlib.redirect_stdout(io.StringIO()):
        ldm = SourceFileLoader("ld", str(ROOT / "notes/rescue_audit_2026-10-01/lavagap_descriptor.py")).load_module()
    run("LavaGap (60)", "partial-progress D", ldm.D, list(ldm.D.columns), lg.diff_A_partial.values, 10)


def lavacrossing():
    from difficulty_scoring import GRADER_FEATURES, descriptor_features, enumerate_pool
    D = ROOT / "notes/rescue_audit_2026-10-01/derisk_out"
    for tag, sizes in [("p7911", [7, 9, 11]), ("p91317", [9, 13, 17])]:
        runs = [json.load(open(f)) for f in sorted(glob.glob(f"{D}/{tag}_ref*.json"))]
        y = np.mean([[1 - np.mean(r["succ"]) for r in run["rows"]] for run in runs], 0)
        pool = enumerate_pool(sizes, [1, 2, 3], 80, 10)
        by_key = {(lo.size, lo.n, lo.seed): lo for lo in pool}
        pool = [by_key[(r["size"], r["n"], r["seed"])] for r in runs[0]["rows"]]  # align features to label rows
        X = pd.DataFrame([{k: descriptor_features(lo.size, lo.n, lo.seed)[k] for k in GRADER_FEATURES} for lo in pool])
        run(f"LavaCrossing {tag[1:]} (90)", "1 - PPO success", X, GRADER_FEATURES, y, 10)


def karel():
    for t in ["WallAvoider", "TopOff", "DoorKey", "Maze", "Snake"]:
        f = ROOT / ("averaged_label_topoff200.csv" if t == "TopOff" else
                    "averaged_label_screen_snake_seeder.csv" if t == "Snake" else f"data/karel_difficulty/karel_{t}_grids.csv")
        if not f.exists():
            f = Path(glob.glob(str(ROOT / f"data/karel_difficulty/karel_{t}_grids.csv"))[0])
        g = pd.read_csv(f)
        g = g[g.task == t] if "task" in g.columns else g
        cols = [c for c in g.columns if c.startswith(("graph_", "spectral_", "grid_"))]
        rew = [c for c in g.columns if c.startswith("reward_")]
        run(f"Karel {t} ({len(g)})", "HC D (R=12)", g, cols, g.mean_difficulty.values, 10, extra=rew)


if __name__ == "__main__":
    for fn in [lavacrossing, karel, sokoban] if "--skip-lavagap" in sys.argv else [lavagap, lavacrossing, karel, sokoban]:
        fn()
    out = pd.DataFrame(ROWS)
    out.to_csv(ROOT / "data/benchmarks/feature_ablation.csv", index=False)
    print("\nwrote data/benchmarks/feature_ablation.csv")
