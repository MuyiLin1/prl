"""Few-labels test (Sokoban 3000): our features + RF vs CNN on the raw grid, trained on n labeled levels,
scored on one fixed held-out set of 1000 levels. Does our method need fewer labels than a generic model?"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from cnn_baseline import dihedral, encode, fit_predict

META = {"level_id", "tier", "astar_search_steps", "astar_solution_len", "ascii"}


def main():
    b = pd.read_csv(ROOT / "data/benchmarks/boxoban_astar/benchmark_levels.csv")
    feats = [c for c in b.columns if c not in META and not c.startswith("kartal") and b[c].std() > 1e-9]
    X, A = b[feats].values, dihedral(np.stack([encode(s) for s in b.ascii]))
    tasks = [("log A* nodes expanded", np.log10(b.astar_search_steps.values).astype(np.float32), False),
             ("Boxoban tier (AUC)", (b.tier == "medium").astype(np.float32).values, True)]
    perm = np.random.default_rng(123).permutation(len(b))
    test, pool = perm[:1000], perm[1000:]
    rows = []
    for n in [60, 200, 500, 1000, 2000]:
        for rep in range(5 if n < 2000 else 1):  # n=2000 is the whole pool: one draw
            tr = np.random.default_rng(1000 * n + rep).choice(pool, n, replace=False)
            for label, y, classify in tasks:
                score = (lambda p: roc_auc_score(y[test], p)) if classify else (lambda p: spearmanr(p, y[test])[0])
                Model = RandomForestClassifier if classify else RandomForestRegressor
                rf = Model(300, random_state=rep, n_jobs=2).fit(X[tr], y[tr])
                p_rf = rf.predict_proba(X[test])[:, 1] if classify else rf.predict(X[test])
                p_cnn = fit_predict(A, y, tr, test, classify, seed=rep)
                rows += [dict(n=n, rep=rep, label=label, method="ours (RF)", value=score(p_rf)),
                         dict(n=n, rep=rep, label=label, method="CNN raw grid", value=score(p_cnn))]
                print(f"n={n:5d} rep{rep} {label:22s} ours={score(p_rf):.3f} cnn={score(p_cnn):.3f}", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(ROOT / "data/benchmarks/few_labels_sokoban.csv", index=False)
    s = r.groupby(["label", "n", "method"]).value.agg(["mean", "std"]).unstack("method")
    print(s.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
