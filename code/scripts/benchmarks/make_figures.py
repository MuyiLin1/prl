"""Figure: what a free prediction is worth. (a) grader vs k runs of the expensive reference; (b) grader vs CNN by #labels."""
import io, contextlib, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold


def search_curve(S, X, ks=(1, 2, 4, 6), splits=40):
    """Label = mean of 6 held-out runs; estimator = mean of k of the other 6; grader = OOF RF on the label."""
    rng = np.random.default_rng(0)
    res, gr = {k: [] for k in ks}, []
    for t in range(splits):
        p = rng.permutation(S.shape[1]); A, B = p[:6], p[6:]
        lab = S[:, A].mean(1)
        for k in ks:
            res[k].append(spearmanr(S[:, rng.choice(B, k, replace=False)].mean(1), lab)[0])
        o = np.zeros(len(lab))
        for tr, te in KFold(5, shuffle=True, random_state=t).split(X):
            o[te] = RandomForestRegressor(300, random_state=t, n_jobs=4).fit(X[tr], lab[tr]).predict(X[te])
        gr.append(spearmanr(o, lab)[0])
    return list(ks), [np.nanmean(res[k]) for k in ks], float(np.mean(gr))


def main():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    roll = pd.read_csv(ROOT / "data/benchmarks/rollout_equivalence_lavacrossing.csv")
    colors = {"LavaCrossing 9/13/17": "C0", "LavaCrossing 7/9/11": "C1", "LavaGap": "C2"}
    for pool, name in [("p91317", "LavaCrossing 9/13/17"), ("p7911", "LavaCrossing 7/9/11")]:
        r = roll[roll.pool == pool]
        k = r[r.method == "trained-agent rollouts"].sort_values("k")
        g = float(r[r.method == "grader (0 rollouts)"].rho.iloc[0])
        a.plot(k.k, k.rho, "o-", color=colors[name], label=f"{name}: agent runs")
        a.axhline(g, color=colors[name], ls="--", lw=1.5)
    sys.argv = ["x"]
    from importlib.machinery import SourceFileLoader
    with contextlib.redirect_stdout(io.StringIO()):
        ldm = SourceFileLoader("ld", str(ROOT / "notes/rescue_audit_2026-10-01/lavagap_descriptor.py")).load_module()
    S = 1 - np.load(ROOT / "direction_c_AB_rerun_per_search.npz")["shaped"]
    X = ldm.D.loc[:, ldm.D.std() > 1e-9].values.astype(float)
    ks, vals, g = search_curve(S, X)
    a.plot(ks, vals, "s-", color=colors["LavaGap"], label="LavaGap: search runs")
    a.axhline(g, color=colors["LavaGap"], ls="--", lw=1.5)
    a.plot([], [], "k--", label="dashed = our grader (0 runs);\nsame colour = same game")
    a.set_xscale("log", base=2); a.set_xticks([1, 2, 4, 8, 16]); a.set_xticklabels(["1", "2", "4", "8", "16"])
    a.set_xlabel("runs of the expensive reference per level ($k$)")
    a.set_ylabel("Spearman with held-out label")
    a.set_title("(a) A free prediction vs. running the reference $k$ times", fontsize=10)
    a.legend(fontsize=7.5, loc="lower right"); a.grid(alpha=0.3)

    f = pd.read_csv(ROOT / "data/benchmarks/few_labels_sokoban.csv")
    f = f[f.label == "log A* nodes expanded"]
    for m, c, mk in [("ours (RF)", "C3", "o"), ("CNN raw grid", "C7", "s")]:
        s = f[f.method == m].groupby("n").value.agg(["mean", "std"]).fillna(0)
        b.errorbar(s.index, s["mean"], yerr=s["std"], fmt=mk + "-", color=c, capsize=3,
                   label="our grader" if m.startswith("ours") else "CNN on raw grid")
    b.set_xscale("log"); b.set_xticks([60, 200, 500, 1000, 2000]); b.set_xticklabels(["60", "200", "500", "1k", "2k"])
    b.set_xlabel("labelled training levels ($n$)")
    b.set_ylabel("Spearman on 1,000 held-out levels")
    b.set_title("(b) Labels needed: grader vs. CNN (Sokoban)", fontsize=10)
    b.legend(fontsize=8, loc="lower right"); b.grid(alpha=0.3)
    fig.tight_layout()
    out = ROOT / "figures"; out.mkdir(exist_ok=True)
    fig.savefig(out / "worth_curves.pdf"); fig.savefig(out / "worth_curves.png", dpi=150)
    print("LavaGap grader", round(g, 3), "search", [round(v, 3) for v in vals])
    print("wrote figures/worth_curves.pdf/.png")


if __name__ == "__main__":
    main()
