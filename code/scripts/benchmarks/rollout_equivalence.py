"""How many rollouts of a trained agent match the grader's zero-rollout accuracy (LavaCrossing)?
Label: difficulty (1 - success) from INDEPENDENT reference agents. Estimator: k episodes of one held-out
reference agent. Grader: 7/9/11 -> 5-fold out-of-fold predictions; 9/13/17 -> transfer from 7/9/11."""
import glob, json, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT), str(ROOT / "scripts/curriculum")]
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from difficulty_scoring import GRADER_FEATURES, descriptor_features, enumerate_pool, fit_grader
from curriculum_harness import tuple_key
D = ROOT / "notes/rescue_audit_2026-10-01/derisk_out"
rng = np.random.default_rng(0)
rows = []
calib = {tuple_key(k): v for k, v in json.load(open(ROOT / "scripts/curriculum/cache/grader_calib_p7911_r6.json")).items()}
cp = enumerate_pool([7, 9, 11], [1, 2, 3], 80, 10)
for tag, sizes in [("p7911", [7, 9, 11]), ("p91317", [9, 13, 17])]:
    runs = [json.load(open(f)) for f in sorted(glob.glob(f"{D}/{tag}_ref*.json"))]
    pool = enumerate_pool(sizes, [1, 2, 3], 80, 10)
    llm = pd.read_csv(ROOT / f"data/benchmarks/llm_prompts/lavacrossing_{tag}_llm_qwen7b.csv", dtype={"id": str}) \
        .set_index("id").llm_expected.loc[[r["sig"] for r in runs[0]["rows"]]].values
    for held in range(3):                                   # estimator agent; label from the others
        others = [r for i, r in enumerate(runs) if i != held]
        succ_h = np.array([r["succ"] for r in runs[held]["rows"]])          # layouts x 20 episodes
        lab = np.mean([[1 - np.mean(r["succ"]) for r in run["rows"]] for run in others], axis=0)
        if tag == "p7911":                                  # independent 3-seed cached oracle joins the label
            cache = json.load(open(ROOT / "scripts/curriculum/cache/oracle_6ccc84561c_r3e20.json"))
            lab = (2 * lab + 3 * np.array([cache[r["sig"]] for r in runs[0]["rows"]])) / 5
            X = np.array([[descriptor_features(lo.size, lo.n, lo.seed)[k] for k in GRADER_FEATURES] for lo in pool])
            y = np.mean([[1 - np.mean(r["succ"]) for r in run["rows"]] for run in others], axis=0)
            g = np.zeros(len(pool))
            for tr, te in KFold(5, shuffle=True, random_state=0).split(X):   # OOF: never sees its own layout
                g[te] = RandomForestRegressor(300, random_state=0).fit(X[tr], y[tr]).predict(X[te])
        else:
            g = fit_grader(cp, calib, seed=0)(pool)
        knobs = np.array([lo.size + lo.n for lo in pool], float)
        rows.append(dict(pool=tag, held=held, method="grader (0 rollouts)", k=0, rho=spearmanr(g, lab)[0]))
        rows.append(dict(pool=tag, held=held, method="knobs size+N (0 rollouts)", k=0, rho=spearmanr(knobs, lab)[0]))
        rows.append(dict(pool=tag, held=held, method="LLM Qwen7B zero-shot (0 rollouts)", k=0, rho=spearmanr(llm, lab)[0]))
        rows.append(dict(pool=tag, held=held, method="grader + LLM rank-avg (0 rollouts)", k=0,
                         rho=spearmanr(rankdata(g) + rankdata(llm), lab)[0]))
        if tag == "p7911":
            g2 = np.zeros(len(pool))
            for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
                g2[te] = RandomForestRegressor(300, random_state=0).fit(np.c_[X, llm][tr], y[tr]).predict(np.c_[X, llm][te])
            rows.append(dict(pool=tag, held=held, method="grader + LLM (RF, CV, 0 rollouts)", k=0, rho=spearmanr(g2, lab)[0]))
        for k in [1, 2, 4, 8, 16]:
            rs = [spearmanr(1 - succ_h[:, rng.choice(20, k, replace=False)].mean(1), lab)[0] for _ in range(200)]
            rows.append(dict(pool=tag, held=held, method="trained-agent rollouts", k=k, rho=np.nanmean(rs)))
r = pd.DataFrame(rows).groupby(["pool", "method", "k"]).rho.mean().reset_index()
print(r.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
r.to_csv(ROOT / "data/benchmarks/rollout_equivalence_lavacrossing.csv", index=False)
