"""Headline comparison: our grader vs baselines on identical instances, per domain (Spearman with label;
AUC for Boxoban tier). Ceiling = sqrt(label reliability) where known."""
import glob, json, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts/curriculum"), str(ROOT / "scripts/sokoban")]
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
P = ROOT / "data/benchmarks/llm_prompts"
def llm(name):
    f = P / f"{name}_llm_qwen7b.csv"
    return pd.read_csv(f, dtype={"id": str}).set_index("id").llm_expected if f.exists() else None
def oof(X, y, classify=False):
    p = np.zeros(len(y))
    for tr, te in (StratifiedKFold if classify else KFold)(5, shuffle=True, random_state=0).split(X, y):
        if classify:
            p[te] = RandomForestClassifier(300, random_state=0, n_jobs=2).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        else:
            p[te] = RandomForestRegressor(300, random_state=0, n_jobs=2).fit(X[tr], y[tr]).predict(X[te])
    return p
rho = lambda a, b: spearmanr(a, b)[0]
R = []
def add(domain, label, method, v, ceil=None, metric="Spearman"):
    R.append(dict(domain=domain, label=label, method=method, metric=metric, value=v, ceiling=ceil))
# Sokoban public benchmark
b = pd.read_csv(ROOT / "data/benchmarks/boxoban_astar/benchmark_levels.csv")
br = pd.read_csv(ROOT / "data/benchmarks/boxoban_astar/benchmark_results_layout_methods.csv")
for _, r in br.iterrows():
    if r.predictor.startswith(("ours_full", "kartal2016 (published")):
        add("Sokoban (3000 public)", r.label, r.predictor, r.value, 1.0, r.metric)
L = llm("sokoban_astar_benchmark")
if L is not None:
    m = b.set_index("level_id").join(L, how="inner")
    add("Sokoban (3000 public)", "log A* nodes expanded", "LLM Qwen7B zero-shot", rho(m.llm_expected, np.log10(m.astar_search_steps)), 1.0)
    add("Sokoban (3000 public)", "Boxoban tier (medium = DRC agents failed)", "LLM Qwen7B zero-shot",
        roc_auc_score(m.tier == "medium", m.llm_expected), 1.0, "AUC")
    feats = [c for c in b.columns if c not in {"level_id", "tier", "astar_search_steps", "astar_solution_len", "ascii"}
             and not c.startswith("kartal") and b[c].std() > 1e-9]
    X1 = np.c_[m[feats].values, m.llm_expected.values]
    print(f"Sokoban 3000: LLM covers {len(m)}/{len(b)} levels")
    for lname, y in [("log A* nodes expanded", np.log10(m.astar_search_steps.values)),
                     ("A* optimal solution length", m.astar_solution_len.values.astype(float))]:
        add("Sokoban (3000 public)", lname, "ours + LLM (RF, CV)", rho(oof(X1, y), y), 1.0)
    t = (m.tier == "medium").astype(int).values
    add("Sokoban (3000 public)", "Boxoban tier (medium = DRC agents failed)", "ours + LLM (RF, CV)",
        roc_auc_score(t, oof(X1, t, classify=True)), 1.0, "AUC")
# Sokoban RL pool (our solver label)
from sokoban_core import extract_features, iter_levels_in_file
pool = pd.read_csv(ROOT / "data/sokoban_fastpath_labels.csv")
cache, F = {}, []
for lid in pool.level_id:
    t, f, i = lid.split("/"); path = ROOT / "data/boxoban" / t / ("" if t == "hard" else "train") / f"{f}.txt"
    cache.setdefault(path, {l.level_id.split("/")[-1]: l for l in iter_levels_in_file(path, t)})
    F.append(extract_features(cache[path][i]))
F = pd.DataFrame(F); F = F.loc[:, F.std() > 1e-9]; y = pool.difficulty.values
add("Sokoban (200, RL-style label)", "budgeted-search D", "ours (RF, CV)", rho(oof(F.values, y), y), 0.99)
add("Sokoban (200, RL-style label)", "budgeted-search D", "Boxoban tier (computed by running agents; not a free knob)", rho(pool.tier.map({"unfiltered": 0, "medium": 1, "hard": 2}), y), 0.99)
L = llm("sokoban_rl_pool")
if L is not None:
    Ls = L.loc[pool.level_id].values
    add("Sokoban (200, RL-style label)", "budgeted-search D", "LLM Qwen7B zero-shot", rho(Ls, y), 0.99)
    add("Sokoban (200, RL-style label)", "budgeted-search D", "ours + LLM (RF, CV)", rho(oof(np.c_[F.values, Ls], y), y), 0.99)
# LavaGap
lg = pd.read_csv(ROOT / "direction_c_AB.csv")
sys.argv = ["x"]
from importlib.machinery import SourceFileLoader
ld = SourceFileLoader("ld", str(ROOT / "notes/rescue_audit_2026-10-01/lavagap_descriptor.py"))
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()): ldm = ld.load_module()
Dg = ldm.D.loc[:, ldm.D.std() > 1e-9].values.astype(float); y = lg.diff_A_partial.values
add("MiniGrid LavaGap (60)", "partial-progress D", "ours 3-layer (RF, CV)", rho(oof(Dg, y), y), np.sqrt(0.79))
add("MiniGrid LavaGap (60)", "partial-progress D", "hand gap features (RF, CV)", rho(oof(ldm.H.values.astype(float), y), y), np.sqrt(0.79))
L = llm("lavagap_shaped60")
if L is not None:
    Lg = L.loc[lg.grid_seed.astype(str)].values
    add("MiniGrid LavaGap (60)", "partial-progress D", "LLM Qwen7B zero-shot", rho(Lg, y), np.sqrt(0.79))
    add("MiniGrid LavaGap (60)", "partial-progress D", "ours 3-layer + LLM (RF, CV)", rho(oof(np.c_[Dg, Lg], y), y), np.sqrt(0.79))
# LavaCrossing
rq = pd.read_csv(ROOT / "data/benchmarks/rollout_equivalence_lavacrossing.csv") if (ROOT / "data/benchmarks/rollout_equivalence_lavacrossing.csv").exists() else None
D = ROOT / "notes/rescue_audit_2026-10-01/derisk_out"
for tag, ceil in [("p7911", np.sqrt(0.57)), ("p91317", np.sqrt(0.58))]:
    dom = f"MiniGrid LavaCrossing {tag[1:]}"
    # every LavaCrossing method is scored on the same rotated held-out-agent label (rollout_equivalence.py)
    if rq is not None:
        for _, r in rq[rq.pool == tag].iterrows():
            add(dom, "1 - success (held-out ref PPOs, rotated)", f"{r.method}" + (f" k={r.k}" if r.k else ""), r.rho, ceil)
# Karel
for f in sorted(glob.glob(str(ROOT / "data/karel_difficulty/karel_*_grids.csv"))) + [str(ROOT / "averaged_label_topoff200.csv"), str(ROOT / "averaged_label_screen_snake_seeder.csv")]:
    d = pd.read_csv(f)
    for t, g in d.groupby("task"):
        cols = [c for c in g.columns if c.startswith(("graph_", "spectral_", "grid_", "reward_")) and g[c].std() > 1e-9]
        y = g.mean_difficulty.values
        if np.std(y) < 1e-9: continue
        add(f"Karel {t} ({len(g)})", "HC D (R=12)", "ours (RF, CV)", rho(oof(g[cols].values, y), y))
        L = llm("karel_all")
        if L is not None:
            Lk = L.loc[[f"{t}/{s}" for s in g.grid_seed]].values
            add(f"Karel {t} ({len(g)})", "HC D (R=12)", "LLM Qwen7B zero-shot", rho(Lk, y))
            add(f"Karel {t} ({len(g)})", "HC D (R=12)", "ours + LLM (RF, CV)", rho(oof(np.c_[g[cols].values, Lk], y), y))
out = pd.DataFrame(R)
# combos need an LLM call per level, so they are not execution-free-and-cheap: report as secondary only
out["role"] = np.where(out.method.str.contains(r"\+ LLM|LLM rank-avg"), "secondary", "main")
out.to_csv(ROOT / "data/benchmarks/difficulty_comparison_table.csv", index=False)
pd.set_option("display.width", 200); print(out.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
