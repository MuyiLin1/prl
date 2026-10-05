"""Bootstrap 95% CIs (resampling levels, B=2000) for the main prediction table (paper Table tab:prediction).
Same predictions and labels as difficulty_table.py / rollout_equivalence.py."""
import glob, io, json, contextlib, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts/curriculum"), str(ROOT / "scripts/sokoban")]
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

B = 2000
P = ROOT / "data/benchmarks/llm_prompts"
ROWS = []


def llm(name):
    return pd.read_csv(P / f"{name}_llm_qwen7b.csv", dtype={"id": str}).set_index("id").llm_expected


def oof(X, y, classify=False):
    p = np.zeros(len(y))
    for tr, te in (StratifiedKFold if classify else KFold)(5, shuffle=True, random_state=0).split(X, y):
        M = RandomForestClassifier if classify else RandomForestRegressor
        m = M(300, random_state=0, n_jobs=4).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1] if classify else m.predict(X[te])
    return p


def boot(pairs, auc=False):
    """pairs: list of (pred, label) arrays over the SAME levels (several = averaged, e.g. label rotations)."""
    f = (lambda a, b: roc_auc_score(b, a)) if auc else (lambda a, b: spearmanr(a, b)[0])
    n = len(pairs[0][1])
    rng = np.random.default_rng(0)
    point = np.mean([f(p, y) for p, y in pairs])
    bs = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        bs.append(np.mean([f(p[i], y[i]) for p, y in pairs]))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return point, lo, hi


def add(domain, label, method, pairs, auc=False):
    v, lo, hi = boot(pairs, auc)
    ROWS.append(dict(domain=domain, label=label, method=method, value=v, lo=lo, hi=hi))
    print(f"{domain:26s} {label:16s} {method:26s} {v:+.2f} [{lo:+.2f}, {hi:+.2f}]", flush=True)


def sokoban_public():
    b = pd.read_csv(ROOT / "data/benchmarks/boxoban_astar/benchmark_levels.csv")
    meta = {"level_id", "tier", "astar_search_steps", "astar_solution_len", "ascii"}
    feats = [c for c in b.columns if c not in meta and not c.startswith("kartal") and b[c].std() > 1e-9]
    L = llm("sokoban_astar_benchmark").loc[b.level_id].values
    for lab, y in [("A* effort", np.log10(b.astar_search_steps.values)), ("A* length", b.astar_solution_len.values * 1.0)]:
        add("Sokoban public", lab, "ours", [(oof(b[feats].values, y), y)])
        add("Sokoban public", lab, "Kartal 2016", [(b.kartal2016.values, y)])
        if lab == "A* effort":
            add("Sokoban public", lab, "LLM", [(L, y)])
    t = (b.tier == "medium").astype(int).values
    add("Sokoban public", "tier (AUC)", "ours", [(oof(b[feats].values, t, True), t)], True)
    add("Sokoban public", "tier (AUC)", "Kartal 2016", [(b.kartal2016.values, t)], True)
    add("Sokoban public", "tier (AUC)", "LLM", [(L, t)], True)


def sokoban200():
    from sokoban_core import extract_features, iter_levels_in_file
    pool = pd.read_csv(ROOT / "data/sokoban_fastpath_labels.csv")
    cache, F = {}, []
    for lid in pool.level_id:
        t, f, i = lid.split("/")
        path = ROOT / "data/boxoban" / t / ("" if t == "hard" else "train") / f"{f}.txt"
        cache.setdefault(path, {l.level_id.split("/")[-1]: l for l in iter_levels_in_file(path, t)})
        F.append(extract_features(cache[path][i]))
    F = pd.DataFrame(F); F = F.loc[:, F.std() > 1e-9]; y = pool.difficulty.values
    add("Sokoban 200", "search D", "ours", [(oof(F.values, y), y)])
    add("Sokoban 200", "search D", "Boxoban tier", [(pool.tier.map({"unfiltered": 0, "medium": 1, "hard": 2}).values * 1.0, y)])
    add("Sokoban 200", "search D", "LLM", [(llm("sokoban_rl_pool").loc[pool.level_id].values, y)])


def lavagap():
    lg = pd.read_csv(ROOT / "direction_c_AB.csv"); y = lg.diff_A_partial.values
    sys.argv = ["x"]
    from importlib.machinery import SourceFileLoader
    with contextlib.redirect_stdout(io.StringIO()):
        ldm = SourceFileLoader("ld", str(ROOT / "notes/rescue_audit_2026-10-01/lavagap_descriptor.py")).load_module()
    D = ldm.D.loc[:, ldm.D.std() > 1e-9].values.astype(float)
    add("LavaGap", "search D", "ours", [(oof(D, y), y)])
    add("LavaGap", "search D", "LLM", [(llm("lavagap_shaped60").loc[lg.grid_seed.astype(str)].values, y)])


def lavacrossing():
    """Same rotated held-out-agent label as rollout_equivalence.py; CI averages the 3 rotations per resample."""
    from difficulty_scoring import GRADER_FEATURES, descriptor_features, enumerate_pool, fit_grader
    from curriculum_harness import tuple_key
    D = ROOT / "notes/rescue_audit_2026-10-01/derisk_out"
    calib = {tuple_key(k): v for k, v in json.load(open(ROOT / "scripts/curriculum/cache/grader_calib_p7911_r6.json")).items()}
    cp = enumerate_pool([7, 9, 11], [1, 2, 3], 80, 10)
    for tag, sizes in [("p7911", [7, 9, 11]), ("p91317", [9, 13, 17])]:
        runs = [json.load(open(f)) for f in sorted(glob.glob(f"{D}/{tag}_ref*.json"))]
        pool = enumerate_pool(sizes, [1, 2, 3], 80, 10)
        L = llm(f"lavacrossing_{tag}").loc[[r["sig"] if isinstance(r["sig"], str) else json.dumps(r["sig"])
                                            for r in runs[0]["rows"]]].values
        knobs = np.array([lo.size + lo.n for lo in pool], float)
        g_pairs, k_pairs, l_pairs = [], [], []
        for held in range(3):
            others = [r for i, r in enumerate(runs) if i != held]
            lab = np.mean([[1 - np.mean(r["succ"]) for r in run["rows"]] for run in others], axis=0)
            if tag == "p7911":
                cache = json.load(open(ROOT / "scripts/curriculum/cache/oracle_6ccc84561c_r3e20.json"))
                lab = (2 * lab + 3 * np.array([cache[r["sig"]] for r in runs[0]["rows"]])) / 5
                X = np.array([[descriptor_features(lo.size, lo.n, lo.seed)[k] for k in GRADER_FEATURES] for lo in pool])
                y = np.mean([[1 - np.mean(r["succ"]) for r in run["rows"]] for run in others], axis=0)
                g = np.zeros(len(pool))
                for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
                    g[te] = RandomForestRegressor(300, random_state=0).fit(X[tr], y[tr]).predict(X[te])
            else:
                g = fit_grader(cp, calib, seed=0)(pool)
            g_pairs.append((g, lab)); k_pairs.append((knobs, lab)); l_pairs.append((L, lab))
        dom = f"LavaCrossing {tag[1:]}"
        add(dom, "1-PPO success", "ours (grader)", g_pairs)
        add(dom, "1-PPO success", "knobs size+N", k_pairs)
        add(dom, "1-PPO success", "LLM", l_pairs)


if __name__ == "__main__":
    for fn in [lavagap, lavacrossing, sokoban200, sokoban_public]:
        fn()
    pd.DataFrame(ROWS).to_csv(ROOT / "data/benchmarks/prediction_table_ci.csv", index=False)
    print("wrote data/benchmarks/prediction_table_ci.csv")
