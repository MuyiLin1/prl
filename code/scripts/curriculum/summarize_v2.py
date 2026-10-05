"""Pre-registered summaries. usage: summarize_v2.py main|scope|sfl|extra"""
import glob, json, sys
import numpy as np
from scipy.stats import mannwhitneyu
from pathlib import Path
H = Path(__file__).resolve().parent
def load(d, conds, seeds):
    out = {}
    for c in conds:
        v = []
        for s in seeds:
            f = H / d / f"{c}_seed{s}.json"
            if f.exists():
                j = json.load(open(f)); v.append((j["final_held_success"], j.get("steps_to_target"), j.get("scoring_steps")))
        out[c] = v
    return out
def table(R):
    print(f"{'condition':18s} {'n':>3s} {'final mean':>10s} {'95% CI':>15s} {'reached 0.30':>12s}")
    for c, v in R.items():
        if not v: continue
        x = np.array([a for a, _, _ in v]); h = 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0
        reached = sum(b is not None for _, b, _ in v) if v[0][1] is not None or any(b for _, b, _ in v) else "-"
        extra = f"  scoring steps {np.mean([s for *_, s in v]):,.0f}" if v[0][2] is not None else ""
        print(f"{c:18s} {len(x):3d} {x.mean():10.3f} [{x.mean()-h:+.2f},{x.mean()+h:+.2f}] {str(reached):>12s}{extra}")
def test(R, a, b, alt):
    x = [v[0] for v in R[a]]; y = [v[0] for v in R[b]]
    if len(x) > 1 and len(y) > 1:
        print(f"  {a} vs {b} ({alt}): diff {np.mean(x)-np.mean(y):+.3f}, MWU p = {mannwhitneyu(x, y, alternative=alt).pvalue:.3f}")
mode = sys.argv[1]
if mode == "maze":
    R = load("results_maze_s19", ["flat", "random_curriculum", "wall_count", "path_len", "grader"], range(20))
    print("== Structured mazes 19x19 (Addendum B/B2/B3), seeds 0-19 =="); table(R)
    print(" primary:")
    test(R, "grader", "random_curriculum", "greater"); test(R, "grader", "wall_count", "greater")
    test(R, "grader", "path_len", "greater")
    print(" secondary:")
    test(R, "random_curriculum", "flat", "greater"); test(R, "wall_count", "random_curriculum", "greater")
if mode in ("main", "extra"):
    seeds = range(20) if mode == "main" else range(40)
    R = load("results_v2_lc91317", ["flat", "size_order", "random_curriculum", "rivers_only", "grader"], seeds)
    print(f"== 9/13/17 curriculum, seeds {seeds.start}-{seeds.stop-1} =="); table(R)
    test(R, "size_order", "random_curriculum", "greater"); test(R, "grader", "random_curriculum", "greater")
    test(R, "random_curriculum", "flat", "greater"); test(R, "grader", "size_order", "two-sided"); test(R, "rivers_only", "random_curriculum", "greater")
if mode == "main":
    R = load("results_v2_lc7911", ["flat", "size_order"], range(10))
    print("== 7/9/11 scope check (expected: flat >= size_order) =="); table(R); test(R, "flat", "size_order", "greater")
if mode in ("sfl", "sfl_extra"):
    seeds = range(10) if mode == "sfl" else range(20)
    R = load("results_v2_sfl_lc91317", ["sfl", "sfl_small", "sfl_grader"], seeds)
    print("== SFL with/without grader ==" if mode == "sfl" else
          "== SFL seeds 0-19: POST-HOC EXPLORATORY (seeds 10-19 added after seeing the 0-9 result) =="); table(R)
    test(R, "sfl_grader", "sfl_small", "greater"); test(R, "sfl_grader", "sfl", "two-sided")
    x = np.array([v[0] for v in R["sfl_grader"]]); y = np.array([v[0] for v in R["sfl"]])
    if len(x) > 1 and len(y) > 1:
        rng = np.random.default_rng(0); bs = [rng.choice(x, len(x)).mean() - rng.choice(y, len(y)).mean() for _ in range(5000)]
        print(f"  sfl_grader - sfl bootstrap 95% CI [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]")
