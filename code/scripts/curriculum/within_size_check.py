"""Does the LavaCrossing grader rank levels WITHIN a grid size, or only by size?

Uses the exact grader and pool of the 9/13/17 curriculum runs and the 3-agent PPO label.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import curriculum_harness as H  # noqa: E402

R = Path(__file__).parent / "results_v2_lc91317"
a = json.loads((R / "grader_seed0.json").read_text())["args"]
pool = H.enumerate_pool(a["sizes"], a["tiers"], a["n_seeds"], a["cap_per_cell"])
print(f"pool: {len(pool)} levels, sizes {sorted(Counter(lo.size for lo in pool).items())}")

calib = {H.tuple_key(k): v for k, v in json.loads((H.CACHE_DIR / H.GRADER_CALIB).read_text()).items()}
cpool = H.enumerate_pool([7, 9, 11], [1, 2, 3], 80, 10)
pred = np.array(H.fit_grader(cpool, calib, seed=0)(pool))

ROOT = Path(__file__).resolve().parents[2]
runs = [json.loads(f.read_text()) for f in sorted((ROOT / "notes/rescue_audit_2026-10-01/derisk_out").glob("p91317_ref*.json"))]
print(f"label: {len(runs)} reference PPO agents (paper's 3-agent 1-success label)")
lab = {}
for run in runs:
    for r in run["rows"]:
        lab.setdefault(H.tuple_key(r["sig"] if isinstance(r["sig"], str) else json.dumps(r["sig"])), []).append(1 - np.mean(r["succ"]))
y = np.array([np.mean(lab[lo.sig]) for lo in pool])
size = np.array([lo.size for lo in pool])
riv = np.array([lo.n for lo in pool])

print(f"\nall levels: grader rho={spearmanr(pred, y)[0]:.2f}  size-knob rho={spearmanr(size, y)[0]:.2f}"
      f"  size+rivers rho={spearmanr(size * 10 + riv, y)[0]:.2f}  grader-vs-size rho={spearmanr(pred, size)[0]:.2f}")
for s in sorted(set(size)):
    m = size == s
    ties = 1 - len(np.unique(y[m])) / m.sum()
    print(f"size {s:2d} (n={m.sum()}): grader rho={spearmanr(pred[m], y[m])[0]:+.2f}  rivers rho={spearmanr(riv[m], y[m])[0]:+.2f}"
          f"  label mean={y[m].mean():.2f} tied={ties:.0%}")

# Rank-level within-size: residualise both on size (partial Spearman)
ry = np.argsort(np.argsort(y)).astype(float)
rp = np.argsort(np.argsort(pred)).astype(float)
for s in set(size):
    m = size == s
    ry[m] -= ry[m].mean(); rp[m] -= rp[m].mean()
print(f"pooled within-size (rank, size-centred): {np.corrcoef(rp, ry)[0, 1]:.2f}")

rng = np.random.default_rng(0)
idx_by = {s: np.where(size == s)[0] for s in set(size)}
def within(ix):
    a_, b_ = [], []
    for s in set(size):
        j = ix[size[ix] == s]
        ra = np.argsort(np.argsort(pred[j])).astype(float); rb = np.argsort(np.argsort(y[j])).astype(float)
        a_.append(ra - ra.mean()); b_.append(rb - rb.mean())
    return np.corrcoef(np.concatenate(a_), np.concatenate(b_))[0, 1]
boot = [within(np.concatenate([rng.choice(v, len(v)) for v in idx_by.values()])) for _ in range(2000)]
obs = within(np.arange(len(pool)))
perm = []
for _ in range(2000):
    p2 = pred.copy()
    for v in idx_by.values():
        p2[v] = rng.permutation(p2[v])
    pp, pred_saved = p2, pred
    pred = pp; perm.append(within(np.arange(len(pool)))); pred = pred_saved
print(f"within-size: {obs:.2f}  95% CI [{np.percentile(boot, 2.5):.2f}, {np.percentile(boot, 97.5):.2f}]"
      f"  one-sided permutation p={np.mean(np.array(perm) >= obs):.4f}")

print("\nbin composition (grader, seed 0 tiebreak):")
for i, b in enumerate(json.loads((R / "grader_seed0.json").read_text())["initial_bins"]):
    sz = Counter()
    for k, v in b.items():
        sz[int(k[1:].split("N")[0])] += v
    print(f"  bin {i}: {dict(sorted(sz.items()))}   full: {b}")
