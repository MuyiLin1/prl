# Sokoban DSL + Hill Climbing — Cluster Labelling Bundle

This bundle adds a **real Sokoban DSL + Hill Climbing reference solver** to
`prog_policies`, so Sokoban difficulty is labelled with the *same* program
synthesis machinery as Karel and MiniGrid, instead of the budgeted state-space
search used in the fast path.

Two changes make the label track real Sokoban difficulty (validated on a
30-level pilot, see §6):

1. **`solveStep` macro action.** Primitive-only Hill Climbing can never push a
   4-box level to completion, so the primitive label was degenerate (and ran
   *backwards* vs the Boxoban tiers). `solveStep` is one bounded DSL action: it
   plans the cheapest single box that can be pushed fully onto a free target and
   applies it, counting as exactly one env call. This gives HC a real solving
   gradient and creates genuine order-dependent (push-ordering) difficulty.
2. **On-target-weighted shaping.** The shaped reward weights *boxes actually on
   target* at `0.8` and matching-distance *progress* at `0.2` (was `0.5/0.5`),
   so the label rewards genuine completion over gameable nudging.

> **Caveat (disclose in the paper):** `solveStep` bakes a single-box push
> planner into the Sokoban DSL. Karel and MiniGrid have no comparable macro, so
> Sokoban is no longer *strictly* apples-to-apples with them. It remains a
> DSL + Hill-Climbing program search; only the action set is richer.

## What it contains

When installed at the repo root it adds / updates:

```
prog_policies/sokoban/__init__.py          # new package
prog_policies/sokoban/environment.py       # SokobanEnvironment(BaseEnvironment)
prog_policies/sokoban/dsl.py               # SokobanDSL(BaseDSL)
prog_policies/sokoban_tasks/__init__.py    # task registry
prog_policies/sokoban_tasks/boxoban.py     # BoxobanTask(BaseTask) + shaped reward
prog_policies/utils/__init__.py            # get_env_name -> adds "sokoban"
scripts/sokoban/hc_label.py                # labelling driver (cluster entrypoint)
scripts/sokoban/sokoban_core.py            # Boxoban parser (dependency of the driver)
scripts/sokoban/cluster/run_sokoban_hc.sh  # run script
data/sokoban_fastpath_labels.csv           # the 200 levels to label (level_id,tier)
```

## 1. Install into the repo on the cluster

The bundle mirrors the repo's directory layout, so you can drop it in place:

```bash
# from inside the bundle directory
rsync -a tree/ /path/to/temp-prl/        # additive; only adds/updates the files above
```

> The only *modified* existing file is `prog_policies/utils/__init__.py`; the
> change is purely additive (it registers the `sokoban` env name). Everything
> else is new.

## 2. Get the Boxoban levels (≈170 MB, not in the bundle)

```bash
cd /path/to/temp-prl
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/boxoban-levels data/boxoban
cd data/boxoban && git sparse-checkout set hard medium unfiltered && cd -
```

The driver reconstructs each level's ASCII from these files using the
`level_id` / `tier` columns in the levels CSV, so it labels exactly the same
200 levels the fast path used.

## 3. Run

```bash
cd /path/to/temp-prl
bash scripts/sokoban/cluster/run_sokoban_hc.sh
```

Tunable via environment variables (defaults shown):

| var | default | meaning |
|-----|---------|---------|
| `CONDA_ENV` | `llm_gs` | conda env to activate |
| `LABELS_CSV` | `data/sokoban_fastpath_labels.csv` | levels to label (`level_id,tier`) |
| `NUM_SEEDS` | `12` | HC seeds per level (R) |
| `BUDGET` | `4000` | program-evaluation budget per seed |
| `K` | `250` | HC neighbours per step |
| `SIGMA` | `0.25` | mutation sigma |
| `MAX_CALLS` | `2000` | max env steps per program rollout |
| `W_ON_TARGET` | `0.8` | shaping weight: boxes actually on target |
| `W_PROGRESS` | `0.2` | shaping weight: matching-distance progress |
| `OUT_PREFIX` | `data/sokoban_hc_labels` | output prefix |

Output: `data/sokoban_hc_labels_labels.csv` (per-level `difficulty`,
`mean_reward`, `solved_frac`) and `data/sokoban_hc_labels_perseed.json`
(per-seed rewards + split-half reliability `rho_reliability`).

## 4. Parallelise (recommended)

The job is embarrassingly parallel across levels. Shard it across N workers /
array-job tasks and concatenate afterwards:

```bash
# e.g. 8 shards with GNU parallel
seq 0 7 | parallel -j8 \
  "EXTRA_ARGS='--shard {} --num-shards 8' OUT_PREFIX=data/sokoban_hc \
   bash scripts/sokoban/cluster/run_sokoban_hc.sh"

# then merge the per-shard label CSVs (header from the first shard)
python - <<'PY'
import csv, glob
rows=[]; 
for f in sorted(glob.glob('data/sokoban_hc_shard*of*_labels.csv')):
    rows += list(csv.DictReader(open(f)))
with open('data/sokoban_hc_labels_labels.csv','w',newline='') as fh:
    w=csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print('merged', len(rows), 'levels')
PY
```

## 5. Sanity check (decision gate)

Before trusting the full run, confirm the label is **graded**, **reliable**, and
**tracks the Boxoban tiers**:

* `difficulty` variance should be clearly > 0 with a spread of values (not all
  ≈ 1.0 — that would mean a degenerate flat HC landscape).
* `rho_reliability` should be high (the 30-level pilot got 0.98).
* `solved_frac` should be > 0 and decline across tiers
  (pilot: unfiltered 0.60, medium 0.10, hard 0.00).
* Mean difficulty should **increase** across tiers: `unfiltered < medium < hard`
  (pilot tier-vs-D Spearman = +0.44; the primitive-only label was −0.34).

### Validated pilot (30 levels, 10 per tier)

With the macro + `0.8/0.2` shaping, all four signals moved the right way:

| metric | primitive-only | macro + shaping |
|--------|----------------|-----------------|
| tier-vs-D Spearman | −0.34 (backwards) | **+0.44** (p=0.016) |
| solved_frac (mean) | 0.00 | **0.23** |
| reliability ρ_rel | 0.94 | **0.98** |
| vs independent fast-path label | +0.04 | **+0.54** (p=0.002) |

Pilot command (run locally, no cluster):

```bash
python scripts/sokoban/hc_label.py --labels <balanced_subset.csv> \
    --num-seeds 6 --budget 1200 --k 64 --sigma 0.25 --max-calls 1500 \
    --w-on-target 0.8 --w-progress 0.2 --out /tmp/sok_pilot
```

After the full run, re-check the tier ranking and the agreement with the
fast-path label:

```bash
python - <<'PY'
import csv, json, numpy as np
from scipy.stats import spearmanr
rows=list(csv.DictReader(open('data/sokoban_hc_labels_labels.csv')))
order={'unfiltered':0,'medium':1,'hard':2}
for r in rows: r['difficulty']=float(r['difficulty']); r['solved_frac']=float(r['solved_frac'])
for t in ['unfiltered','medium','hard']:
    sub=[r for r in rows if r['tier']==t]
    print(f"{t:11s} n={len(sub):3d} meanD={np.mean([r['difficulty'] for r in sub]):.3f} "
          f"solved={np.mean([r['solved_frac'] for r in sub]):.3f}")
rho,p=spearmanr([order[r['tier']] for r in rows],[r['difficulty'] for r in rows])
print(f"tier-vs-D Spearman = {rho:+.3f} (p={p:.3f})  [want positive]")
fp={r['level_id']:float(r['difficulty']) for r in csv.DictReader(open('data/sokoban_fastpath_labels.csv'))}
c=[(r['difficulty'],fp[r['level_id']]) for r in rows if r['level_id'] in fp]
if len(c)>3:
    a,b=zip(*c); rr,pp=spearmanr(a,b); print(f"HC vs fast-path = {rr:+.3f} (n={len(c)}, p={pp:.3f})")
PY
```

## Notes

* **No LLM anywhere.** HC is initialised from a random program (same as Karel /
  MiniGrid); there is no language-model component.
* The shaped reward (box-on-target fraction + min-cost matching-distance
  progress) plus the `solveStep` macro are what give Hill Climbing a usable
  gradient over program space — without them the landscape is flat (primitive
  label) or backwards vs the tiers.
