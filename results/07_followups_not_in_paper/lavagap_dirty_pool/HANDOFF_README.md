# Dirty-pool curriculum: reproduction + patches + results

This bundle contains everything to reproduce the curation-vs-random follow-up,
plus the two small code fixes needed to run it in a torch-free environment.

## 1. Code patches (apply first)

The programmatic-only experiment (`dirty_pool_curriculum.py`) fails to import in a
clean environment because `prog_policies` eagerly imports the latent-space search
methods (CEM/CEBS/HillClimbingLatent + the VAE), which drag in torch / h5py /
old-gym that the experiment never uses. Two one-file patches make those imports
lazy. They change NO experimental logic — the smoke test reproduces the original
CSVs exactly.

From the repo root:
    patch -p1 < patches/01_search_space_lazy_import.patch
    patch -p1 < patches/02_search_methods_lazy_import.patch
(or `git apply patches/*.patch`)

After patching, only numpy/pandas/scikit-learn/scipy + the local prog_policies
are needed. Legacy `gym==0.15.4` may still be required by an unrelated import;
`pip install gym==0.15.4` if so.

## 2. Reproduce a point (sharded, so each shard fits a short wall-clock)

    # 5 shards, run each then merge. Settings below are the LOW-FIDELITY config
    # used in the sandbox (size 9). For a real run see section 4.
    for S in 0 1 2 3 4; do
      python scripts/curriculum/dirty_pool_curriculum.py \
        --gen-size 250 --pool-size 120 --k 16 --heldout 30 --size 9 \
        --probe-seeds 1 --n-iter 40 --k-neighbors 32 --max-calls 400 \
        --train-seeds 3 --full-train-seeds 1 --draws 8 \
        --frac-trivial 0.55 --frac-hard 0.15 --frac-dup 0.15 \
        --out sweep2/p055 --shard $S --num-shards 5
    done
    python scripts/curriculum/dirty_pool_curriculum.py --out sweep2/p055 --merge

## 3. Analyze (tier breakdown + proper stats)

    python scripts/analyze_dirty_pool.py sweep2/p020 sweep2/p055

This prints, per point: the pool's easy/medium/hard composition and per-condition
mean + bootstrap 95% CI + one-sided Mann-Whitney vs random.

## 4. Key findings (high-draw, 24 runs/condition)

    frac_trivial=0.20:  random .299 | band .285 (p.81) | coverage .327 (+.028,p.09) | both .282 (p.80)
    frac_trivial=0.55:  random .304 | band .294 (p.54) | coverage .321 (+.017,p.20) | both .298 (p.35)

- `both` (full curation) TIES random at every contamination level. The +0.016
  we saw at 8 runs/condition was small-sample noise; it vanished at 24 runs.
- Only `coverage` ever edges above random, and it's NOT significant and is
  stronger on the CLEAN pool than the dirty one (backwards from the hypothesis).
- `band` (difficulty filter) consistently HURTS.

## 5. WHY — the diagnosis

The difficulty label is degenerate / binary. `d_probe` only takes values in
[0.50, 1.00] and clumps at the two ends:

    frac_trivial=0.20 pool:  easy=102  medium=0  hard=96
    frac_trivial=0.55 pool:  easy=66   medium=0  hard=54

There are ZERO medium tasks. Every task is solved fast (~0.5) or not at all (~1.0).
So the difficulty band has no learnable middle to keep (that's why band hurts),
coverage has a near-binary space to spread over, and random reliably grabs a fine
mix. Curation can't beat random because size-9 LavaGap has no graded difficulty
structure to exploit. This is the paper's Gate-1 (degenerate-label) problem
appearing inside the curriculum experiment.

## 6. Recommended next steps (do NOT re-run size-9 LavaGap)

1. Regrade the label first: larger probe budget or shaped reward so tasks spread
   across easy/medium/hard instead of clumping at 0.5 and 1.0. `medium=0` alone
   could explain every null.
2. Move to a genuinely graded environment (Sokoban or multi-river LavaCrossing).
   Grid size is a fake knob here — bigger LavaGap is just more empty space, same
   one-gap decision.
3. Then the distribution-shift test (train easy/medium, test hard) — only once
   medium/hard actually exist as populated tiers.

Bar for reporting: the effect must be SUBSTANTIAL (>=~0.05 with non-overlapping
CIs), not merely significant. +0.02 that reaches p<0.05 with enough draws is not
worth a paragraph.
