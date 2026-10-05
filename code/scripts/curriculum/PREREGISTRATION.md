# Pre-registration: curriculum v2 runs (written 2026-10-02, before any v2 result exists)

Repo HEAD 02bcc8b (`scripts/curriculum/` is untracked; every result JSON stores md5 of
`curriculum_harness.py` and `difficulty_scoring.py`). Originals of both files are saved in
`notes/rescue_audit_2026-10-01/*.ORIGINAL.py`.

## Runs
- **Main (results_v2_lc91317/):** LavaCrossing sizes {9,13,17} × rivers {1,2,3}; 90 train / 36 held-out
  layouts; 1.5M steps; gate 0.55; deadline 0.25; target 0.30; seeds 0–19 for every condition, all on one machine.
- **Scope (results_v2_lc7911/):** sizes {7,9,11}, 600k steps, seeds 0–9, conditions flat and size_order.

## Conditions (all but `flat` share identical curriculum mechanics; ties broken by a seeded random key)
| condition | score |
|---|---|
| flat | none: all layouts from step 0 (= old `random`) |
| size_order | old `geometric_score` (= grid size order on 9/13/17; = old `geometry`) |
| random_curriculum | uniform random score: curriculum mechanics with an uninformative order |
| rivers_only | tier N (MiniGrid's own difficulty knob) |
| grader | RandomForest(300, seed 0) on `GRADER_FEATURES` (three-layer descriptor; no N, no est_rivers, no explicit size), fit on 6-reference-seed labels (1 − success) of the separate 7/9/11 calibration pool (`cache/grader_calib_p7911_r6.json`). No 9/13/17 rollouts are used to build it. |

Checked before launch (does not involve curriculum outcomes): grader transfer Spearman vs fresh 3-seed
9/13/17 success labels = +0.48 (within-cell +0.38); size order = +0.32. The grader's bins mix sizes.

## Metrics and tests
- **Primary metric:** final held-out success at 1.5M steps, per seed.
- **Primary tests (one-sided Mann–Whitney, α = 0.05, no multiplicity correction; reported as such):**
  1. size_order > random_curriculum (does *order* matter?)
  2. grader > random_curriculum
  3. random_curriculum > flat (do curriculum *mechanics* matter?)
- **Secondary:** grader vs size_order (two-sided); area under the held-out curve; steps-to-target 0.30;
  number of seeds reaching target; same tests on the 7/9/11 scope run (expected: flat ≥ size_order).
- **Power:** pooled per-seed SD ≈ 0.31, so 20 seeds per arm detect differences of ≈0.25 at 80% power.
  Smaller differences will be reported as "not resolved", never as "equivalent".

## Decision rules for the write-up
- Test 1 or 2 significant → "a cheap structural difficulty order improves curricula over a random order".
- Tests 1 and 2 not significant, test 3 significant → "the benefit comes from staged growth of the
  training set, not from difficulty order"; the ordering claim is dropped.
- No scoring rule is changed after seeing results. Any later variant is reported as a separate, post hoc row.

---
## Addendum A (2026-10-02, before any SFL or extra-seed result exists)

### SFL with and without the grader (`sfl_harness.py`, results_v2_sfl_lc91317/)
Same pool generator cells (9/13/17 × N 1–3; candidate seeds < 100k), same held-out set, PPO, 1.5M steps.
Refresh every 250k steps; k = 4 rollouts per scored candidate; buffer B = 30, trained with 30 fresh random layouts.
- `sfl`: 100 candidates scored by rollouts (standard SFL).
- `sfl_small`: 20 candidates scored by rollouts (scoring budget matched to sfl_grader).
- `sfl_grader`: 100 candidates; 20 rolled out to fit a logistic map grader score → current success; buffer =
  top-30 by predicted p(1−p) over all 100. If all 20 calibration rollouts fail (succeed), take the 30
  lowest (highest) grader scores.
Seeds 0–9 per condition.
- **Primary:** final held-out success. **Tests:** sfl_grader > sfl_small (one-sided MWU); sfl_grader vs sfl
  (two-sided MWU, plus the difference and its bootstrap 95% CI); scoring env-steps reported for all.
- **Claim if supported:** "matches standard SFL at ~1/5 of the scoring rollouts" requires sfl_grader > sfl_small
  or the CI of (sfl_grader − sfl) to exclude differences below −0.10; otherwise reported as not resolved.

### Extra seeds for the main curriculum run
Seeds 20–39 for flat, size_order, random_curriculum, rivers_only, grader (same config). The primary analysis
uses seeds 0–19 as pre-registered above; 0–39 is reported as a pre-specified secondary analysis.

### Same-dataset prediction benchmarks (no training runs)
Boxoban public A* labels (search effort, solution length) and tier; local Qwen2.5-7B-Instruct (4-bit, MLX) zero-shot
judge scored by the expected digit 0–9 on identical instances in every domain. Fixed prompt in
`scripts/benchmarks/make_llm_prompts.py`; no prompt tuning.

---
## Addendum B (2026-10-03, written while the maze pilot runs; before any pilot or main maze result exists)

### Why
The LavaCrossing grader tied ordering by grid size (a designer knob). This tests the grader on a level family
whose generator has no size knob: domain-randomized MiniGrid mazes (`maze_harness.py`): fixed grid size, a
uniform number of interior walls in [0, max_walls], random start and goal, resampled until solvable.

### Pilot (used ONLY to choose the maze size; reported in the appendix, never pooled with the main run)
`cluster/maze_pilot.sh`: sizes 13/15/17 (max_walls 50/70/90), conditions flat and wall_count, seeds 0–1,
1.5M steps. **Size rule (fixed now):** pick the smallest pilot size whose mean flat final held-out success is
≤ 0.30. If no size qualifies, use 17; if all sizes have both conditions below 0.05 (nothing learnable), stop
and report the maze test as infeasible at this budget.

### Calibration of the grader (no main-run outcome involved)
Separate calibration pool: 90 mazes, seeds ≥ 500,000, same size and max_walls as the main run. Three
reference PPOs trained flat for 1.5M steps; label = 1 − success over 20 episodes per maze. Report the tie
fraction at 1.0 and the agreement between reference agents. The grader is RandomForest(300, seed 0) on
`GRADER_FEATS` (no explicit wall count). If more than 50% of calibration labels are tied at 1.0, recalibrate
once on the next smaller pilot size (the LavaCrossing recipe: calibrate where agents learn, transfer up) and
say so.

### Main run
90 training / 36 held-out mazes; 1.5M steps; gate 0.55; deadline 0.25; target 0.30; seeds 0–19.
Conditions: flat, random_curriculum, wall_count (the generator's only parameter = designer knob),
path_len (BFS start–goal distance), grader.
- **Primary tests (one-sided Mann–Whitney, α = 0.05, no multiplicity correction; reported as such):**
  1. grader > random_curriculum
  2. grader > wall_count
  3. grader > path_len
- **Secondary:** random_curriculum > flat; wall_count > random_curriculum; seeds reaching 0.30.
- **Decision rules:** test 1 significant → "the grader orders a knob-free generator better than chance".
  Tests 2 and 3 significant → "and better than the simplest layout heuristics". Otherwise report the
  ordering of means and "not resolved". No scoring rule is changed after seeing results.

### Amendment B1 (2026-10-03 16:15, after pilot 1, before any main maze run)
Pilot 1 result: flat PPO already solves sparse random mazes at every size (final held-out success:
13×13 0.90/0.95, 15×15 0.90/0.72, 17×17 0.84). Following the size rule literally (use 17) would run a test in
the regime where curricula cannot help (as on LavaCrossing 7/9/11). Amendment: run pilot 2 with **dense**
mazes, (size, max_walls) ∈ {(17, 160), (21, 240)}, flat and wall_count, seeds 0–1, same budget. Same rule:
pick the first config (in that order) with mean flat final success ≤ 0.30. If neither qualifies, the maze
test is reported as "flat training already succeeds; no curriculum regime found" and the main run is not
launched. Pilot 1 is reported in the appendix.
Update (16:30, still before any main maze run): a generator check (no training) showed that independent random
walls cannot produce hard mazes. With corner-to-corner start/goal, rejection sampling for solvability keeps
the median detour over the Manhattan path at 0 (13×13 to 17×17). Pilot 2 is therefore NOT run with this
generator. Any further maze test needs a structured generator (e.g. randomized depth-first perfect maze with
a fraction of walls removed) and will get its own addendum before it is run.

### Amendment B2 (2026-10-03, structured mazes; before any training on them)
Generator (`maze_harness.py --perfect --p-max 0.5`): randomized depth-first perfect maze on an odd grid, start
top-left, goal bottom-right, then a uniform fraction p ∈ [0, 0.5] of the removable walls is knocked out (adds
loops and shortcuts). p is the generator's only parameter; the `wall_count` condition orders by the wall count it
induces (the designer knob). Generator check (no training): detour over Manhattan distance q90 = 20 (15×15) and
32 (19×19); Spearman(wall count, path length) = 0.62–0.81.
**Pilot 3** (`cluster/maze_pilot3.sh`): sizes 11/15/19, flat and wall_count, seeds 0–1, 1.5M steps; used only
to choose the size. **Size rule:** smallest size with mean flat final held-out success ≤ 0.30 and mean
wall_count final success ≥ 0.05 (something is learnable). If none qualifies, report "no curriculum regime
found for structured mazes at this budget" and do not run the main test. Calibration, main-run conditions,
tests, and decision rules are exactly those of Addendum B.

### Amendment B3 (2026-10-04 11:55, applying the B2 size rule early; before any main maze run)
A watchdog bug (pause on any swap use) froze all pilot-3 jobs overnight; fixed. Pilot state at 11:55: size 15
flat seed 0 finished at 0.82, so the size-15 flat mean cannot be ≤ 0.30 (would need seed 1 < 0); size 11 flat
seeds are at 0.96/0.96 after 550–600k steps. Size 19: flat seed 0 = 0.00, wall_count seed 0 = 0.77 (wall_count
mean ≥ 0.05 already guaranteed). Size 11 and 15 runs are stopped (unfinished; reported as such). **Chosen
size: 19** (p_max 0.5), conditional on size-19 flat seed 1 finishing ≤ 0.60 (which keeps the mean ≤ 0.30);
that run continues. Calibration starts now at size 19.

---
## Addendum C (2026-10-04 ~20:45, after the B3 main run, before any run of this test)

### Why
The B3 main run (19×19, p_max 0.5) was not resolved: flat training already reached ≥ 0.25 on 14/20 seeds,
so no ordering could separate. That result stays reported (appendix). This addendum tests a harder version of
the same knob-free generator.

### Setting
Same generator and size (19×19 structured mazes), fewer shortcuts: p_max ∈ {0.25, 0.10} (median start–goal
path 40 and 56 vs 32 at p_max 0.5; generator check only, no training).
**Pilot** (`cluster/maze_c.sh`): for each p_max, flat seeds 0–3 and wall_count seeds 0–1, 1.5M steps; used
only to choose p_max. **Rule (stricter than B, because a 2-seed pilot misled us):** take the first of
[0.25, 0.10] with mean flat final success ≤ 0.15 and mean wall_count final success ≥ 0.05. If neither
qualifies, report "no curriculum regime found" and do not run the main test.

### Grader
Reuses the existing calibration (`cache/maze_calib_s19_perfect_p0.5.json`: 90 separate mazes at p_max 0.5,
three reference PPOs), fixed before this addendum. This is the LavaCrossing recipe: calibrate where agents can
learn, apply to the harder setting. Flags: `--p-max <chosen> --calib-p 0.5`.

### Main run
Conditions, budget, controller, seeds 0–19, primary tests, and decision rules exactly as in Addendum B
(grader > random_curriculum, > wall_count, > path_len; one-sided MWU, α = 0.05). Results in
`results_maze_s19_p<chosen>/`. Pilot runs are reported in the appendix and never pooled.
