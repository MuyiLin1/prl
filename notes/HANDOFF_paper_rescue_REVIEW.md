# Review of `HANDOFF_paper_rescue.md`: verification, new evidence, and a 5-day fix

**Date:** 2026-10-01. **Deadline:** AISTATS 2027 full paper **Tue Oct 6, 23:59 AoE** (abstracts were due Tue Sep 29).
**Evidence:** every number below can be regenerated from `notes/rescue_audit_2026-10-01/` (scripts plus raw de-risk outputs) and the repo's existing result files.

---

## 0. TL;DR

1. **The handoff's diagnosis is mostly right.** F1, F2, F4–F8 are confirmed against the current repo. F3 holds that a change was made after a failed run, but the history is probably different: the scoring module wasn't edited between the runs; the harness was (§1).
2. **Its prescription won't work in time, and parts of it won't work at all:**
   - **Power.** Per-seed curriculum outcomes are bimodal (seeds end at about 0 or at 0.6–0.9; pooled SD 0.31). The pre-registered TOST "matches the oracle within ±0.10" needs **about 120 seeds per arm**. Detecting a 0.20 gap needs about 30. With 10 seeds the only resolvable effect is curriculum vs flat.
   - **The oracle can't be fixed on 9/13/17.** A reference PPO trained flat can't label the regime where flat training fails. Fresh 3-seed rebuild: success label **76% tied**, within-cell reliability **0.21**.
   - **The recommended "Option A" progress label is unusable.** Its reliability is **0.12** on 7/9/11 (noise). On 9/13/17 it is reliable (0.82 within cells), but it is almost fully explained by where the first river gap sits on the path (ρ −0.82, exploratory): reliable, but mechanical, not difficulty.
   - **Min-cut is constant on LavaCrossing.** Start and goal are fixed corners and every river has exactly one gap, so the s–t cut is always 1.
   - **Calendar.** The E0–E4 tier was estimated at 1–2 days of code plus compute, before any rewriting. E5–E8 don't fit.
3. **What the evidence supports is a measurement-first paper**, which fits AISTATS better than a curriculum-engineering paper:
   - *Difficulty labels for generated environments are usually not measurable from one run; here is how to certify them, how many seeds each task needs, and how much of a certified label layout can explain.*
   - Downstream uses are reported at the strength the data actually supports.
4. **One cheap experiment is essential:** a **random-order curriculum** control (F5), 20 seeds, on this Mac overnight (about 4 h). Every current curriculum claim confounds "order" with "staged training-set growth".
5. **A genuinely new finding to report: the labeling paradox.**
   - Where reference-PPO labels are reliable and the descriptor beats the designer knobs (7/9/11: descriptor CV ρ **0.37** vs knobs **−0.09**), curricula aren't needed: flat training wins.
   - Where curricula are needed (9/13/17), flat-trained labels collapse.
   - That is the actual argument for execution-free structural signals.

---

## 1. Verification of the handoff's audit

| ID | Claim | Verdict | Evidence |
|---|---|---|---|
| F1 | "Geometry" = smallest grids first | **Confirmed** | Rebuilt the 90-layout pool: Geometry bins = {all s9}, {all s13}, {all s17}. `est_rivers == N` for 90/90 layouts. |
| F2 | Oracle degenerate; its order is a sort artifact | **Confirmed** | 82/90 labels = 1.0 (4 distinct values). Stable-sort bins: 28/30 s9, 28/30 s13, 29/30 s17. |
| F3 | Score changed after a failed run | **Confirmed, but the history is probably different** | See below. |
| F4 | Flat beats every curriculum on 7/9/11 | **Confirmed** | `v1_buggygate`: flat 0.434 vs geometry 0.194 (10 seeds). `lc7911neg`: 0.567 vs 0.211 (3 seeds). |
| F5 | "Random" = no curriculum | **Confirmed** | `is_curriculum = condition != "random"`. |
| F6 | Unlocks are deadline-driven | **Confirmed** | Geometry first unlocks: 9/10 seeds at 400k (deadline 375k → next 50k checkpoint), second at 800k. Gate-driven unlocks are rare. |
| F7 | MiniGrid prediction doesn't use the descriptor; min-cut not implemented | **Confirmed and quantified** | §2.3. `grep` finds no min-cut / connectivity code anywhere, yet `paper.tex` lists it (lines 177, 913). |
| F8 | `feature_vector()` leaks the tier `n` | **Confirmed** | `difficulty_scoring.py:221`. |
| E8 | "Was the Sokoban dirty-pool sweep run?" | **Yes: null** | `scripts/curriculum/step2_sokoban_*`: band 0.613 / random 0.602 / full 0.612. `step3_sokoban_hard_*` (hard-tier shift): band 0.477 / random 0.492 / full 0.498. |

**F3 details.**
- **Not byte-identical.** The two run folders aren't byte-identical, contrary to the handoff. *All* conditions were re-run on Jun 19 (19:00–22:00). Random/oracle/probe match the Jun 18 run except for wall time, so that box was deterministic.
- **The scoring module didn't change.** `difficulty_scoring.py` was last modified **Jun 17 15:02**, *before both* runs, so `geometric_score` already had the `(size−7)/2` term on Jun 18. `curriculum_harness.py` was modified **Jun 19 19:06**.
- **The harness change looks oracle-related.** Its `initial_scores()` docstring stresses "cheap, policy-free, ORACLE-FREE `geometric_score`", and the new folder is named `oraclefree`. Meanwhile `fit_geometry_oof()` — an RF on oracle labels that its docstring calls "the paper's fitted-predictor method (Part B)" — exists but is now never called.
- **Most likely history.** On Jun 18, the geometry-family conditions used something derived from the degenerate oracle, most plausibly the fitted RF. Reconstructed RF-OOF bins put 7–8 of the s17N1 layouts into the *easy* bin (their oracle labels are 0.9, not 1.0). That run scored **0.085**. The Jun 19 rerun used the hand formula and scored **0.503**.
- **This can't be verified from artifacts.** `scripts/curriculum/` is untracked in git, the JSONs don't log the scoring method, and runs don't reproduce across machines (I checked). **Needs your memory.**
- If this history is right, the paper's own fitted predictor was tried in the curriculum and failed: it was trained on degenerate labels, which is Gate 1 again. That has to be disclosed either way.

**F9.** The current `paper.tex` already states 1 reference seed / 12 episodes in Table `tab:curriculum-config`. The open item is to *say* that this oracle's labels were 91% tied.

---

## 2. New evidence from today

### 2.1 Statistical power of the curriculum experiment (`results_oraclefree_lc91317`)
- **Final held-out success is bimodal.**
  - Geometry: [0, .01, .15, .39, .62, .69, .71, .77, .79, .90].
  - Oracle: [0, .15, .15, .19, .53, .62, .73, .74, .86, .87].
- **Pooled SD = 0.31.** Seeds needed per arm (80% power): TOST ±0.10 → **121**; ±0.15 → 54; ±0.20 → 30. Detecting a 0.30 difference → 13.
- **Current tests (Mann–Whitney):**

  | Comparison | p |
  |---|---|
  | geometry vs oracle | 0.97 |
  | geometry vs probe | 0.087 |
  | oracle vs flat | 0.001 |

  **Only "curriculum ≫ flat" is resolved.**

### 2.2 Oracle de-risk: 3 fresh reference PPOs per pool, 600k steps, 20 episodes per layout
Script: `notes/.../derisk_oracle.py`; analysis: `derisk_analyze.py`.

| Pool | Label | Tied at 1 | Reliability (3 seeds) | Within-cell reliability | Variance explained by (size, N) | Knobs-only CV ρ | Knob-free descriptor CV ρ | Descriptor on within-cell residual |
|---|---|---|---|---|---|---|---|---|
| 7/9/11 | 1 − success | 7% | **0.57** | 0.44 | 9% | **−0.09** | **+0.37** | +0.32 |
| 7/9/11 | 1 − progress | 0% | **0.12** | 0.13 | 32% | +0.33 | +0.48 | +0.21 |
| 9/13/17 | 1 − success | **76%** | 0.58 | **0.21** | 18% | +0.15 | +0.34 | +0.15 |
| 9/13/17 | 1 − progress | 0% | **0.83** | **0.82** | 32% | +0.35 | +0.26 | **−0.10** |

- **Test–retest on 7/9/11:** my independent 3-seed build vs the cached `r3e20` oracle gives ρ **0.66** overall and **0.59** within cells. Within-cell variation is roughly half real.
- **The 9/13/17 progress label is reliable but mechanical** (exploratory, `chokepoint_probe.py`). Within cells it is explained by the position of the first river gap along the shortest path (ρ −0.82). A failing agent walks to the first gap and stops; "progress" measures where that gap is, not how hard the layout is. **Reliability ≠ validity.**

### 2.3 F7 fix: shaped LavaGap with the real three-layer descriptor (`lavagap_descriptor.py`)
Same 60 grids, same R=12 labels (`direction_c_AB.csv`), same RF/CV protocol:

| Difficulty definition | Hand gap features (in paper) | **Three-layer descriptor** | Spectral only | Best single feature |
|---|---|---|---|---|
| Partial progress | 0.76 | **0.57** | 0.37 | graph_diameter −0.47 |
| Success rate | 0.73 | **0.64** | 0.42 | graph_diameter −0.51 |
| Search cost | 0.69 | **0.61** | 0.51 | graph_diameter −0.60 |

The hand-feature column reproduces the paper exactly, which validates the pipeline. The spectrum survives with the honest descriptor: LavaGap ≈0.6 > Sokoban 0.35 > semantic Karel ≈0. Spectral features alone carry signal, which ties the result to the theory.

### 2.4 Variance components (G-study) and seeds needed (D-study), from the existing label files
One-way random effects (grid vs search seed), from per-grid mean/SD with R=12:

| Task | Single-run ICC | Reliability at R=12 (ANOVA) | Paper ρ_rel (split-half) | Seeds for 0.7 | Seeds for 0.8 |
|---|---|---|---|---|---|
| WallAvoider | 0.35 | 0.87 | 0.82 | 4 | 7 |
| TopOff | 0.19 | 0.74 | 0.57 | 10 | 17 |
| DoorKey | 0.18 | 0.73 | 0.71 | 10 | 18 |
| PathFollow | 0.18 | 0.73 | **0.43** | 10 | 18 |
| Snake | 0.17 | 0.72 | 0.71 | 11 | 19 |
| Maze | 0.10 | 0.57 | 0.70 | 21 | 36 |
| OneStroke | 0.05 | 0.40 | 0.44 | 42 | 73 |
| StairClimber | 0.04 | 0.34 | 0.25 | 54 | 92 |
| CleanHouse | 0.03 | 0.29 | 0.24 | 67 | 115 |
| FourCorners | 0.03 | 0.26 | 0.23 | 79 | 135 |
| Seeder | 0.02 | 0.16 | ≈0 | 144 | 247 |
| Harvester | 0.01 | 0.10 | ≈0.05 | 254 | 436 |

- A single search run's label is **1–35% signal**. This turns the qualitative "single-run labels are irreproducible" into a quantitative design rule.
- **Estimator sensitivity:** PathFollow (0.73 vs 0.43) and TopOff (0.74 vs 0.57) disagree between ANOVA and rank split-half. Zero-inflated labels are the likely cause. The gate's verdict for borderline tasks depends on the estimator; say so.

### 2.5 Confidence intervals and the attenuation ceiling
Bootstrap over out-of-fold predictions (`boot_cv.py`):

| Task | CV ρ | 95% CI |
|---|---|---|
| WallAvoider | 0.36 | [0.22, 0.50] |
| TopOff | 0.28 | [0.14, 0.41] |
| Maze | 0.03 | [−0.11, 0.17] |
| Snake | 0.07 | [−0.16, 0.29] |
| DoorKey | −0.14 | [−0.27, −0.01] |

- **Point estimates move by about ±0.1 with forest seed and feature set.** My reruns give WallAvoider 0.36 vs the paper's 0.42. The paper should report repeated-CV means with CIs.
- **WallAvoider vs TopOff overlap.** "Far above every other reliable Karel task" holds for DoorKey, Maze and Snake, not for TopOff.
- **Attenuation ceiling:** a perfect predictor of the true difficulty can reach at most √ρ_rel against a noisy label. Disattenuated values: LavaGap 0.64 (descriptor), WallAvoider 0.46, TopOff 0.37, Sokoban 0.35, DoorKey/Maze/Snake ≈0. This is the principled way to compare predictability across tasks whose labels have different reliability.

---

## 3. Recommended story

**Working title:** *Measure Before You Predict: When Can Environment Difficulty Be Read from Layout?* (or keep the current title and change the framing).

| Claim | Content | Evidence status |
|---|---|---|
| **C1 Theory** | Dynamics gap ≤ C·‖A_θ − A_θ'‖_F + residual; the residual vanishes on deterministic shared grids. Motivates the layout descriptor. | In paper. Trim Hypothesis 4. |
| **C2 Measurement** | Single-run labels are 1–35% signal (G-study). The reliability gate plus a D-study tells you how many seeds each task needs. **The gate applies to oracles too:** the paper's PPO oracle was 91% tied; the progress label is noise on small grids and mechanical on large ones. | Existing data plus today's reanalysis. |
| **C3 Predictability map** | With the real descriptor, disattenuated and with CIs: LavaGap 0.57–0.64, WallAvoider ≈0.4, Sokoban 0.35, TopOff 0.28, semantic Karel ≈0. **New domain:** LavaCrossing 7/9/11, where descriptor 0.37 beats designer knobs (−0.09). Two gates: graded reward, spatial cause. | Mostly existing; LavaCrossing from today. |
| **C4 Use I: prefilter** | Random subsets of ≤10% of the pool match the full pool at 3–15× less compute. Difficulty-aware curation never beats random, replicated across **four** settings: LavaGap clean; LavaGap dirty at 4 contamination levels; Sokoban in-distribution; Sokoban hard-tier shift. A robust negative result with a diagnosis (labels clump, low-dimensional descriptor space). | Existing; the replications are already run. |
| **C5 Use II: ordering** | Curricula help only when flat training fails (9/13/17: 0.50 vs 0.02) and hurt when it succeeds (7/9/11: flat 0.43–0.57 vs 0.19). In the hard regime, empirical labels from flat-trained references collapse (**labeling paradox**), so a structural order is the only signal available at no cost. **New:** a random-order-curriculum control tests whether order matters at all. | Existing plus **one overnight run**. |

**Why this beats the handoff's target for this deadline:**
- Every claim is true today or after one overnight run.
- It plays to the strengths the simulated review already credited (the reliability gate, WallAvoider, honesty).
- It's the statistics paper AISTATS reviewers are equipped to value.
- The grader-curriculum story isn't abandoned. §6 lists it as follow-up work with the de-risk results that scope it.

---

## 4. Five-day plan

**Tonight / Oct 2 (needs your OK):**
1. **Pre-register** in `scripts/curriculum/PREREGISTRATION.md`, with date and hash, before launching:
   - **Primary metric:** final held-out success, seeds 0–19.
   - **Primary test:** one-sided Mann–Whitney, size-order > random-order curriculum.
   - **Secondary:** area under the held-out curve; steps-to-target 0.30.
   - **Power note:** at 20 seeds only differences ≳0.25 are detectable.
2. **Harness changes (about 1 h):**
   - add `random_curriculum` (random scores with curriculum mechanics) and `rivers_only` (N + random tiebreak);
   - rename `random` → `flat` and `geometry` → `size_order`, keeping the old names as aliases so old results still aggregate;
   - break ties randomly with a seed;
   - log the scoring method, git hash, args, per-bin cell composition, unlock kind and steps, and tie fraction into every JSON;
   - write to `results_v2_<tag>/`.
3. **Launch on this Mac (14 cores; about 1,500 s per job at 1 thread):**
   - 9/13/17: {flat, size_order, random_curriculum, rivers_only} × seeds 0–19 = 80 jobs. All seeds run on the same machine, because runs don't reproduce across machines.
   - 7/9/11 scope check: {flat, size_order} × 10 seeds at 600k steps = 20 jobs.
   - About 4–5 h wall time at 12 concurrent jobs.

**Oct 2 (in parallel with the runs):** turn §2.3–2.5 into paper tables:
- LavaGap three-layer descriptor (and RedBlueDoor: its per-grid raw shards exist; PutNear only has a summary, so leave it at hand features with a note);
- G-study / D-study table;
- repeated-CV means with CIs and disattenuated values.
- **Drop "minimum cut"** from the descriptor text. Implementing and rerunning Karel/Sokoban is possible, but it buys nothing on LavaCrossing.

**Oct 3:** analyze the curriculum runs and write C5 using the pre-specified rule:
- If **size_order > random_curriculum** (p < 0.05): "ordering by a structural signal matters, and in the hard regime it is the only signal available."
- If **≈**: "the benefit comes from staged training-set growth, not from difficulty order." The ordering claim is dropped; C2–C4 carry the paper.
- Report rivers_only either way as the second designer knob.

**Oct 4–5:** rewrite the intro, contributions and Sections 5–6 around C1–C5. Write the appendix: run history (Jun 18 vs Jun 19, F3), 7/9/11 results (F4), oracle reliability, deadline-driven unlocks (F6), stochastic evaluation policy. Update the checklist.

**Oct 6:** buffer, proofread, submit.

---

## 5. Paper edits required regardless of outcome

- **Rename curriculum conditions:**
  - Random → **Flat (no curriculum)**;
  - Geometry → **Size order (designer knob)**;
  - Oracle → **Reference-PPO oracle (1 seed; 91% of labels tied)**.

  Remove "execution-free ordering matches an empirical oracle" as a headline.
- **Disclose** the Jun 18 run (geometry 0.085, adaptive_coarse 0.166) and what changed.
- **Report** that flat training wins on 7/9/11.
- **LavaGap:** lead with three-layer descriptor numbers (0.57 / 0.64 / 0.61); give hand features as a reference upper bound.
- **Remove minimum cut** from the descriptor lists (`paper.tex:177`, `:913`), or implement it.
- **Soften** "WallAvoider … far above every other reliable Karel task" (the CI overlaps TopOff). Add CIs throughout.
- **Prefilter section:** add the dirty-pool and Sokoban replications. Note that "Full ± 0.076" is the population SD (ddof=0; the sample SD is 0.083).
- **State** that unlocks are deadline-driven and that held-out evaluation uses a stochastic policy (4 episodes per layout).

## 6. Deferred (not feasible by Oct 6; scoped by today's de-risk)
- **Calibrated grader transfer (7/9/11 → 9/13/17).** Risky: the 9/13/17 success label has no within-cell reliability (0.21), and RFs don't extrapolate in size.
- **Within-cell curriculum (E5-a).** Only 7/9/11 has reliable within-cell labels, and flat training already wins there.
- **Knob-free generator (E5-b) and oracle_v2.** These are the right next paper, but each needs its own de-risk plus about 30 seeds per arm.
- **The labeling-paradox fix for the hard regime:** label difficulty with a *curriculum-trained* reference policy (non-circular). This is promising follow-up work.

## 7. Open questions
1. **Was the abstract registered on OpenReview by Sep 29?** If not, check whether AISTATS 2027 is still possible at all.
2. **F3 history:** what did the geometry-family conditions use in the Jun 18 `results_cluster_lc91317` run?
3. **Approval** to edit `curriculum_harness.py` (aliases keep old results readable) and run about 100 jobs on this Mac overnight.
4. **Is local `paper.tex` the latest?** The handoff mentions `aistats2027_submission_v6.zip` (`main.tex`). Please drop it into the repo if it's newer.
5. **Framing:** measurement-first (recommended), or keep the curriculum as the headline with downgraded claims?
