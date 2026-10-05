# HANDOFF: Rescuing the AISTATS 2027 paper — analysis, code audit, and experiment plan

**From:** the Claude session that formatted and reviewed the paper.
**To:** the Claude Code instance working in this repo (the code that was exported as `curriculum_export.zip`).
**Paper:** *Predicting Task Difficulty from Geometry: Execution-Free Curricula for Reinforcement Learning* (AISTATS 2027 submission, anonymous).
**Latest paper source:** the user has `aistats2027_submission_v6.zip` (`main.tex`, `checklist.tex`, `references.bib`, figures). Ask the user to drop it into the repo, for example under `paper/`, if you need to edit the paper.

Read this whole document before changing code. Sections 3 and 4 explain *why* the current results do not support the story. Section 6 is the work plan. Section 7 says which results we hope to see and what to write for each possible outcome. Section 10 lists the rules that keep the work honest.

---

## 0. TL;DR

1. **The story we want:** "Scoring generated environments by running agents is expensive. We build a *grader* that scores difficulty from the environment's layout alone, give theory for why layout predicts difficulty, give a test for when the grades can be trusted, and show that a curriculum built from the grader works as well as an expensive oracle that must run agents first."
2. **The paper does not tell that story right now.** The curriculum result (Section 6.6, the LavaCrossing table) does **not** use our grader. The "Geometry" condition is a hand-written formula, and on the grid sizes used it reduces to **"train on the smallest grids first."**
3. **The "Oracle" baseline is broken.** 82 of 90 oracle labels are exactly 1.0, a tie. Python's stable sort then keeps the generation order (size 9 first), so the oracle's easy bin is *also* "smallest grids first." That is why Geometry (0.50) and Oracle (0.48) match: they are nearly the same ordering, and it comes from grid size.
4. **An earlier version of the Geometry score failed (0.085) and the paper reports only the later version (0.503).** Random, Oracle and Probe are byte-identical across the two runs; only the conditions that use the geometry score changed. This must be disclosed, or, better, replaced with a pre-registered fitted grader.
5. **On smaller grids (7/9/11), flat training with no curriculum beats every curriculum** (0.43–0.57 vs 0.19–0.21 for Geometry). This is not in the paper.
6. **Big opportunity we found:** on 7/9/11, difficulty varies **inside** each (size, river-count) cell from about 0 to about 1 (std 0.3–0.4). The designer's knobs (size, rivers) explain little; the *layout* matters. A grader that reads the layout can capture this; a knob ordering cannot. This is where our method can genuinely win.
7. **The plan (Section 6):**
   - fix the oracle (graded labels, random tie-breaks);
   - implement the real three-layer grader for LavaCrossing and call it from the harness;
   - add fair baselines (random-order curriculum, size-only, rivers-only, size+rivers);
   - run the main comparison;
   - add a setting with no usable designer knobs, where the grader should win;
   - report everything, including the negative results.
8. **Rules (Section 10):** pre-register the grader and the primary metric *before* running. Never change a scoring formula after seeing results without reporting both versions. Log the scoring method and git hash in every result JSON.

---

## 1. Context

### 1.1 What the paper currently contains (main text, 8 pages)

1. **Introduction:** curriculum methods score candidate environments by running agents, which is expensive. Can difficulty be read from the layout? Contributions list.
2. **Related Work:** a short version; the full version is in Appendix A.
3. **Problem Setting:** environments share states and actions but differ in layout (which moves are possible) and reward. The value gap splits into a reward term (handled by successor features) and a dynamics term.
   - **Theorem 2:** the dynamics term is bounded by the Frobenius distance between the adjacency matrices of the "kinodynamic feasibility graphs", plus a shared-edge residual.
   - **Corollary 3:** the residual is zero for deterministic grids.
   - **Hypothesis 4 ("Geometric Lipschitz"):** difficulty is Lipschitz in a descriptor distance.
   - **Three-layer descriptor (Section 3.1):** (1) local counts, (2) graph reachability (diameter, minimum cut, bridges, articulation points, degree statistics), (3) normalized-Laplacian eigenvalues.
4. **Framework:**
   - the regressor and curriculum score 𝒰 (target-difficulty band plus ridge-leverage coverage);
   - the reference solver (DSL hill climbing) and difficulty label D_HC = 1 − mean best reward over R = 12 seeds;
   - the reliability gate (split-half, Spearman–Brown);
   - Predictor 1, a random forest on the descriptor, and Predictor 2, a low-cost probe;
   - shaping for sparse rewards.
5. **Experimental Setup:** Karel (12 tasks), MiniGrid (LavaGap, RedBlueDoor, PutNear), Sokoban (Boxoban).
6. **Results:**
   - **6.1 Karel:** labels are reliable but geometry fails (CV ρ ≈ 0), except WallAvoider (0.42).
   - **6.2 MiniGrid:** sparse rewards make the label degenerate; shaped LavaGap gives ρ_rel 0.79 and CV ρ 0.76.
   - **6.3 Sokoban:** ρ_rel 0.985, CV ρ 0.35, geometry + probe 0.44.
   - **6.4 Main result:** two gates (graded reward, geometric causation); predictability tracks the nature of the difficulty.
   - **6.5 Prefiltering on LavaGap:** subsets of 10 match the full pool at 2.7–15× less compute, but curated = random.
   - **6.6 Ordering on LavaCrossing (PPO):** Random 0/10 seeds, Geometry 0.50 (7/10), Oracle 0.48 (7/10), Adaptive-coarse 0.54 (8/10). There is a caveat paragraph saying the Geometry score uses MiniGrid's knobs.
7. **Conclusion and Limitations.**

### 1.2 Score from a simulated AISTATS review

**4/10 (borderline reject)** as it stands. That would be about 5 for a reviewer who sees only the PDF, and 4 for one who sees the code. **Target after this work: 6–7.**

Strengths:
- a timely question;
- the reliability gate is a real contribution (single-run labels give ρ ≈ 0.03; 5 of 12 Karel tasks fail);
- WallAvoider is a clever within-domain control;
- the paper is honest about its negatives.

Weaknesses:
- **W1:** the story is "when is difficulty predictable?" (a measurement paper); the curriculum is an afterthought, which is the opposite of the intended story.
- **W2:** the curriculum result doesn't use our grader.
- **W3:** the theory is loosely connected; the bound is numerically vacuous and the experiments test a random forest, not the bound.
- **W4:** small samples; the task classification was decided after measurement.
- **W5:** missing baselines (random-order curriculum, designer-knob ordering, search-node-count proxies, policy information capacity).

---

## 2. The story we want, written as testable claims

The paper should be organized around these claims. Every experiment below exists to support or refute one of them.

- **C1 (Theory):** for layout-parameterized families with deterministic dynamics, the dynamics part of the value gap is controlled by how much the feasibility graph changes. So graph features of the layout *should* carry difficulty information. *Already in the paper (Theorem 2, Corollary 3).*
- **C2 (Trust):** the grader is only meaningful when the difficulty label is reliable and the difficulty is spatially caused. The reliability gate and the three-domain study tell a user when to trust the grader. *Already in the paper (Sections 6.1–6.4); keep it, but frame it as "when to trust the grader".*
- **C3 (Grader quality on the curriculum domain):** on the curriculum domain (LavaCrossing), the grader predicts *graded, reliable* difficulty labels from the layout alone (cross-validated ρ, compared with label reliability). **Missing today.**
- **C4 (Main result: grader-built curricula work):** a curriculum ordered by the grader:
  - (a) beats flat training;
  - (b) beats a random-order curriculum with the same bins and gates, so *ordering* matters;
  - (c) matches an expensive, properly built oracle at a small fraction of its cost;
  - (d) matches or beats designer-knob orderings without being told the knobs.

  **Missing today:** the current table has none of (b), (d), a working oracle, or our grader.
- **C5 (Where it wins):** when the designer's knobs are uninformative (difficulty varies within a knob cell, or the generator exposes no knob), the grader beats knob orderings and random orderings. **Missing today; this is the strongest possible evidence for the story.**
- **C6 (Scope, honest):** when flat training already succeeds (small grids), a curriculum doesn't help. **We have this data; it must be reported.**

---

## 3. Code audit: issues found (with evidence)

All paths are relative to the repo root (`curriculum_export/`). Each finding is something a reviewer could discover from released code, or that affects whether a claim in the paper is true.

### F1. "Geometry" in Section 6.6 = "smallest grids first" (size ordering)

- **File:** `scripts/curriculum/difficulty_scoring.py`, `geometric_score()`:
  ```python
  size_term = (layout.size - 7) / 2.0
  return f["est_rivers"] + size_term + f["detour"] / 100.0
  ```
- On the paper's sizes {9, 13, 17}, `size_term` ∈ {1, 3, 5} while `est_rivers` ∈ {1, 2, 3}. Size dominates.
- Reconstructing the bins from the oracle cache's layout signatures (`bin_layouts`, 3 quantile bins over 90 layouts) gives:
  - Geometry bin 0 = all 30 size-9 layouts
  - bin 1 = all 30 size-13 layouts
  - bin 2 = all 30 size-17 layouts
- The function's own docstring says the size term was designed for sizes {7, 9, 11}, where it would be {0, 1, 2}.
- `est_rivers` is read from the grid, but it equals MiniGrid's `num_crossings` tier N. So the score is built entirely from the generator's two difficulty knobs plus a 1/100 tiebreak.
- **Consequence:** the 6.6 headline does not test the paper's descriptor, regressor or theory. The paper's caveat paragraph admits part of this, but understates it: the ordering is purely by grid size.

### F2. The oracle labels are degenerate, and the oracle's ordering is a sort-order artifact

- **Cache used by the paper's run:** `scripts/curriculum/cache/oracle_b5fce063f3_r1e12.json`, built with 1 reference seed and 12 evaluation episodes (`run_cluster.sh` sets `ORACLE_FLAGS="--oracle-ref-seeds 1 --oracle-eval-episodes 12"`).
- 90 layouts; **82/90 (91.1%) have label exactly 1.0**. Mean by cell:
  - size 9: 0.70 (N1), 0.90 (N2), 0.825 (N3)
  - size 13 and 17: about 0.94–1.0
- `bin_layouts()` uses Python `sorted()`, which is stable, so ties keep pool order. `enumerate_pool()` iterates sizes [9, 13, 17], then tiers, then seeds. Result:
  - Oracle bin 0 = 28/30 size 9
  - bin 1 = 28/30 size 13
  - bin 2 = 29/30 size 17
- So **Oracle ≈ Geometry ≈ size ordering**, and the 0.48 vs 0.50 "match" is an artifact.
- `run_cluster.sh` even says: *"The oracle's failure is structural (ref PPOs cannot solve hard layouts -> degenerate labels)."* The paper calls this condition "expensive ground truth", which is not accurate.
- **Contrast:** the 7/9/11 cache `oracle_6ccc84561c_r3e20.json` (3 reference seeds, 20 episodes) is graded. Only 2% of labels are 1.0, there are 33 distinct values, and the median is 0.31. So a graded oracle *is* achievable; it just failed on the harder 9/13/17 pool with 1 reference seed.

### F3. The Geometry score was changed after a failed run; only the successful version is reported

| Condition | `results_cluster_lc91317/` (Jun 18) | `results_oraclefree_lc91317/` (Jun 19, used in paper) |
|---|---|---|
| random | 0.024 | 0.024 (identical) |
| oracle | 0.484 | 0.484 (identical) |
| probe | 0.220 | 0.220 (identical) |
| combined | 0.370 | 0.395 |
| **geometry** | **0.085 (1/10 reached)** | **0.503 (7/10)** |
| adaptive_coarse | 0.166 (2/10) | 0.537 (8/10) |
| adaptive_fine | 0.426 | 0.328 |

- Only the conditions that consume `geometric_score` changed.
- The module docstring at the top of `difficulty_scoring.py` still says: *"`geometric_score` = river_count (dominant) with detour as a normalized tiebreaker"*, with no size term.
- **Most likely history:** v1 ordered by rivers, which mixes small and large grids in every bin, and failed. v2 added `(size-7)/2`, which turns it into size ordering, and succeeded.
- The result JSONs do not record which formula was used. **Confirm this with the user or from git history** (`git log -p scripts/curriculum/difficulty_scoring.py`).
- **Consequence:** as written, this is post-hoc selection of the scoring rule. It must be disclosed or superseded by a pre-registered grader (Section 6).

### F4. On smaller grids, flat training beats every curriculum (unreported)

| Run | Sizes | Seeds | random (flat) | geometry | oracle | probe |
|---|---|---|---|---|---|---|
| `results_v1_buggygate/` | 7, 9, 11 | 10 | **0.434** | 0.194 | 0.371 | 0.444 |
| `results_lc7911neg/` | 7, 9, 11 | 3 | **0.567** | 0.211 | 0.234 | 0.218 |

- `v1_buggygate` had a gate bug, per its folder name; `lc7911neg` is after the fix, but has only 3 seeds.
- **Interpretation:** a curriculum helps only when flat training fails (9/13/17). That is a legitimate and useful scope finding, but leaving it out looks like setting selection.
- **Must be rerun cleanly (10+ seeds, current harness) and reported.**

### F5. The "Random" baseline is *no curriculum*, not a *random-order curriculum*

- In `curriculum_harness.py`, `run_condition()` sets `is_curriculum = condition != "random"`, and Random unlocks every bin at step 0.
- So "Geometry ≫ Random" confounds two effects:
  - (i) training on a restricted, growing subset (any curriculum mechanics);
  - (ii) the specific order.
- **Needed:** a `random_curriculum` condition (random scores, same bins, gates and deadlines).

### F6. The curriculum is driven by the deadline, not the competence gate

- In the paper's run, unlock times for Geometry and Oracle are almost always at env-steps 350k–400k and then 700k–800k. The deadline is 0.25 × 1.5M = 375k, rounded up to the 50k checkpoint.
- The gate (0.55 success on the active set) is rarely met.
- **Effectively, every curriculum is a fixed schedule:** bin 0 for about 400k steps, bins 0–1 for about 400k, then all bins.
- This is fine, but **say so in the paper**, log `unlock_kind` in the JSON (it's computed but not saved), and consider a sensitivity check on `--bin-deadline-frac` (0.15 / 0.25 / 0.35).

### F7. The paper's best prediction number doesn't use the three-layer descriptor

- **Karel:** `prog_policies/karel_tasks/karel_curriculum_difficulty_predictor.py` computes the real graph and spectral features: `build_grid_graph`, `bridges_and_articulations`, `graph_laplacian_eigs`, diameter, eccentricity, dead-ends and so on.
- **Sokoban:** `scripts/sokoban/sokoban_core.py::extract_features` reuses those helpers. ✔
- **MiniGrid:** uses hand-picked task features, not the descriptor.
  - LavaGap: `scripts/option_a_lavagap.py::lavagap_features` gives gap_x, gap_y, gap_col, gap_row_offset, path_len, manhattan, detour, size. These drive the headline ρ = 0.76.
  - PutNear and RedBlueDoor: `scripts/minigrid_shaped_extra.py` gives object coordinates.
- **LavaCrossing curriculum:** rivers / size / detour only (F1).
- **Prefiltering (6.5):** `scripts/curriculum/lavagap_curriculum.py` curates by the *probe* band (`d_probe`) plus coverage over `_DESCRIPTOR_COLS = [gap_x, gap_y, gap_col, gap_row_offset, path_len, detour]`. That isn't the three-layer descriptor either.
- **Minimum cut:** the paper lists a "minimum cut between spawn and goal" feature. **It is not implemented anywhere** (grep for min_cut / maxflow / networkx finds nothing).
- **Consequence:** the claim "the descriptor singled out by the theory predicts difficulty" is only tested on Karel and Sokoban. Either rerun MiniGrid prediction with the real descriptor (preferred; Section 6, E7) or change the text.

### F8. The fitted grader exists but is unused, and it leaks the tier label

- `difficulty_scoring.py::fit_geometry_oof()` is exactly the paper's "Predictor 1" for LavaCrossing: an out-of-fold random forest on `feature_vector()`. **The harness never calls it.**
- `feature_vector()` includes `layout.n`, which is MiniGrid's tier (num_crossings). It must not be used as a feature: it is a designer label, not something read from the layout.

### F9. Smaller inconsistencies to fix in the paper

- The paper says the oracle costs "541 s for one seed (~1620 s at the default of three)". The run used **1** reference seed, so state that the oracle used in the table had 1 reference seed and 12 evaluation episodes, and that this produced degenerate labels.
- **Sizes:** the harness default is `--sizes 7 9 11`; the paper run used `9 13 17` via `run_cluster.sh`. Make sure the config table matches what was run.
- **Evaluation policy:** `eval_layouts` uses a *stochastic* policy (`deterministic=False`) with 4 episodes for each held-out layout. That's fine, but state it.

---

## 4. Claim-versus-evidence table (current state)

| Paper claim | Supported by current code and results? | Fix |
|---|---|---|
| Theorem 2 / Corollary 3 | Yes (math) | — |
| Reliability gate; single-run labels irreproducible | Yes | — |
| Karel: reliable but unpredictable by the descriptor; WallAvoider 0.42 | Yes (three-layer descriptor) | — |
| Shaped LavaGap CV ρ 0.76 *with the three-layer descriptor* | **No**: hand gap features | E7: rerun with the descriptor, or reword |
| Sokoban 0.35 with the descriptor | Yes | — |
| "Minimum cut" feature | **No**: not implemented | Implement (E2) or remove from the text |
| Prefiltering: curated = random | Yes, but curation used the probe band and gap features | Reword; optional rerun |
| 6.6: execution-free ordering matches the oracle | **No**: both are size ordering; the oracle is degenerate; the score changed after a failure | E1–E4 |
| Curricula help | Only on 9/13/17; not on 7/9/11 | E6: report |

---

## 5. Hypotheses we want the new experiments to confirm (pre-register these)

Write these into `scripts/curriculum/PREREGISTRATION.md` **before running anything**, with the date and git hash.

- **H1 (grader quality):** on LavaCrossing, graded oracle labels are reliable (split-half ρ_rel ≥ 0.6 across reference seeds), and the three-layer grader predicts them out of sample with CV ρ meaningfully above 0. We hope for ≥ 0.5, and close to ρ_rel. This must also hold **within (size, N) cells**, where the knobs are constant.
- **H2 (ordering matters):** grader curriculum > random-order curriculum (same mechanics) on final held-out success and on steps-to-target.
- **H3 (cheap = expensive):** grader curriculum ≈ fixed oracle curriculum (no significant difference), at less than 10% of the oracle's labeling cost *on the curriculum pool*.
- **H4 (no knowledge of knobs needed):** grader curriculum ≥ size+rivers knob ordering on 9/13/17, without the grader being given N.
- **H5 (where the grader wins):** in the knob-free setting (E5), grader > knob ordering and grader > random-order curriculum.
- **H6 (scope):** on 7/9/11, flat training ≥ every curriculum. We expect to confirm this and report it.

**Primary metric** (fix it now): final held-out success at 1.5M steps, averaged over 10 seeds, with 95% CI. Pre-specified tests:
- Mann–Whitney one-sided, grader > random_curriculum;
- two-sided, grader vs oracle;
- TOST equivalence with a ±0.10 margin for "matches".

**Secondary metrics:** steps-to-target (0.30), area under the held-out learning curve, wall-clock including labeling cost.

---

## 6. Work plan, in detail

Do these in order. **E0–E4 are the minimum** to make the story true. E5 is what makes the paper strong. E6–E8 close the remaining gaps.

### E0. Bookkeeping (do first, about 1 hour)

1. **Write `PREREGISTRATION.md`** (Section 5) and commit it.
2. **Make every result JSON self-describing.** In `curriculum_harness.py::run_condition()`, add to the `result` dict:
   - `scoring_method` (string, e.g. `"grader_rf_v1"`), and `scoring_config` (feature list, label type, calibration set spec);
   - `git_hash` (`subprocess.check_output(["git","rev-parse","HEAD"])`);
   - all CLI args (`vars(args)`);
   - the **bin composition**: for each bin, the counts of (size, N) cells and the list of layout signatures (or seeds);
   - `unlock_kind` and the env-steps of each unlock (they're computed but not saved);
   - the fraction of tied scores in the condition's scores.
3. **Use a new results directory for every run family** (`results_v2_<tag>/`). Never overwrite.
4. **Record the history in `notes/`.** Write down that the `results_cluster_lc91317` vs `results_oraclefree_lc91317` change came from editing `geometric_score` (confirm from git), so the paper can disclose it.

### E1. Fix the oracle (graded labels, random tie-breaking)

**Goal:** an honest "expensive ground truth" baseline whose labels actually vary and whose ordering isn't a sort artifact.

1. **Graded label.** Replace "1 − success" with a graded per-layout difficulty, computed from reference-policy rollouts. Pick **one** primary option before looking at results; log the others.
   - **Option A (recommended): progress-based.** For each evaluation episode, record the closest BFS distance to the goal the agent reached, avoiding lava (`_bfs` with `block_lava=True`, computed from each visited cell to the goal; cache the distance map). Then:
     ```
     progress = (d_start − d_min) / d_start      # 1.0 on success
     ```
     This mirrors the paper's shaping equation. Set label = 1 − mean progress.
   - **Option B:** success rate averaged over several checkpoints of the reference run (for example at 200k, 400k and 600k steps), which is an area under the curve.
   - **Option C:** the same as now, but with a longer reference budget and more reference seeds, until fewer than 30% of labels are tied. That may be expensive on 17×17.
2. **More reference seeds:** use **3** reference seeds (the default `--oracle-ref-seeds 3`) and **≥ 20** evaluation episodes. Compute and log the **oracle label reliability**: split-half Spearman across reference seeds (seeds {0, 1} vs {2}, or bootstrap the episodes), with Spearman–Brown correction. *Applying our own reliability gate to the oracle is a nice touch for the paper.*
3. **Random tie-breaking** for every condition. In `bins_from_scores` / `bin_layouts`, sort by `(score, random_tiebreak)` where the tiebreak is drawn from `np.random.default_rng(seed)`. Report the tie fraction.
4. **Cache key** must include the label type and number of reference seeds (`_pool_hash` already takes the pool spec; extend the filename).
5. **Cost accounting:** record wall-clock and env-steps spent building the oracle (reference training plus evaluation rollouts).

**Diagnostics to print and save:**
- the distribution of oracle labels (overall and within each cell);
- the fraction tied;
- label reliability;
- the bin composition by cell.

**Acceptance criteria:**
- tied fraction < 30%;
- ρ_rel ≥ 0.6;
- the bins are not a pure function of size.

If size 17 is still nearly all 1.0 under Option A, that's real (17×17 with 3 rivers is genuinely hard), but progress labels should still spread.

### E2. Implement the real grader for LavaCrossing (three-layer descriptor plus fitted regressor)

**Goal:** the paper's actual method, applied to the curriculum domain, with no designer labels.

1. **New feature function** in `scripts/curriculum/difficulty_scoring.py`:
   ```python
   def descriptor_features(size, n, seed) -> dict
   ```
   - Get `img, start, goal` via `_grid_objects` (already exists).
   - **Feasibility graph:** nodes = cells that are not WALL and not LAVA (stepping into lava ends the episode, so lava cells are infeasible). Build it with `build_grid_graph` from `prog_policies/karel_tasks/karel_curriculum_difficulty_predictor.py`. Import it the way `scripts/sokoban/sokoban_core.py` does; the helper is coordinate-agnostic and takes `(a, b)` tuples.
   - **Layer 1 (local):**
     - grid width/height/area (these are readable from the layout, so allowed; see the variants below);
     - number of free cells; number of lava cells; lava fraction; wall fraction;
     - start→goal Manhattan distance;
     - number of free neighbors of the start; dead-end count (degree-1 cells); corridor cells (degree 2); branch cells (degree ≥ 3).
   - **Layer 2 (reachability):**
     - BFS shortest path start→goal avoiding lava (= the existing `detour`); BFS ignoring lava (`free_path`); `detour_excess`;
     - diameter and mean pair distance of the start's component;
     - number of components; fraction of free cells reachable from the start;
     - start eccentricity;
     - bridges and articulation points (`bridges_and_articulations`);
     - mean / min / max / variance of degree;
     - **minimum s–t cut** (edge connectivity between start and goal). Implement with `networkx.minimum_edge_cut` or `networkx.edge_connectivity(G, s, t)`, and add `networkx` to requirements. This also fixes F7's missing feature. Also compute the **vertex** cut size, `networkx.node_connectivity(G, s, t)`, which counts single-cell chokepoints. For LavaCrossing (rivers with single gaps) these should be strongly informative.
     - "Number of chokepoints on the shortest path": articulation points that lie on the BFS shortest path. This is cheap and probably very predictive.
   - **Layer 3 (spectral):** the smallest k = 6 eigenvalues of the normalized Laplacian of the **start's component** (`graph_laplacian_eigs`). λ₂ (algebraic connectivity / Cheeger) is the key one. Because component sizes differ, also log λ₂ × (number of nodes) as a size-normalized variant.
   - **Do NOT include** `n` (the tier) or `est_rivers` in the primary grader. `est_rivers` is a hand-designed readout of the generator knob; keep it only for a diagnostic variant.
2. **Grader variants.** Pre-register exactly one as primary.
   - **`grader` (PRIMARY):** all three layers, *including* grid size (it's legitimately visible in the layout), *excluding* `n` and `est_rivers`.
   - **`grader_noknobs`:** all three layers *excluding* grid width/height/area *and* `est_rivers`. This shows the graph structure alone recovers difficulty.
   - **`grader_spectral_only`:** Layer 3 only. An ablation that ties the result to the theory.
3. **Regressor:** `RandomForestRegressor(n_estimators=300, random_state=seed)`; there's no scaling needed. Optionally add `GradientBoostingRegressor` as a robustness check.
4. **Where the training labels come from.** This is critical for the "execution-free" claim. **The grader must not be trained on oracle labels of the same layouts it then orders.** Pre-register one of these:
   - **(a) Calibration-transfer, RECOMMENDED for the story.** Train the grader on graded labels from a **separate calibration pool** of cheaper layouts, then apply it to the curriculum pool *without any rollouts on the curriculum pool*.
     - **Calibration pool:** sizes {7, 9, 11} × N {1, 2, 3}, *disjoint seeds* (for example `seed_start=50_000`), 10–20 layouts in each cell.
     - **Labels:** graded oracle labels (E1, Option A) from reference PPO runs on the calibration pool. We already know 7/9/11 produces graded labels (the `r3e20` cache has 2% ties and median 0.31).
     - **Curriculum pool:** sizes {9, 13, 17}, the usual seeds. The grader sees **no labels** from 13 or 17. **This tests extrapolation to bigger, unseen sizes from layout structure alone**, which is a strong, story-aligned result if it works.
     - **Cost:** report the calibration cost once, as amortized. The cost *at use time* on the curriculum pool is milliseconds.
   - **(b) Cross-fitting on the curriculum pool** (`fit_geometry_oof`, 5 folds). Each layout's score comes from a forest trained on the other 80% of the pool, using that pool's oracle labels. This is *not* execution-free at use time, because labels were needed on 80% of the pool. Use it only as a diagnostic ("how good is the grader when given in-distribution labels?"), not as the headline.
5. **Grader quality report (supports H1, C3)**, saved as JSON and CSV:
   - CV ρ on the calibration pool (5-fold);
   - **transfer ρ:** grader predictions on the curriculum pool vs that pool's (fixed) oracle labels, which are used *only for evaluation*;
   - **within-cell ρ:** the same Spearman computed inside each (size, N) cell, then averaged. This is the key evidence that the grader sees more than the knobs;
   - comparison against label reliability ρ_rel;
   - feature importances (top 10).
6. **Harness integration** (`curriculum_harness.py::initial_scores`): add conditions
   - `"grader"`, `"grader_noknobs"`, `"grader_spectral_only"`, each loading a trained model from `cache/grader_<config_hash>.pkl`;
   - a `--build-grader-only` flag, mirroring `--build-oracle-only`, which builds the calibration labels and fits and caches the model.

### E3. Fair baselines (new conditions in the harness)

Add these to `CONDITIONS` and `initial_scores()`. All except `flat` use the same curriculum mechanics: 3 quantile bins, competence gate 0.55, deadline 0.25, random tie-breaking.

| Condition | Score | Purpose |
|---|---|---|
| `flat` (rename the current `random`) | none; everything unlocked at step 0 | no curriculum |
| `random_curriculum` | `rng.random()`, with curriculum mechanics on | isolates the *order* effect (F5) |
| `size_only` | grid size + random tiebreak | the dominant designer knob |
| `rivers_only` | N (tier) + random tiebreak | MiniGrid's own difficulty knob |
| `knobs` (= current `geometric_score`) | rivers + (size−7)/2 + detour/100 | the current "Geometry"; report it **as a designer-knob baseline** |
| `oracle_v2` | graded oracle from E1 | expensive ground truth |
| `grader` (primary), `grader_noknobs`, `grader_spectral_only` | E2 | our method and its ablations |
| `probe` | existing random-walk probe | keep for continuity |
| `adaptive_coarse` | start from **`grader`** (not `knobs`), re-score at unlocks | keep; it was the best condition before |

Drop `adaptive_fine` and `combined` from the main table unless needed. They can stay in the appendix with the old numbers, clearly labeled.

### E4. Main curriculum run on {9, 13, 17}

- **Same settings as the paper's run**, unless noted: 1.5M steps, `eval_every` 50k, 10 layouts in each cell for training (90) and 4 for held-out (36), 4 evaluation episodes, gate 0.55, deadline 0.25, target 0.30, 3 navigation actions.
- **Seeds:** 10 at minimum; **20 if compute allows**. The old run had high variance, with SDs around 0.3.
- **Conditions:** all of E3.
- **Launch:** adapt `scripts/curriculum/cluster/run_cluster.sh`:
  - build the oracle_v2 and grader caches first, serially;
  - then run the (condition × seed) jobs in parallel;
  - pass `--results-dir results_v2_main`.
- **Compute estimate:** each job is about 1,500 s at one thread (from the logs). 10 conditions × 10 seeds = 100 jobs ≈ 42 CPU-hours, about 3.5 h wall time at 12 parallel jobs.
  - oracle_v2 build: 3 reference seeds × about 600k steps plus evaluation, about 30–60 min serial.
  - grader calibration labels on 7/9/11: 3 reference seeds on the calibration pool, about 30–60 min.
- **Analysis script:** extend `aggregate_results.py` to output:
  - the main table: final success mean ± 95% CI, the number of seeds reaching the target, steps-to-target, area under the curve, labeling cost, total cost;
  - pre-registered tests (Section 5);
  - a per-cell final success table;
  - bin composition for each condition (shows what each ordering actually does);
  - unlock timing (gate vs deadline).

### E5. A setting without usable designer knobs (where the grader should win)

This is the experiment that makes the paper. The current table can't separate "reads the layout" from "knows the knobs". Two designs; do the de-risk for both and pick one.

**E5-a: Within-cell curriculum (cheapest; uses existing envs).**
- Fix a single cell, for example size 11 and N = 2, or a narrow band such as size 11 with N ∈ {2, 3}. Generate many distinct layouts (for example 150 train and 60 held-out).
- The knobs are now constant, so `size_only`, `rivers_only` and `knobs` all collapse to random ordering.
- From the 7/9/11 cache, **within-cell difficulty is large** (example: size 11, N = 3 has mean 0.53, range 0–0.97, SD 0.34). So there's real signal for a layout-reading grader.
- **De-risk first:**
  1. **Label reliability within the cell:** is ρ_rel ≥ 0.6? If within-cell variation is mostly PPO noise, this design fails.
  2. **Grader CV ρ within the cell** on the calibration data.
  3. **Does flat training fail or struggle in this cell?** A curriculum only helps if flat training is hard. If flat training succeeds, pick a harder cell (size 13, N = 3) or reduce the budget.

**E5-b: Knob-free procedural generator (strongest story; more work).**
- Write a small MiniGrid env `RandomLavaFieldEnv(size=13)`:
  - place lava cells by a random process (random rectangles, line segments with random gaps, or blobs) with a hidden density drawn from a range;
  - random start and goal on opposite sides;
  - reject unsolvable layouts with BFS.
- There's no tier argument and nothing a designer sorts by. This mimics "an LLM wrote this environment program".
- Include a **`density_knob`** baseline (sort by the hidden density). That's the only knob, and it's fair to give it to a baseline. The grader shouldn't see it directly; the lava fraction is visible, which is fine.
- **De-risk the same three things as E5-a**, and also that PPO with the existing small CNN can solve the easy layouts.

**Conditions for E5:** flat, random_curriculum, (density_knob for E5-b), oracle_v2, grader, grader_noknobs, probe.

**Hoped-for result:** grader ≈ oracle_v2 > random_curriculum ≈ knob-based > flat (H5).

### E6. Scope control on {7, 9, 11}

- Rerun E4's condition set on sizes {7, 9, 11}, 10 seeds, current harness (the gate bug is fixed).
- **Expected (H6):** flat ≥ every curriculum.
- Report it in the paper as: *"Curricula help only when flat training fails; our grader is a tool for that case."* Pair it with the E1 finding that the 7/9/11 oracle is graded.
- **Optional:** a budget sweep on 9/13/17 (0.75M, 1.5M, 3M) to show where flat training starts to succeed.

### E7. Make the prediction results use the three-layer descriptor everywhere (fixes F7)

- **LavaGap, PutNear, RedBlueDoor:** recompute predictive power using the three-layer descriptor (the E2 graph features adapted to each env's grid) instead of hand-picked task features.
  - The labels already exist: these scripts store raw per-seed shaped rewards, and `minigrid_shaped_extra.py` reads shards with `shaped_0..shaped_{R-1}`. Find the label CSVs and shards in the repo; if missing, regenerate.
  - Report **both** numbers: hand features (current) and the three-layer descriptor (new).
  - If the descriptor is lower than 0.76, say so honestly. The story still works if it is clearly above Karel.
- **Implement the minimum-cut feature** in the shared helper (`karel_curriculum_difficulty_predictor.py`) so Karel and Sokoban also get it. Then either rerun those or state in the paper that min-cut was added only for MiniGrid. Simplest alternative: drop "minimum cut" from the paper text for Karel and Sokoban.

### E8. Optional: prefiltering on a domain where curation can matter

- `scripts/curriculum/sokoban_dirty_pool.py` and `cluster/sweep_sokoban_dirty_pool.sh` already exist: contaminated Boxoban pool; random / band / coverage / both; k ∈ {15, 30, 50, 80}. **No results are in the export.** If it was run, find the outputs; if not, it's ready to launch.
- If curation beats random on Sokoban, that's a strong addition to Section 6.5 ("curation helps when the descriptor space is rich and the pool is dirty; it doesn't on LavaGap because the space collapses to the gap position"). If not, it's a clean confirmation of the null.
- **Also:** the LavaGap prefilter curates by the *probe* band, not the grader. Either relabel this in the paper or rerun with the grader as the band score.

---

## 7. Results we want, and what to write for each outcome

### 7.1 The best case (what to aim for)

| Condition (9/13/17, 10–20 seeds) | Final success | Labeling cost on the curriculum pool |
|---|---|---|
| flat | ≈ 0.02 | 0 |
| random_curriculum | clearly lower than grader (e.g. 0.2–0.3) | 0 |
| size_only / rivers_only / knobs | ≈ grader | 0 (needs designer knowledge) |
| oracle_v2 (graded, 3 reference seeds) | ≈ grader | high (≈ 30–60 min of PPO) |
| **grader (calibrated on 7/9/11, applied to 9/13/17)** | **≈ oracle_v2 (≈ 0.5)** | **milliseconds (plus amortized calibration)** |
| grader_noknobs | ≈ grader | milliseconds |

And in E5 (knobs uninformative): **grader ≈ oracle_v2 > random_curriculum ≈ knob-based > flat.**

**Story this supports:** "Our grader, trained once on small cheap environments, reads difficulty from the layout of new, larger environments it has never seen. A curriculum built on it matches a properly built expensive oracle at a tiny fraction of the cost, without knowing the generator's difficulty knobs. When those knobs are uninformative, it beats knob-based orderings."

### 7.2 Likely outcomes and how to handle each

- **A. The grader matches the oracle and beats random_curriculum on 9/13/17, but knobs do just as well.** → Fine. Say that the grader *rediscovers* the designer's knobs from the layout without being told them (show feature importances and grader_noknobs). Then E5 is where it *beats* knobs. Lead with E5 in the paper.
- **B. The grader beats random_curriculum but loses to the oracle.** → Report the gap and the cost ratio (accuracy vs cost trade-off). Check whether grader quality (H1 transfer ρ) is the bottleneck. If within-cell ρ is low, the descriptor is missing something: add the chokepoint-on-path and vertex-cut features (E2), and retry **once**, reporting both versions.
- **C. random_curriculum ≈ grader ≈ oracle (order doesn't matter, only curriculum mechanics).** → An important, honest finding. The benefit then comes from *restricting and growing the training set*, not from difficulty ordering. The story becomes weaker. Check whether that's because unlocks are deadline-driven (F6): try a stricter gate, lower deadline pressure (`--bin-deadline-frac 0.4`) and more bins (5). If ordering still doesn't matter, reframe the paper toward C2 (when to trust a grader) and prefiltering, and report the curriculum as a negative result.
- **D. The oracle_v2 labels are still degenerate on 9/13/17.** → Use progress labels (E1 Option A); they can't all tie unless the agent never moves. If the reference agents still learn nothing on 17×17, reduce the hard end (sizes {9, 11, 13}) and say why.
- **E. The grader fails to transfer from 7/9/11 to 9/13/17.** → Report it. Use cross-fitting (E2 4b) as the in-distribution upper bound and be explicit that the grader then needs labels on part of the pool. Lean on E5-a (within-cell, same distribution), where calibration and use share sizes.
- **F. In E5, nothing beats random_curriculum.** → The grader-for-curriculum story is not supported in knob-free settings. Report it, and narrow the claim to "matches the oracle cheaply where difficulty is spatial".

### 7.3 What to drop or relabel in the current paper regardless of outcome

- **Rename "Geometry"** in the old table to **"designer knobs (size + rivers)"**. Don't present it as our method.
- **Rename "Oracle"** in the old table to **"oracle (1 ref seed; 91% tied labels)"**, or drop it and replace it with oracle_v2.
- **Disclose F3:** the earlier rivers-first formula and its 0.085 result.
- **Report F4 / E6:** flat training wins on 7/9/11.
- **Rename "Random"** to **"flat (no curriculum)"**.

---

## 8. Paper rewrite plan (after the experiments)

**New title options:**
- *Grading Environments Before Training: Execution-Free Difficulty Prediction for Curriculum Learning*
- *Reading Difficulty from Layout: Execution-Free Curricula for Reinforcement Learning*

The current title can stay if C4 holds.

**New structure (8 pages):**

1. **Introduction (1 page).** Problem: scoring cost. Our grader. When to trust it. **Main result:** grader-built curricula match an expensive oracle at a tiny cost and beat knob orderings when knobs are uninformative. List C1–C6 as contributions.
2. **Related Work (0.4 page).** As now.
3. **Setting and Theory (1.3 pages).** As now (Theorem 2, Corollary 3, descriptor). Cut the Hypothesis 4 discussion to 2 sentences.
4. **The Grader (0.8 page).** Descriptor, regressor, calibration-transfer protocol (E2 4a), reliability gate, shaping.
5. **When Can the Grader Be Trusted? (1.3 pages).** Compressed current 6.1–6.4: Karel / MiniGrid / Sokoban; two gates; WallAvoider. Framed as a *user guide*: trust the grader when labels are reliable and difficulty is spatial. Use three-layer descriptor numbers (E7).
6. **Curricula from the Grader (2 pages, the main result).**
   - **6.1 Setup** (LavaCrossing, all conditions, costs).
   - **6.2 Grader quality on LavaCrossing** (H1: CV, transfer and within-cell ρ).
   - **6.3 Main table** (E4).
   - **6.4 Knob-free setting** (E5).
   - **6.5 When curricula don't help** (E6, one paragraph).
   - **6.6 Prefiltering** (short; current 6.5, plus E8 if run).
7. **Limitations and Conclusion (0.4 page).**

**Appendix additions:**
- the pre-registration;
- the full history of the old runs (F3/F4), with tables;
- bin compositions;
- grader feature importances;
- the oracle-label reliability analysis;
- the deadline-vs-gate unlock analysis.

---

## 9. Priorities and timeline

Check the AISTATS 2027 deadline with the user first. If time is short, do the tiers in order and stop where time runs out.

| Tier | Items | Effort | What it buys |
|---|---|---|---|
| **1 (must)** | E0, E1, E2 (primary grader, calibration-transfer), E3, E4 at 10 seeds; disclose F3/F4 | about 1–2 days of code + about 4–6 h compute | the curriculum claim becomes true or honestly refuted; integrity issues fixed |
| **2 (strongly recommended)** | E5-a (within-cell), E6 | about 0.5–1 day + about 4 h compute | "where the grader wins" plus scope |
| **3 (if time)** | E5-b (knob-free generator), E7 (descriptor everywhere, min-cut), E8 (Sokoban prefilter), 20 seeds | 1–3 days | the strongest version of the paper |

If only Tier 1 is possible and the result is outcome A or B (Section 7.2), the paper is still much stronger than now. If it's outcome C, reframe as described.

---

## 10. Rules for this work (integrity guardrails)

1. **Pre-register** the grader features, label type, calibration protocol, primary metric and tests (Section 5) **before** running E4 or E5. Commit with a timestamp.
2. **Never change a scoring rule after seeing curriculum results without reporting both versions.** If you must iterate (Section 7.2 B), report "v1 (pre-registered)" and "v2 (after inspection)" as separate rows.
3. **Never use designer labels as grader features** (`layout.n`, `est_rivers` in the primary grader, tier names, hidden density).
4. **Never train the grader on labels from the layouts it orders** in the headline condition (calibration-transfer only). Cross-fitting is a diagnostic.
5. **Break ties randomly**, with a seed, everywhere.
6. **Keep every result directory.** Report negative and earlier runs in the appendix.
7. **Every JSON** records the scoring method, config, git hash, args, bin composition, unlock events and tie fraction (E0).
8. **Costs:** report labeling cost (env-steps and wall-clock) separately from training cost, for every condition.

---

## 11. Quick reference

### 11.1 Key numbers from the current results (`results_oraclefree_lc91317/corrected_aggregate.csv`, used in the paper)

| Condition | Seeds | Reached 0.30 | Final success (mean ± 95% CI) | Wall time (s) |
|---|---|---|---|---|
| random (flat) | 10 | 0 | 0.024 ± 0.030 | 1396 |
| oracle (1 ref seed) | 10 | 7 | 0.484 ± 0.205 | 1469 (+541 s oracle build) |
| geometry (= size order) | 10 | 7 | 0.504 ± 0.210 | 1492 |
| probe | 10 | 3 | 0.220 ± 0.203 | 1552 |
| combined | 10 | 6 | 0.395 ± 0.192 | 1561 |
| adaptive_fine | 10 | 7 | 0.328 ± 0.196 | 2603 |
| adaptive_coarse | 10 | 8 | 0.537 ± 0.156 | 1529 |

### 11.2 Commands used to verify the audit findings (re-run to confirm)

```bash
# F2: oracle label degeneracy
python - <<'EOF'
import json, numpy as np
d = json.load(open('scripts/curriculum/cache/oracle_b5fce063f3_r1e12.json'))
v = np.array(list(d.values())); print(len(v), (v >= 0.999).mean())
EOF
# F3: compare the two 9/13/17 result folders by condition (final_held_success mean)
# F4: results_v1_buggygate/ and results_lc7911neg/ on sizes [7,9,11]
# F6: unlock steps = curve entries where 'frontier' changes (mostly 350k-400k and 700k-800k)
```

### 11.3 Files to touch

| File | Changes |
|---|---|
| `scripts/curriculum/difficulty_scoring.py` | `descriptor_features()`, grader training and loading, random tie-break in `bin_layouts`, remove `n` from the grader features, fix the docstring |
| `scripts/curriculum/curriculum_harness.py` | new conditions (E3), graded oracle (E1), `--build-grader-only`, richer JSON (E0), rename random → flat |
| `scripts/curriculum/aggregate_results.py` | pre-registered tests, per-cell tables, bin composition, cost columns |
| `scripts/curriculum/cluster/run_cluster.sh` | build the caches, then launch the E4 / E5 / E6 grids |
| `prog_policies/karel_tasks/karel_curriculum_difficulty_predictor.py` | add the min-cut / vertex-cut helper (shared) |
| new: `scripts/curriculum/PREREGISTRATION.md`, `scripts/curriculum/derisk_within_cell.py`, optional `prog_policies/minigrid_tasks/random_lava_field.py` | — |

### 11.4 Things to ask the user

1. The AISTATS 2027 deadline and the compute available (cores on the cluster).
2. Confirm the history behind F3 (which `geometric_score` version produced `results_cluster_lc91317`).
3. Whether the Sokoban dirty-pool sweep (E8) was ever run, and where its outputs are.
4. Which label option (E1 A / B / C) to pre-register. Recommendation: **A, progress-based**.
