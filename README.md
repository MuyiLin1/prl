# Export: predicting level difficulty before running it

This folder is a self-contained snapshot of the code, results, seeds, and logs
behind `paper/paper.tex` (AISTATS 2027 submission, title *Predicting Level Difficulty
Before Running It: Reliable Labels, Layout Graders, and Cheaper Curricula*).
**Everything here is a copy.** `MANIFEST.csv` maps each exported file to its original
repo path. Last updated 2026-10-05: sections 08--11, the new logs and notes, and
refreshed copies of every script that changed after the first export (2026-10-01).

```
export/
├── README.md            ← this file
├── MANIFEST.csv         ← export path → original repo path, size
├── paper/               paper.tex, references.bib, original proposal PDF
├── figures/             figures the paper includes; figures/superseded/ = old curriculum plots
├── code/                runnable code snapshot (same relative layout as the repo)
│   └── notes/rescue_audit_2026-10-01/   audit + de-risk scripts that produced the PPO labels
├── results/             all experiment outputs, grouped by paper section
│   ├── 00_early_exploration/            (pre-reliability-gate work; not in paper)
│   ├── 01_karel/                        Table tab:karel
│   ├── 02_minigrid/                     LavaGap labels, foils
│   ├── 03_sokoban/                      Sokoban-200 labels, two-solver check
│   ├── 04_probe/                        Appendix probe
│   ├── 05_prefilter_lavagap/            Appendix selection-vs-random null
│   ├── 06_neural_curriculum_lavacrossing/   SUPERSEDED (old geometry-vs-oracle result; see app:history)
│   ├── 07_followups_not_in_paper/
│   ├── 08_prediction_benchmarks/        Tables tab:prediction, tab:ablation, tab:worth, Fig. fig:worth
│   ├── 09_curriculum_v2_lavacrossing/   Table tab:curriculum-v2 + within-size check (pre-registered)
│   ├── 10_sfl_lavacrossing/             Table tab:sfl
│   └── 11_maze/                         Appendix app:maze (pre-registered knob-free test)
├── logs/                run logs (benchmarks queue, curriculum v2, maze, RedBlueDoor)
└── notes/               handoff documents and the running paper plan (dated decision log)
```

---

## 1. What the paper claims, in one paragraph

How much of a level's difficulty can be read from its layout before anything is run?
(1) **Measurement:** difficulty labels are often unreliable (a single search run is 1--35%
signal on Karel), so every prediction claim is gated on split-half reliability, and
√ρ_rel is reported as the ceiling. (2) **Prediction:** a random forest on layout,
reachability, and Laplacian-spectrum features ("the grader"), motivated by a bound tying
changes in dynamics to changes in the feasibility graph, is the best execution-free
predictor on spatial domains (Sokoban, LavaGap, LavaCrossing), matches 16 trained-agent
rollouts on the hardest pool, and needs fewer labels than a CNN on Sokoban. It fails on
logic-heavy Karel. (3) **Curriculum (pre-registered):** the grader's easy-to-hard order
beats a random order on LavaCrossing 9/13/17 and ties ordering by grid size while ranking
levels within a size; inside SFL it matches standard SFL with about a quarter of the
scoring rollouts. A knob-free maze test found no setting where a curriculum was both
needed and sufficient.

Pre-registrations: `code/scripts/curriculum/PREREGISTRATION.md` (main run, Addendum A
for SFL and extra seeds, Addendum B/B1--B3 and C for mazes). The dated decision log is
`notes/PAPER_PLAN_2026-10-02.txt`.

---

## 2. Paper element → code → result files

All paths below are relative to `export/`. "Repo" commands assume you run them
from the **repo root** (or from `export/code/` after copying the needed result
files back next to it; see §4).

### Table `tab:karel`: Karel 12-task screen (N=200 / 80, R=12)

| What | Where |
|---|---|
| Labeling + geometry test | `code/scripts/averaged_label_pilot.py` (R-seed averaged HC label, reliability, best single feature, 5-fold RF CV ρ) |
| Cluster driver (8 tasks) | `code/scripts/cluster/run_karel_difficulty.sh` (GRIDS=200 REPS=12 MAXPROG=5000) |
| Descriptor features | `code/karel_curriculum_difficulty_predictor.py` (graph/spectral/reward features), `code/scripts/generate_multiseed_dataset.py` |
| 8-task screen output | `results/01_karel/full_screen_R12/karel_<Task>_grids.csv` (per-grid, per-seed) and `karel_<Task>_summary.csv`; combined in `karel_difficulty_summary.csv` |
| TopOff (N=200) | `results/01_karel/averaged_label_topoff200.csv` |
| Snake + Seeder (N=80) | `results/01_karel/averaged_label_screen_snake_seeder.csv` |
| TopOff + Harvester pilot (N=80) | `results/01_karel/averaged_label_pilot.csv` |
| "Single run ρ≈0.03" (seed-averaging appendix) | `code/scripts/stability_check.py`, `code/scripts/analyze_direction_b.py` → `results/01_karel/direction_b_500seeds.csv` (500 seeds, single HC run each) |
| **Single-run signal (ICC), runs for 0.7** (Table `tab:karel` columns 1 and 3) | `code/scripts/gstudy_karel.py` (one-way ANOVA from the per-grid mean/SD above; run it from `export/` with no arguments) |

Checked: `karel_difficulty_summary.csv` reproduces the reliability column (WallAvoider ρ_rel 0.817, DoorKey 0.709, Maze 0.702, …), and `gstudy_karel.py` reproduces every ICC and runs-needed value in the table.

### MiniGrid (§ MiniGrid, Appendix `app:foils`)

| What | Where |
|---|---|
| Shaped-reward wrapper (distance-to-goal) | `code/scripts/minigrid_shaped.py` |
| LavaGap, 3 difficulty definitions (partial-progress / success-rate / search-cost), N=60, R=12 | `code/scripts/direction_c_AB.py` → `results/02_minigrid/direction_c_AB.csv` |
| Earlier LavaGap version (R=8) | `code/scripts/option_a_lavagap.py` (its output `option_a_lavagap.csv` is not in the repo) |
| Sparse-reward degeneracy smoke test | `code/scripts/minigrid_smoke_test.py` |
| PutNear / RedBlueDoor foils (N=200, R=12) | `code/scripts/minigrid_shaped_extra.py`, drivers `code/scripts/cluster/run_minigrid_difficulty.sh` and `run_redbluedoor_sharded.sh` (16 shards) |
| Foil outputs | `results/02_minigrid/foils_putnear_redbluedoor/` (`minigrid_*_summary.csv`, `raw/redbluedoor_shard_*.csv`); logs in `logs/redbluedoor_shard_{0,1,2}.log` |
| Cross-domain table (old paper version; no longer in the paper) | `code/scripts/option_c_reframe.py` → `results/02_minigrid/option_c_cross_domain.csv` |

Checked: the summary gives PutNear ρ_rel 0.55 / CV 0.39 and RedBlueDoor 0.43 / 0.34, matching the paper.

### Sokoban: Sokoban-200 row of `tab:prediction`, Table `tab:sokoban-solver`, Figure `fig:sokoban-dsl-hc`

| What | Where |
|---|---|
| Level parsing + static descriptors | `code/scripts/sokoban/sokoban_core.py` |
| State-space reference solver (budgeted best-first) | `code/scripts/sokoban/solver.py` |
| Stratified 200-level pool + state-space label (70 unfiltered / 70 medium / 60 hard, R=12, budget 4000, `--seed 0`) | `code/scripts/sokoban/sample_and_label.py` → `results/03_sokoban/sokoban_fastpath_labels.csv` + `sokoban_fastpath_perseed.json` (per-seed rewards) |
| Probe (greedy/random push rollouts) | `code/scripts/sokoban/probe.py` |
| Geometry vs probe vs both (CV ρ 0.35 / 0.09 / 0.44) | `code/scripts/sokoban/predictor.py` (prints the table; no CSV output) |
| DSL + hill-climbing label (cross-check) | `code/prog_policies/sokoban/` (DSL with `solveStep` macro), `code/prog_policies/sokoban_tasks/boxoban.py` (0.8/0.2 shaping), `code/scripts/sokoban/hc_label.py`, cluster `code/scripts/sokoban/cluster/` → `results/03_sokoban/sokoban_hc_dsl_labels.csv` |
| Pilot (45 levels) | `results/03_sokoban/sokoban_pilot_labels.csv`, `sokoban_pilot_perseed.json` |
| Boxoban levels used | `results/03_sokoban/boxoban_levels_used/`: the 3 source files all 200 levels come from (`unfiltered/train/316.txt`, `medium/train/168.txt`, `hard/002.txt`), plus the Boxoban license |
| Figure | `figures/sokoban_dsl_hc.pdf` / `.png`. **No script in the repo generates this figure.** It appears to have been made ad hoc from the two label CSVs. |

### Probe appendix: Table `tab:probe`, truncated search ρ=+0.51

| What | Where |
|---|---|
| Fixed-library probe features | `code/scripts/cheap_rollout_features.py`, `code/scripts/explore_cheap_rollout.py` |
| Full-population evaluation (TopOff 200, Snake 80) + truncated-search + wall-clock | `code/scripts/probe_full_eval.py` → `results/04_probe/probe_full_topoff.csv`, `probe_full_snake.csv`, `lavagap_trunc_search.csv` |
| Earlier 50-grid join (superseded; "optimistic") | `code/scripts/probe_vs_geometry.py` + `results/04_probe/multiseed_with_rollout.csv` |

### Prefilter: Table `tab:curriculum` (shaped LavaGap, N=150, M=80, 36 runs/condition)

| What | Where |
|---|---|
| Experiment | `code/scripts/curriculum/lavagap_curriculum.py` |
| Sweep driver (paper config: POOL=150 HELDOUT=80 NITER=250 TRAINSEEDS=6 RANDOMDRAWS=6 CURATEDDRAWS=6, k∈{10,15,20,30,50}, 20 shards) | `code/scripts/curriculum/cluster/sweep_lavagap_curriculum.sh` → `run_lavagap_curriculum.sh` |
| **Paper results** | `results/05_prefilter_lavagap/paper_fair_ensemble/lavagap_curriculum_fair_k{10,15,20,30,50}_results.csv` (one row per run: condition, draw, search_seed, held-out reward, env evals, wall time) |
| Earlier single-subset comparison (the "k=15 advantage that did not survive") | `results/05_prefilter_lavagap/earlier_single_subset/` and a differing copy from the old `curriculum_export/` in `earlier_single_subset_curriculum_export_copy/` |

Checked: group means match the table. The "Full ± 0.076" uses the population s.d. (ddof=0) over the 6 full-pool runs. Pandas' default (ddof=1) gives 0.083.

### Neural curriculum, OLD version (superseded; described in Appendix `app:history`)

This is the earlier geometry-vs-oracle result (0.50 vs 0.48). An audit found the geometry bins were exactly grid size, the oracle was 91% tied, and there was no random-order control. It is kept for the record only; the paper's curriculum results are in §§ 09--11 below.

| What | Where |
|---|---|
| Difficulty scores (geometry / pool enumeration / bins) | `code/scripts/curriculum/difficulty_scoring.py` |
| PPO harness, 7 conditions | `code/scripts/curriculum/curriculum_harness.py` (SB3 PPO) |
| Driver (sizes 9 13 17, 1.5M steps, 1 oracle ref seed, 12 eval eps) | `code/scripts/curriculum/cluster/run_cluster.sh`; local version `run_multiseed.sh` |
| De-risk checks that motivated the setup | `derisk_lavagap_difficulty.py`, `derisk_lavacrossing_difficulty.py`, `probe_regime.py` |
| **Old results** (superseded): 7 conditions × seeds 0–9 | `results/06_neural_curriculum_lavacrossing/paper_results_oraclefree_lc91317/<condition>_seed<N>.json` (full learning curve per checkpoint + headline fields) and `corrected_aggregate.csv` |
| Cached oracle labels | `results/06_neural_curriculum_lavacrossing/oracle_cache/oracle_*_r1e12.json` (1 ref seed × 12 episodes, the paper config) and `oracle_*_r3e20.json` (default 3 × 20) |
| Aggregation | `code/scripts/curriculum/aggregate_results.py --results-dir … --target 0.30` |
| Figures | `code/scripts/curriculum/plot_curriculum_results.py` (reads `results_oraclefree_lc91317/` next to itself, writes to `images/`) |
| Superseded runs | `results/06_neural_curriculum_lavacrossing/superseded/`: `results_v1_buggygate/` (600k steps, gate bug), `results_lc7911neg/` (sizes 7/9/11, 600k steps, negative result, which motivated larger grids), `results_cluster_lc91317/` (same config, run before the final oracle-free version; the name suggests the non-oracle conditions were not yet fully oracle-free), `results/` (3-condition, seed-0 smoke run) |

Checked: `corrected_aggregate.csv` matches the old table exactly (Geometry 0.5035 ± 0.21, 7/10, Oracle 0.484, Adaptive-coarse 0.5368, 8/10, wall 1492 / 2603 s, …). The old figures are in `figures/superseded/`.

### Prediction: Tables `tab:prediction`, `tab:domains`, `tab:ablation`, `tab:worth`, Figure `fig:worth` → `results/08_prediction_benchmarks/`

| What | Where |
|---|---|
| Main comparison table (ours, Kartal, LLM, knobs, ceiling) | `code/scripts/benchmarks/difficulty_table.py` → `difficulty_comparison_table.csv` |
| Bootstrap 95% CIs in the `tab:prediction` caption | `code/scripts/benchmarks/prediction_table_ci.py` → `prediction_table_ci.csv` |
| Public Sokoban 3,000 (published A\* effort / length, tier) | `code/scripts/benchmarks/sokoban_astar_benchmark.py`; data `boxoban_astar/` (`benchmark_levels.csv`, `benchmark_results_layout_methods.csv`, source `*_valid.csv.gz` + dataset `README.md`) |
| CNN on the raw grid (all labels) | `code/scripts/benchmarks/cnn_baseline.py` → `cnn_baseline_sokoban3000.csv` |
| Few labels (`tab:worth` bottom, Fig. 1b) | `code/scripts/benchmarks/few_labels_sokoban.py` → `few_labels_sokoban.csv` |
| LLM judge (Qwen2.5-7B-Instruct, 4-bit, MLX) | prompts `code/scripts/benchmarks/make_llm_prompts.py`, scoring `llm_judge_mlx.py` (queue `gpu_queue.sh`) → `llm_prompts/<domain>.jsonl` and `<domain>_llm_qwen7b.csv` |
| Feature-layer ablation (`tab:ablation`) | `code/scripts/benchmarks/feature_ablation.py` → `feature_ablation.csv` |
| LavaCrossing labels (3 reference PPO agents per pool, 600k steps, 20 episodes/level) | `code/notes/rescue_audit_2026-10-01/derisk_oracle.py` → `lavacrossing_ppo_labels/p{7911,91317}_ref{0,1,2}.json` (per-level, per-episode success); analysis `derisk_analyze.py` |
| Grader vs k agent runs, LavaCrossing (`tab:worth` top) | `code/scripts/benchmarks/rollout_equivalence.py` → `rollout_equivalence_lavacrossing.csv` |
| Grader vs k search runs, LavaGap | per-search rerun `lavagap_rerun_k_runs/direction_c_AB_rerun{.csv,_per_search.npz}` (from `code/scripts/direction_c_AB.py`, which now saves the per-search matrix); curve in `make_figures.py` |
| Grader vs k search runs, Karel WallAvoider / DoorKey | per-search rerun `karel_rerun_k_runs/karel_<Task>_grids{.csv,_<Task>_per_search.npy}` (from `code/scripts/averaged_label_pilot.py`, driver `code/scripts/curriculum/cluster/overnight_queue.sh`); the reruns reproduce the stored labels exactly |
| Figure `fig:worth` | `code/scripts/benchmarks/make_figures.py` → `figures/worth_curves.{pdf,png}` |
| LavaGap descriptor used by the grader | `code/notes/rescue_audit_2026-10-01/lavagap_descriptor.py` |

### Curriculum (pre-registered): Table `tab:curriculum-v2`, within-size check → `results/09_curriculum_v2_lavacrossing/`

| What | Where |
|---|---|
| Pre-registration | `code/scripts/curriculum/PREREGISTRATION.md` (main section + Addendum A for seeds 20--39) |
| PPO harness (conditions `flat`, `random_curriculum`, `rivers_only`, `size_order`, `grader`) | `code/scripts/curriculum/curriculum_harness.py`, scores in `difficulty_scoring.py` |
| Grader calibration labels (7/9/11 pool, 6 reference agents) | `grader_calib_p7911_r6.json` |
| Drivers | `code/scripts/curriculum/cluster/run_v2_local.sh`, `cpu_queue.sh`, `overnight_queue.sh` |
| Results, 5 conditions × seeds 0--39 (9/13/17) | `results_v2_lc91317/<condition>_seed<N>.json` (curve, bins, unlocks, code hash) |
| Scope check 7/9/11 (flat vs size order, 10 seeds) | `results_v2_lc7911/` |
| Summary + tests | `code/scripts/curriculum/summarize_v2.py` |
| "Is the grader grid size in disguise?" (within-size Spearman, CI, permutation p, bin mix) | `code/scripts/curriculum/within_size_check.py` (reads `results_v2_lc91317/grader_seed0.json` and the 9/13/17 PPO labels) |

Checked from the exported JSONs: 20 seeds flat 0.055, random 0.11, rivers 0.15, grader 0.31, size 0.35, grader > random p = 0.016; 40 seeds flat 0.064, random 0.12, rivers 0.11, grader 0.35, size 0.43.

### SFL: Table `tab:sfl` (= `tab:curriculum-cost`) → `results/10_sfl_lavacrossing/`

| What | Where |
|---|---|
| Harness (`sfl`, `sfl_small`, `sfl_grader`) | `code/scripts/curriculum/sfl_harness.py` |
| Results, seeds 0--9 primary, 10--19 post hoc | `results_v2_sfl_lc91317/<condition>_seed<N>.json` (each records `scoring_steps` and per-refresh `mean_p` and buffer composition, used for the "why standard SFL is weak" paragraph) |

Checked: seeds 0--9 standard 0.07, cheap 0.06, grader 0.25.

### Knob-free maze test: Appendix `app:maze` → `results/11_maze/`

| What | Where |
|---|---|
| Harness (random-wall, corner, and perfect-maze generators; grader calibration; `--calib-p` transfer) | `code/scripts/curriculum/maze_harness.py` |
| Size pilots (random walls, then structured) | `results_maze_pilot/`, `results_maze_pilot3/`; drivers `cluster/maze_pilot.sh`, `maze_pilot3.sh` (`maze_pilot2.sh` was written but not run) |
| Calibration (90 mazes × 3 reference PPOs) | `calibration/maze_calib_s19_perfect_p0.5{,_ref0,_ref1,_ref2}.json` |
| Main run 19×19, p ≤ 0.5 (5 conditions × 20 seeds) | `results_maze_s19/`; driver `cluster/maze_main.sh` |
| Harder settings (Addendum C: p ≤ 0.25, p ≤ 0.10 pilots; main run not launched by rule) | `results_maze_c_pilot/p0.25`, `p0.1`; driver `cluster/maze_c.sh`; decision in `logs/maze/maze_c_chain.log` |

Checked: flat 0.40, random 0.35, wall count 0.47, path length 0.43, grader 0.42.

### Not in the paper's tables

* **`results/00_early_exploration/`**: work done before the reliability gate existed. It covers single-run HC ground truth over all Karel tasks (`run_all_hc.py` → `ground_truth_difficulty.*`), naive heuristics (`naive_heuristic_baseline.py`), an LLM zero-shot difficulty scorer (`llm_difficulty_baseline.py`), the correlation summary (`correlation_analysis.py`), the multi-seed datasets for `train_difficulty_predictor.py` and `comprehensive_evaluation.py`, and the CleanHouse RTD report (`karel_curriculum_difficulty_predictor.py`). These single-run labels are the ones the paper later shows are near-irreproducible (ρ≈0.03).
* **`results/07_followups_not_in_paper/`**: follow-ups on "can curation beat random?":
  * `lavagap_dirty_pool/`: contaminated-pool test (`code/scripts/curriculum/dirty_pool_curriculum.py`, analysis `code/scripts/analyze_dirty_pool.py`). The sweep over frac_trivial ∈ {0.20, 0.40, 0.55, 0.70} is in `handoff_sweep/` (`p020`…`p070`). `HANDOFF_README.md` is the write-up. Result: curation ties random, and the label is binary (zero "medium" tasks).
  * `graded_probe_step1/`: an attempt to make the probe graded (`graded_probe.py`).
  * `sokoban_dirty_pool/`: the same test on Sokoban (`sokoban_dirty_pool.py`), with root, step2, and step3-hard runs (pool / meta / shard CSVs).
  * `code/scripts/curriculum/ppo_curation.py`: a PPO-learner version of the curation test. No output files exist locally.

---

## 3. Seeds

* **Environment/grid seeds:** Karel and MiniGrid grids are `grid_seed = 0 … N−1` (the `grid_seed` column in every label CSV). Sokoban levels are identified by `level_id = <tier>/<file>/<index>`.
* **Search seeds:** R=12 HC seeds per grid. Per-seed values are kept in `karel_*_grids.csv`, `sokoban_*_perseed.json`, and the RedBlueDoor `raw/` shards. Reliability (split-half Spearman–Brown) is recomputed from these.
* **Prefilter:** each row of `lavagap_curriculum_fair_k*_results.csv` records `draw` and `search_seed`. Held-out layouts use disjoint seeds.
* **Neural curriculum:** training seeds 0–9 per condition (in the filename and the `seed` field). Held-out layouts start at seed 100000 (`--eval-seed-start`). The paper notes per-seed values are not bit-identical across CPU architectures; aggregates reproduce.
* **Curriculum v2, SFL, mazes:** seeds 0--19 primary and 20--39 secondary (curriculum), 0--9 primary and 10--19 post hoc (SFL), 0--19 (mazes); all runs were on one Apple M4 Pro. Ties in bin assignment are broken by a key seeded with the run seed. Reference PPO agents for labels use seeds 12345+r.

## 4. Running the code

```bash
cd code
conda env create -n llm_gs_env -f environment.yml && pip install -r requirements.txt
# Programmatic-only experiments in a torch-free env: apply the lazy-import patches first
patch -p1 < patches/01_search_space_lazy_import.patch
patch -p1 < patches/02_search_methods_lazy_import.patch
# Neural curriculum additionally needs: torch, stable-baselines3, gymnasium, minigrid
```

Scripts use repo-relative paths (`sys.path.append(".")`, `data/...`, root-level CSVs).
To re-run an **analysis** on exported results, copy the needed CSV back to the path
listed in `MANIFEST.csv`'s `original_path` column, relative to `code/`.
**Sokoban:** `hc_label.py` / `predictor.py` only need the 3 level files. Place
them under `code/data/boxoban/` using the same sub-paths. Re-*sampling* the pool
(`sample_and_label.py`) needs the full Boxoban corpus
(`git clone https://github.com/deepmind/boxoban-levels data/boxoban`).

Example commands for each experiment, from the repo root:

```bash
# Karel 8-task screen / TopOff / Snake+Seeder
bash scripts/cluster/run_karel_difficulty.sh
python scripts/averaged_label_pilot.py --tasks TopOff --grids 200 --output averaged_label_topoff200.csv
python scripts/averaged_label_pilot.py --tasks Snake Seeder --grids 80 --output averaged_label_screen_snake_seeder.csv
# MiniGrid
python scripts/direction_c_AB.py --grids 60 --reps 12            # shaped LavaGap
bash scripts/cluster/run_minigrid_difficulty.sh                   # PutNear, RedBlueDoor
# Sokoban
python scripts/sokoban/sample_and_label.py && python scripts/sokoban/predictor.py
python scripts/sokoban/hc_label.py --out data/sokoban_hc_dsl      # cluster: scripts/sokoban/cluster/
# Probe
python scripts/probe_full_eval.py
# Prefilter (paper sweep)
bash scripts/curriculum/cluster/sweep_lavagap_curriculum.sh
# Neural curriculum
cd scripts/curriculum && bash cluster/run_cluster.sh && python aggregate_results.py --results-dir results_oraclefree_lc91317 --target 0.30 && python plot_curriculum_results.py
```

The exact flags for each paper run are only partly recorded: in the cluster drivers' defaults and in the paper's
configuration tables. Where a run was launched by hand, the commands above are
reconstructed from the scripts' defaults and the paper text. Check them before treating
them as the exact invocation.

## 5. What's in `code/`, and what was left out

Included:

* `prog_policies/`: DSLs, environments, tasks (Karel, MiniGrid incl. shaped wrapper, Sokoban), and search methods (HC used as the reference solver).
* `scripts/`: every experiment and analysis script listed above. Result files that lived inside `scripts/curriculum/` were moved into `results/` in this export.
* `karel_curriculum_difficulty_predictor.py`, environment files, `slurm/`, and `patches/` (from `handoff/`).
* `leaps/` and `llm/`: inherited from the upstream LLM-GS / LEAPS repo, which `prog_policies` imports from.
* Upstream LLM-GS runners (`scripts/{LLM-GS,HC,CEBS,LEAPS,LLM-Revision}/run_*.sh`, `main.py`, `baseline.py`, `revision.py`) are kept because they're part of the codebase. The paper's experiments don't use them. The upstream README is `code/UPSTREAM_LLM-GS_README.md`.

Left out on purpose (still in the repo):

* `leaps/weights/` (49 MB), `params/` (9 MB VAE weights), `leaps/docs/`. These are only for the latent-space baselines (LEAPS / CEBS), which the paper doesn't use.
* The full Boxoban corpus (~170 MB). Only the 3 files used are included.
* `__pycache__`, `.DS_Store`, the old `curriculum_export/` folder and its zip. That folder was an earlier partial snapshot. Its unique result files are included above, and its scripts are older copies of `scripts/`.

## 6. Known gaps

* **Missing logs (old runs):** for runs before 2026-10-02, only the three RedBlueDoor shard logs exist. Newer runs have logs in `logs/` (`benchmarks_queue/`, `curriculum_v2/`, `curriculum_v2_crashed_20261002/` from a crashed first launch that was restarted, `maze/`).
* **No figure script:** `sokoban_dsl_hc.pdf` has no generating script in the repo.
* **Ad hoc analyses:** the Karel k-search numbers in `tab:worth` were computed from the exported `karel_rerun_k_runs/*_per_search.npy` with the same `search_curve` logic as `make_figures.py`, but not saved by a script. The ICC column of `tab:karel` uses the stored per-grid SD (population SD over 12 runs) as the within-grid variance; an unbiased correction lowers each ICC by about 0.01--0.02 and does not change any gate decision.
* **Print-only outputs:** `option_a_lavagap.csv` (earlier R=8 LavaGap run) and the `sokoban/predictor.py` numbers exist only as printed output, not as files.
* **No outputs:** `ppo_curation.py` has no output files.
