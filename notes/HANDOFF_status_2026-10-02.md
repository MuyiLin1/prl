# Handoff: paper status, results, running experiments (2026-10-02)

**Paper:** `paper.tex` (latest version), AISTATS 2027. Abstract submitted; **full paper due Tue Oct 6, 23:59 AoE**.
**Read alongside:**
- `HANDOFF_paper_rescue.md`: the original teammate audit.
- `HANDOFF_paper_rescue_REVIEW.md`: verification of that audit plus the first de-risk.
- `scripts/curriculum/PREREGISTRATION.md`: what each experiment tests and how it is judged, written before results.

---

## 1. The story

**Part 1 (strong now): cheap difficulty prediction.**
- We predict how hard a generated level is **from its layout alone, in milliseconds, without running anything**.
- **Where difficulty is spatial** (layout, chokepoints, paths), our grader is the best execution-free method we tested. It matches or beats a trained RL agent playing each level up to 16 times.
- **Where difficulty is logical** (multi-step program logic, as in most Karel tasks), it doesn't work. An LLM judge does somewhat better there, but nothing does well.
- We also certify that the difficulty labels are repeatable before predicting them: the reliability gate.

**Part 2 (pending tonight): using it for curriculum learning, cheaply.**
- Order training levels with the grader, and plug the grader into the field's best curriculum method (SFL) to replace most of its expensive scoring rollouts.
- **If part 2 is weak, part 1 alone is the paper**, and curriculum becomes a short "first use" section or a limitation.

**Open question to research:** are most tasks studied today spatial? MiniGrid navigation, Procgen mazes, Sokoban, robot navigation and level generators look spatial. Karel-style program tasks, text and web agent tasks do not. A short survey of the major curriculum and environment-design benchmarks (PLR, ACCEL, SFL, JaxNav, Kinetix, Craftax, XLand-MiniGrid) classifying their difficulty as spatial vs logical would show how much of the field our method covers. Not done yet.

---

## 2. How the experiments work (plain version)

**Difficulty label (the "truth"):** run a solver or agent on a level many times; difficulty = how often it fails (or how much search it needs). It is expensive, which is exactly what we want to avoid at use time.

**Domains**

| Domain | What it is | Label |
|---|---|---|
| Karel (12 tasks, 80–200 grids each) | Grid puzzles solved by a small program | Program-search failure, averaged over 12 searches |
| MiniGrid LavaGap (60 levels) | Reach the goal through one gap in a lava wall | Program-search failure (shaped reward) |
| MiniGrid LavaCrossing (90 + 90 levels) | Cross 1–3 lava rivers; grid sizes 7–17 | Failure rate of trained PPO agents |
| Sokoban: 3,000 public Boxoban levels | Push boxes onto targets | **Published by others**: A* search effort, A* solution length, Boxoban tier (the "medium" tier = levels DRC agents failed) |
| Sokoban: 200 levels | Same | Our budgeted-search label |

**Metrics**
- **Spearman ρ:** do the predicted and true *orderings* (easy to hard) agree? 1 = same order, 0 = random, negative = backwards.
- **AUC:** for a yes/no label (hard tier or not), the chance the method rates a random hard level above a random easy one. 0.5 = coin flip, 1 = perfect.
- **Ceiling:** labels are noisy, so even a perfect predictor can't reach 1. The max is √(label reliability), for example 0.89 on LavaGap.

**Methods compared on identical levels**

| Method | What it is | Cost |
|---|---|---|
| **Ours (grader)** | Layout features (local counts, reachability graph, Laplacian spectrum) → random forest | ms; nothing run |
| Designer knobs | Generator settings: grid size + river count (LavaCrossing), Boxoban tier | Free, if the generator has them |
| Kartal 2016 | Published Sokoban difficulty formula | Free |
| LLM judge | Local Qwen2.5-7B shown the ASCII layout + rules, asked "difficulty 0–9" (expected digit) | ~1 s per level on GPU |
| Trained-agent rollouts | Train a PPO agent (600k steps), play each level k times, use its failure rate. This is what SFL/PLR-style curriculum methods do. | Expensive |

n/a cells: LavaGap has no knob, no agent-based label and no published formula. On the 3,000-level Sokoban set the tier *is* a label, and the label comes from a solver, not an agent.

---

## 3. Results so far

### 3.1 Difficulty prediction (`data/benchmarks/difficulty_comparison_table.csv`)

| Domain (label) | **Ours** | Kartal 2016 | Qwen 7B | Knobs | Trained-agent rollouts |
|---|---|---|---|---|---|
| Sokoban 3,000 public (A* search effort) | **0.75** | 0.41 | 0.18 | n/a | n/a |
| Sokoban 3,000 public (tier, AUC) | **0.80** | 0.64 | 0.57 | n/a | n/a |
| LavaGap (ceiling 0.89) | **0.57** (hand features 0.77) | n/a | 0.29 | n/a | n/a |
| LavaCrossing 9/13/17 (ceiling 0.76) | **0.46** | n/a | 0.28 | 0.35 | 0.36 (1 ep) → 0.44 (16 eps) |
| LavaCrossing 7/9/11 (ceiling 0.75) | 0.40 | n/a | 0.20 | 0.24 | 0.42 (1 ep) → 0.49 (16 eps) |
| Sokoban 200 (our label) | 0.32 | n/a | 0.01 | 0.57 (tier, made by running agents) | n/a |

**Karel (logic-heavy):**
- Ours wins on WallAvoider 0.36, OneStroke 0.35 (label too noisy to trust) and TopOff 0.28.
- Qwen wins on DoorKey 0.41, PathFollow 0.27, CleanHouse 0.25, plus Snake, Maze, StairClimber and FourCorners by small margins.
- Most values are below 0.3, with error bars of about ±0.13. Neither method reads Karel well.

**Takeaways**
1. On spatial tasks our grader beats every execution-free alternative and equals or beats up to 16 rollouts of a trained agent.
2. Layout reads spatial difficulty; an LLM reads logical difficulty. The two are complementary.
3. Honest caveats:
   - Hand-picked gap features beat our general descriptor on LavaGap (0.77 vs 0.57).
   - Boxoban's tier beats us on the 200-level pool, but the tier needs agents to compute.

### 3.2 Label measurement (the reliability gate)
- **One run is mostly noise.** A single search run's label is only 1–35% signal. The needed seed count runs from 4 (WallAvoider) to over 140 (Harvester, Seeder) for 0.7 reliability. 5 of 12 Karel tasks fail the reliability bar.
- **The gate catches bad oracles.** The old paper's PPO "oracle" had 91% of labels tied. A progress-based label is noise on small grids, and on big grids it is reliable but mechanical: it measures where the first gap is.
- Details: `HANDOFF_paper_rescue_REVIEW.md` §2.

### 3.3 Corrections owed in the paper, whatever the outcome
- **The old curriculum headline was misleading.** "Geometry ≈ Oracle" was really "smallest grids first", and the oracle's labels were 91% tied.
- **Disclose** the Jun 18 failed run (geometry 0.085) and that the scoring changed before the Jun 19 rerun. We don't know exactly what changed.
- **Report** that flat training beats curricula on small grids.
- **Remove "minimum cut"** from the paper; it was never implemented.
- **Use our descriptor** for LavaGap (0.57) rather than presenting the hand features as our method.
- Full list: `HANDOFF_paper_rescue_REVIEW.md` §5.

---

## 4. What is running (started 2026-10-02 ~10:35)

Chain, each stage starting automatically when slots free up (8 CPU jobs max):

| Stage | Experiment | Output | Est. done |
|---|---|---|---|
| 1 | **Main curriculum**, LavaCrossing 9/13/17, 1.5M steps, 20 seeds each: `flat` (no curriculum), `size_order` (old "geometry"), `random_curriculum` (curriculum with random order), `rivers_only` (knob), **`grader`** (ours, fit only on small grids) | `scripts/curriculum/results_v2_lc91317/` | ~6–7 pm Oct 2 |
| 1b | Scope check on 7/9/11: flat vs size_order, 10 seeds (expect flat to win) | `results_v2_lc7911/` | with stage 1 |
| 2 | **SFL** (the field's best curriculum method) vs SFL with 20 scored candidates vs **SFL+grader** (20 rollouts to calibrate, grader picks from 100), 10 seeds each | `results_v2_sfl_lc91317/` | ~11 pm Oct 2 |
| 3 | Extra seeds 20–39 for stage 1 | `results_v2_lc91317/` | ~6–7 am Oct 3 |

**Machine safety:**
- `scripts/benchmarks/watchdog.sh` pauses jobs if free memory drops below 30% and kills ours below 15%. Its log is `logs_queue/watchdog.log`.
- An earlier run crashed the Mac (12 jobs, an uncapped LLM, other projects running). The limits above are the fix.
- If you also run other heavy jobs (e.g. `DMwithLLM/wiki_qwen`), lower the load to 6 jobs by restarting the queues with `CAP=6`.

---

## 5. How to check and read the results

All commands from the repo root `~/code/temp-prl`.

**Is it still running? Is the machine OK?**
```bash
pgrep -f "curriculum_harness|sfl_harness" | wc -l    # running jobs (8 while busy, 0 when done)
tail -3 logs_queue/watchdog.log                        # free memory %; PAUSE/KILL lines = trouble
ls scripts/curriculum/results_v2_lc91317 | wc -l       # finished main runs (100 = stage 1 done, 200 = stage 3)
ls scripts/curriculum/results_v2_sfl_lc91317 | wc -l   # finished SFL runs (30 = done)
```

**Results:**
```bash
cd scripts/curriculum
python3 summarize_v2.py main    # stage 1 + scope check (pre-registered: seeds 0-19)
python3 summarize_v2.py sfl     # stage 2
python3 summarize_v2.py extra   # stages 1+3 together (seeds 0-39)
```

**Reading the output:**
- `final mean`: average held-out success at the end of training (0 to 1). Higher is better; it's the share of new layouts the agent solves.
- `95% CI`: the range the true average likely falls in. **If two conditions' ranges overlap a lot, don't claim a difference.**
- `reached 0.30`: how many seeds crossed 30% held-out success at any point.
- `X vs Y: diff …, MWU p = …`: the pre-registered test. **p < 0.05 means X is reliably better than Y.** A larger p means "not shown", not "equal".

**What each comparison means:**

| Line | Question | If p < 0.05 |
|---|---|---|
| `random_curriculum vs flat` | Does training in stages help at all? | Curriculum mechanics help |
| `size_order vs random_curriculum` | Does the *order* matter (by the designer's knob)? | Yes |
| **`grader vs random_curriculum`** | Does *our* cheap order beat a random one? | **Key curriculum result** |
| `grader vs size_order` (two-sided) | Is ours different from knob ordering? | Ours better or worse; check the sign of `diff` |
| `flat vs size_order` (7/9/11) | Do curricula hurt when flat training already works? | Expected; report as scope |
| **`sfl_grader vs sfl_small`** | Equal scoring budget: does the grader pick better levels? | **Key "cheaper" result** |
| `sfl_grader vs sfl` + bootstrap CI | Does ours match full SFL with ~1/5 of the rollouts? | Matching claim if the CI's lower end is above −0.10 (see PREREGISTRATION) |

**Writing rules** (from the pre-registration): don't change any scoring rule after seeing results, and report runs that didn't work.

**If something breaks:**
- **Stop everything:** `pkill -f "run_v2_local|cpu_queue|gpu_queue|curriculum_harness|sfl_harness|llm_judge"`
- **Restart the CPU chain** (it skips finished runs): `cd scripts/curriculum && nohup bash cluster/run_v2_local.sh > logs_v2/launcher.log 2>&1 & sleep 3; nohup bash cluster/cpu_queue.sh > logs_v2/cpu_queue.log 2>&1 &`
- **Restart the watchdog:** `nohup bash scripts/benchmarks/watchdog.sh >/dev/null 2>&1 &`
- **Lose at most one run:** a run that dies mid-way leaves no file and is redone on restart.

---

## 6. Still to do (priority order)

1. **Add baselines to the difficulty table:**
   - **CNN on the raw grid.** Answers "is it our features, or any learned model?" The most important one. About 20 min on the GPU; `scripts/train_difficulty_predictor.py` has a starting point.
   - **Our features + LLM score combined.** Likely the best overall predictor, since the two read different kinds of difficulty. About 10 min.
   - **Agent-population embeddings (Mahajan et al. 2024).** Embed tasks via a population of agents. Expensive; at minimum, cite and discuss it as rollout-based related work.
   - Optional: a larger LLM judge (GPT-4-class API) to show the LLM result isn't just a weak 7B model.
2. **Research question:** classify the major current RL curriculum benchmarks as spatial vs logical difficulty, to show how much of the field the method covers (§1).
3. **Analyze stages 1–3** with `summarize_v2.py`, then write the curriculum section by the pre-registered decision rules.
4. **Rewrite `paper.tex`** around the two-part story:
   - Theory short.
   - Section on label measurement (reliability gate).
   - Section on the comparison table (§3.1), with scope: spatial vs logical.
   - Section on curriculum (stages 1–2), or a short first-use section if results are weak.
   - Cite and position against Krsteski & Meyer 2026, "Predicting Task Difficulty Without Rollouts" (arXiv 2608.05797): Spearman 0.40 in-distribution, 0.23 on new benchmarks, LLM-agent tasks, no label-reliability check.
5. **Apply the corrections in §3.3.**

## 7. Where things are

| What | Path |
|---|---|
| Sokoban public benchmark (features, labels, results) | `data/benchmarks/boxoban_astar/`, `scripts/benchmarks/sokoban_astar_benchmark.py` |
| LLM judge prompts and scores | `data/benchmarks/llm_prompts/`, `scripts/benchmarks/{make_llm_prompts,llm_judge_mlx}.py` (MLX venv at `~/.venvs/mlx`, model at `~/.venvs/qwen25-7b-instruct-4bit`) |
| Comparison table / rollout curve | `scripts/benchmarks/{difficulty_table,rollout_equivalence}.py` → `data/benchmarks/*.csv` |
| Grader code | `scripts/curriculum/difficulty_scoring.py` (`descriptor_features`, `fit_grader`), calibration labels `scripts/curriculum/cache/grader_calib_p7911_r6.json` |
| Curriculum / SFL code | `scripts/curriculum/{curriculum_harness,sfl_harness,summarize_v2}.py`, launchers in `scripts/curriculum/cluster/` |
| Audit scripts and de-risk data | `notes/rescue_audit_2026-10-01/` (also original copies of the edited harness files) |
| Python env for experiments | `/Users/linmuyi/miniconda3/envs/llm_gs/bin/python` |
