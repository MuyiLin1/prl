#!/bin/bash
# Always-on CPU queue: (1) wait for the main v2 curriculum launcher, (2) SFL x3 conditions x10 seeds,
# (3) extra curriculum seeds 20-39. 12 concurrent single-thread jobs.
cd "$(dirname "$0")/.."
PY=/Users/linmuyi/miniconda3/envs/llm_gs/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CAP=${CAP:-8}
busy() { pgrep -f "curriculum_harness.py|sfl_harness.py" | wc -l; }
run() { while [ "$(busy)" -ge "$CAP" ]; do sleep 10; done; "$@" & sleep 2; }
# do not compete with run_v2_local.sh while it still has jobs to launch
while [ ! -f logs_v2/lc91317_grader_seed19.log ] && pgrep -f run_v2_local.sh >/dev/null; do sleep 30; done
# start filling slots as soon as the first batch's jobs start finishing (no idle tail)
for s in $(seq 0 9); do
  for c in sfl_grader sfl sfl_small; do
    [ -f results_v2_sfl_lc91317/${c}_seed${s}.json ] && continue
    run nice -n 10 $PY -u sfl_harness.py --condition $c --seed $s > logs_v2/sfl_${c}_seed${s}.log 2>&1
  done
done
for s in $(seq 20 39); do
  for c in flat size_order random_curriculum rivers_only grader; do
    [ -f results_v2_lc91317/${c}_seed${s}.json ] && continue
    run nice -n 10 $PY -u curriculum_harness.py --condition $c --seed $s --sizes 9 13 17 --total-steps 1500000 \
        --results-dir results_v2_lc91317 > logs_v2/lc91317_${c}_seed${s}.log 2>&1
  done
done
wait
echo "CPU QUEUE DONE $(date)"
