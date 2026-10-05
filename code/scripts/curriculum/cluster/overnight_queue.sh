#!/bin/bash
# Overnight follow-on (2026-10-03). Starts only after cpu_queue.sh (stage 3) has finished.
#  (1) SFL seeds 10-19, all 3 conditions: POST-HOC EXPLORATORY (added after seeing seeds 0-9).
#  (2) Karel per-search reruns (same fixed seeds) for WallAvoider and DoorKey -> "k searches" curve.
cd "$(dirname "$0")/.."
ROOT=$(cd ../.. && pwd)
PY=/Users/linmuyi/miniconda3/envs/llm_gs/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CAP=${CAP:-7}
busy() { pgrep -f "curriculum_harness.py|sfl_harness.py" | wc -l; }
run() { while [ "$(busy)" -ge "$CAP" ]; do sleep 10; done; "$@" & sleep 2; }

while pgrep -f cpu_queue.sh >/dev/null; do sleep 60; done
echo "stage 3 finished; overnight queue starting $(date)"

mkdir -p "$ROOT/data/karel_difficulty_rerun"
for t in WallAvoider DoorKey; do
  (cd "$ROOT" && nice -n 15 $PY -u scripts/averaged_label_pilot.py --tasks $t --grids 200 --reps 12 \
      --max-programs 5000 --output data/karel_difficulty_rerun/karel_${t}_grids.csv \
      > logs_queue/karel_rerun_${t}.log 2>&1) &
done

for s in $(seq 10 19); do
  for c in sfl_grader sfl sfl_small; do
    [ -f results_v2_sfl_lc91317/${c}_seed${s}.json ] && continue
    run nice -n 10 $PY -u sfl_harness.py --condition $c --seed $s > logs_v2/sfl_${c}_seed${s}.log 2>&1
  done
done
wait
echo "OVERNIGHT QUEUE DONE $(date)"
