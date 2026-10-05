#!/bin/bash
# Main maze run (Addendum B + B2/B3): 19x19 structured mazes, 5 conditions x seeds 0-19.
# Non-grader conditions start immediately; grader runs are held back until the 3 calibration agents finish and
# their labels are merged, then launched ahead of the remaining queue. At most CAP python jobs at once.
cd "$(dirname "$0")/.."
PY=/Users/linmuyi/miniconda3/envs/llm_gs/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CAP=${CAP:-12}
M="--perfect --p-max 0.5"
OUT=results_maze_s19
MERGED=cache/maze_calib_s19_perfect_p0.5.json
mkdir -p logs_maze $OUT
busy() { pgrep -f "python -u (maze|curriculum|sfl)_harness.py" | wc -l; }
wait_slot() { while [ "$(busy)" -ge "$CAP" ]; do sleep 15; done; }
launch() {
  [ -f $OUT/$1_seed$2.json ] && return
  wait_slot
  nice -n 10 $PY -u maze_harness.py $M --condition $1 --seed $2 --sizes 19 \
      --n-seeds 90 --eval-n-seeds 36 --total-steps 1500000 --eval-every 50000 \
      --results-dir $OUT > logs_maze/main_s19_$1_seed$2.log 2>&1 &
  sleep 2
}
calib_ready() {
  if [ ! -f $MERGED ] && [ $(ls cache/maze_calib_s19_perfect_p0.5_ref{0,1,2}.json 2>/dev/null | wc -l) -eq 3 ]; then
    $PY -u maze_harness.py $M --merge-calib --sizes 19 2>&1 | grep -E "calib|Error|Trace"
  fi
  [ -f $MERGED ]
}
deferred=()
flush_grader() { if [ ${#deferred[@]} -gt 0 ] && calib_ready; then for s in "${deferred[@]}"; do launch grader $s; done; deferred=(); fi; }
for s in $(seq 0 19); do
  for c in grader random_curriculum wall_count path_len flat; do
    if [ $c = grader ] && ! calib_ready; then deferred+=($s); continue; fi
    flush_grader
    launch $c $s
  done
done
while [ ${#deferred[@]} -gt 0 ]; do flush_grader; sleep 30; done
wait
echo "MAZE MAIN DONE $(date)"
