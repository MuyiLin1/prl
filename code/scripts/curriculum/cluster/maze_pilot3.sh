#!/bin/bash
# Maze pilot 3 (Amendment B2): structured mazes. Used ONLY to choose the size.
cd "$(dirname "$0")/.."
PY=/Users/linmuyi/miniconda3/envs/llm_gs/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CAP=${CAP:-8}
busy() { pgrep -f "maze_harness.py|curriculum_harness.py|sfl_harness.py" | wc -l; }
run() { while [ "$(busy)" -ge "$CAP" ]; do sleep 10; done; "$@" & sleep 2; }
mkdir -p logs_maze
for s in 0 1; do
  for size in 15 19 11; do
    for c in flat wall_count; do
      out=results_maze_pilot3/s${size}
      [ -f $out/${c}_seed${s}.json ] && continue
      run nice -n 10 $PY -u maze_harness.py --perfect --p-max 0.5 --condition $c --seed $s --sizes $size \
          --n-seeds 90 --eval-n-seeds 36 --total-steps 1500000 --eval-every 50000 \
          --results-dir $out > logs_maze/pilot3_s${size}_${c}_seed${s}.log 2>&1
    done
  done
done
wait
echo "MAZE PILOT 3 DONE $(date)"
