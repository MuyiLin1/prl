#!/bin/bash
# v2 runs (see PREREGISTRATION.md). 12 concurrent single-thread jobs on one machine.
cd "$(dirname "$0")/.."
PY=/Users/linmuyi/miniconda3/envs/llm_gs/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CAP=${CAP:-8}
run() { while [ "$(jobs -rp | wc -l)" -ge "$CAP" ]; do sleep 5; done; "$@" & sleep 1; }
for s in $(seq 0 19); do
  for c in flat size_order random_curriculum rivers_only grader; do
    [ -f results_v2_lc91317/${c}_seed${s}.json ] && continue
    run nice -n 10 $PY -u curriculum_harness.py --condition $c --seed $s --sizes 9 13 17 --total-steps 1500000 \
        --results-dir results_v2_lc91317 > logs_v2/lc91317_${c}_seed${s}.log 2>&1
  done
  if [ $s -lt 10 ]; then
    for c in flat size_order; do
      [ -f results_v2_lc7911/${c}_seed${s}.json ] && continue
      run nice -n 10 $PY -u curriculum_harness.py --condition $c --seed $s --sizes 7 9 11 --total-steps 600000 \
          --results-dir results_v2_lc7911 > logs_v2/lc7911_${c}_seed${s}.log 2>&1
    done
  fi
done
wait
echo "ALL V2 JOBS DONE $(date)"
