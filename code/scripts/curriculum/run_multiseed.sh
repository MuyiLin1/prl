#!/bin/bash
# Orchestrate the multi-seed curriculum experiment.
#
#   1. Build the shared (denoised) oracle cache ONCE, serially, so parallel seeds don't race.
#   2. Run all 7 conditions for each seed, with a concurrency cap (env-steps headline metric is
#      hardware-independent, so parallel seeds don't corrupt it).
#
# Usage:  SEEDS="0 1 2 ... 9"  CAP=4  bash run_multiseed.sh
# Logs:   /tmp/oracle_build.log, /tmp/run_seed<N>.log
set -u
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate llm_gs

SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
CAP="${CAP:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

echo "=== [1/2] building shared oracle cache (once, serial) ==="
python -u curriculum_harness.py --condition oracle --build-oracle-only > /tmp/oracle_build.log 2>&1
echo "oracle build exit=$? (see /tmp/oracle_build.log)"
grep -E "cached|done" /tmp/oracle_build.log | tail -2

echo "=== [2/2] running seeds [$SEEDS] with concurrency cap $CAP ==="
for s in $SEEDS; do
  # throttle: wait until a slot frees up
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$CAP" ]; do sleep 10; done
  echo "  launching seed $s -> /tmp/run_seed${s}.log"
  python -u curriculum_harness.py --condition all --seed "$s" > "/tmp/run_seed${s}.log" 2>&1 &
  sleep 3   # small stagger so launches don't collide on pool enumeration
done

wait
echo "=== ALL SEEDS DONE ==="
