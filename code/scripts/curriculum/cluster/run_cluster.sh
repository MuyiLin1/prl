#!/bin/bash
# Cluster launcher for the {9,13,17} 7-method x multi-seed curriculum run.
#
# Single shared box, NO scheduler: validate/build the shared oracle cache ONCE (serial),
# then run every (condition x seed) job in parallel under a concurrency cap. Each job is
# CPU-bound (tiny CNN, env-stepping dominated), so we pin 1 thread/job and use cores for
# job-level parallelism. Launch the whole thing under nohup so it survives an ssh drop.
#
# Usage:   SEEDS="0 1 2 3 4"  CAP=12  bash run_cluster.sh
#   (recommended launch:  nohup bash run_cluster.sh > logs/launcher.log 2>&1 & )
#
# Logs:    logs/oracle_build.log, logs/<condition>_seed<N>.log, logs/launcher.log
# Results: results/<condition>_seed<N>.json   (aggregate with aggregate_results.py)
set -u
cd "$(dirname "$0")"
source ~/anaconda3/etc/profile.d/conda.sh && conda activate curric

SIZES="9 13 17"
TOTAL_STEPS="${TOTAL_STEPS:-1500000}"
SEEDS="${SEEDS:-0 1 2 3 4}"
CAP="${CAP:-12}"
CONDITIONS="random oracle geometry probe combined adaptive_fine adaptive_coarse"

# Match the seed-0 run's oracle config (cache file oracle_<hash>_r1e12.json) so the build
# step and every oracle-using job hit the EXISTING cache instantly, instead of rebuilding
# the r3e20 default (3x600k steps = hours serial on this contended box). The oracle's
# failure is structural (ref PPOs cannot solve hard layouts -> degenerate labels), so the
# ref-seed count does not change the finding; this keeps the multiseed faithful to seed 0.
ORACLE_FLAGS="--oracle-ref-seeds 1 --oracle-eval-episodes 12"

# One thread per job: the work is env-stepping (serial Python), not matmul, so we spend
# cores on running many jobs at once rather than on intra-job threading.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

mkdir -p logs results

echo "=== [1/2] validate/build oracle cache once (serial) ==="
python -u curriculum_harness.py --condition oracle --build-oracle-only \
       --sizes $SIZES --total-steps $TOTAL_STEPS $ORACLE_FLAGS > logs/oracle_build.log 2>&1
echo "oracle build exit=$? -- see logs/oracle_build.log"
grep -iE "cache|building|loaded|done" logs/oracle_build.log | head -3

echo "=== [2/2] launching condition x seed jobs (SEEDS=[$SEEDS] CAP=$CAP) ==="
for s in $SEEDS; do
  for c in $CONDITIONS; do
    while [ "$(jobs -rp | wc -l)" -ge "$CAP" ]; do sleep 5; done
    echo "  launch $c seed $s -> logs/${c}_seed${s}.log"
    python -u curriculum_harness.py --condition "$c" --seed "$s" \
           --sizes $SIZES --total-steps $TOTAL_STEPS $ORACLE_FLAGS > "logs/${c}_seed${s}.log" 2>&1 &
    sleep 1
  done
done
wait
echo "=== ALL JOBS DONE ($(date)) ==="
