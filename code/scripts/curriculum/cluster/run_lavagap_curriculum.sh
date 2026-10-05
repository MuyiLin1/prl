#!/usr/bin/env bash
# Section 6.6 LavaGap curriculum prefilter — sharded cluster run (GNU parallel).
#
# Mirrors the Sokoban HC cluster pattern. Each shard recomputes the (deterministic)
# pool scoring + curated/random subsets, then runs only its assigned training jobs,
# writing <OUT>_shard<i>of<N>.csv. After all shards finish, --merge aggregates.
#
# Usage (on bruinml, inside ~/temp-prl):
#   source ~/anaconda3/etc/profile.d/conda.sh && conda activate llm_gs
#   bash scripts/curriculum/cluster/run_lavagap_curriculum.sh
#
# Env overrides: POOL K HELDOUT SIZE NITER KNB TRAINSEEDS RANDOMDRAWS CURATEDDRAWS POOLSUBSAMPLE PROBESEEDS NUMSHARDS OUT
set -euo pipefail

POOL=${POOL:-150}
K=${K:-50}
HELDOUT=${HELDOUT:-80}
SIZE=${SIZE:-9}
NITER=${NITER:-250}
KNB=${KNB:-128}
TRAINSEEDS=${TRAINSEEDS:-6}
RANDOMDRAWS=${RANDOMDRAWS:-5}
CURATEDDRAWS=${CURATEDDRAWS:-5}
POOLSUBSAMPLE=${POOLSUBSAMPLE:-0.85}
PROBESEEDS=${PROBESEEDS:-3}
NUMSHARDS=${NUMSHARDS:-20}
OUT=${OUT:-data/lavagap_curriculum}

PY=$(command -v python)
SCRIPT=scripts/curriculum/lavagap_curriculum.py

echo "== LavaGap curriculum sweep =="
echo "pool=$POOL k=$K heldout=$HELDOUT size=$SIZE n_iter=$NITER k_nb=$KNB"
echo "train_seeds=$TRAINSEEDS random_draws=$RANDOMDRAWS curated_draws=$CURATEDDRAWS shards=$NUMSHARDS out=$OUT"
mkdir -p "$(dirname "$OUT")" logs

run_shard() {
  local s=$1
  "$PY" "$SCRIPT" \
    --pool-size "$POOL" --k "$K" --heldout "$HELDOUT" --size "$SIZE" \
    --n-iter "$NITER" --k-neighbors "$KNB" \
    --train-seeds "$TRAINSEEDS" --random-draws "$RANDOMDRAWS" \
    --curated-draws "$CURATEDDRAWS" --pool-subsample "$POOLSUBSAMPLE" \
    --probe-seeds "$PROBESEEDS" \
    --shard "$s" --num-shards "$NUMSHARDS" --out "$OUT" \
    > "logs/curriculum_shard${s}.log" 2>&1
  echo "shard $s done"
}
export -f run_shard
export PY SCRIPT POOL K HELDOUT SIZE NITER KNB TRAINSEEDS RANDOMDRAWS CURATEDDRAWS POOLSUBSAMPLE PROBESEEDS NUMSHARDS OUT

if command -v parallel >/dev/null 2>&1; then
  seq 0 $((NUMSHARDS - 1)) | parallel -j "$NUMSHARDS" run_shard {}
else
  echo "GNU parallel not found; falling back to xargs"
  seq 0 $((NUMSHARDS - 1)) | xargs -P "$NUMSHARDS" -I{} bash -c 'run_shard "$@"' _ {}
fi

echo "== merging =="
"$PY" "$SCRIPT" --merge --out "$OUT"
echo "== done; results at ${OUT}_results.csv / _summary.csv =="
