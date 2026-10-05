#!/usr/bin/env bash
# Path 1: Sokoban dirty-pool curation ablation — sharded cluster run (GNU parallel).
#
# Mirrors run_dirty_pool_curriculum.sh. Each shard deterministically rebuilds the
# scored bank + contaminated pool + 4-way subsets, runs its assigned training jobs,
# writes <OUT>_shard<i>of<N>.csv. After all shards finish, --merge aggregates.
#
# Usage (on bruinml, inside ~/temp-prl):
#   source ~/anaconda3/etc/profile.d/conda.sh && conda activate llm_gs
#   bash scripts/curriculum/cluster/run_sokoban_dirty_pool.sh
#
# Env overrides: NUNF NMED NHARD POOL K HELDOUT PROBEBUDGET PROBEK PROBESEEDS
#   NITER KNB TRAINSEEDS FULLTRAINSEEDS DRAWS BANDLO BANDHI FRACTRIVIAL FRACHARD
#   FRACDUP DUPCLUSTERS TRIVIALQ HARDQ MAXCALLS NUMSHARDS OUT
set -euo pipefail

NUNF=${NUNF:-700}
NMED=${NMED:-500}
NHARD=${NHARD:-200}
POOL=${POOL:-1000}
K=${K:-40}
HELDOUT=${HELDOUT:-120}
PROBEBUDGET=${PROBEBUDGET:-150}
PROBEK=${PROBEK:-32}
PROBESEEDS=${PROBESEEDS:-2}
NITER=${NITER:-250}
KNB=${KNB:-128}
TRAINSEEDS=${TRAINSEEDS:-6}
FULLTRAINSEEDS=${FULLTRAINSEEDS:-2}   # full-pool baseline is expensive; fewer seeds
DRAWS=${DRAWS:-6}
BANDLO=${BANDLO:-0.15}
BANDHI=${BANDHI:-0.85}
FRACTRIVIAL=${FRACTRIVIAL:-0.55}
FRACHARD=${FRACHARD:-0.15}
FRACDUP=${FRACDUP:-0.15}
DUPCLUSTERS=${DUPCLUSTERS:-3}
TRIVIALQ=${TRIVIALQ:-0.33}
HARDQ=${HARDQ:-0.67}
MAXCALLS=${MAXCALLS:-2000}
NUMSHARDS=${NUMSHARDS:-20}
OUT=${OUT:-data/sokoban_dirty_pool}

PY=$(command -v python)
SCRIPT=scripts/curriculum/sokoban_dirty_pool.py

echo "== Sokoban dirty-pool curation ablation =="
echo "bank=($NUNF,$NMED,$NHARD) pool=$POOL k=$K heldout=$HELDOUT probe=($PROBEBUDGET,$PROBEK,$PROBESEEDS)"
echo "n_iter=$NITER k_nb=$KNB train_seeds=$TRAINSEEDS full_seeds=$FULLTRAINSEEDS draws=$DRAWS shards=$NUMSHARDS out=$OUT"
echo "contamination: trivial=$FRACTRIVIAL hard=$FRACHARD dup=$FRACDUP clusters=$DUPCLUSTERS q=[$TRIVIALQ,$HARDQ]"
mkdir -p "$(dirname "$OUT")" logs

run_shard() {
  local s=$1
  "$PY" "$SCRIPT" \
    --n-unfiltered "$NUNF" --n-medium "$NMED" --n-hard "$NHARD" \
    --pool-size "$POOL" --k "$K" --heldout "$HELDOUT" \
    --probe-budget "$PROBEBUDGET" --probe-k "$PROBEK" --probe-seeds "$PROBESEEDS" \
    --n-iter "$NITER" --k-neighbors "$KNB" --max-calls "$MAXCALLS" \
    --train-seeds "$TRAINSEEDS" --full-train-seeds "$FULLTRAINSEEDS" --draws "$DRAWS" \
    --band-lo "$BANDLO" --band-hi "$BANDHI" \
    --frac-trivial "$FRACTRIVIAL" --frac-hard "$FRACHARD" --frac-dup "$FRACDUP" \
    --dup-clusters "$DUPCLUSTERS" --trivial-q "$TRIVIALQ" --hard-q "$HARDQ" \
    --shard "$s" --num-shards "$NUMSHARDS" --out "$OUT" \
    > "logs/sokoban_dirty_shard${s}.log" 2>&1
  echo "shard $s done"
}
export -f run_shard
export PY SCRIPT NUNF NMED NHARD POOL K HELDOUT PROBEBUDGET PROBEK PROBESEEDS
export NITER KNB TRAINSEEDS FULLTRAINSEEDS DRAWS BANDLO BANDHI FRACTRIVIAL FRACHARD
export FRACDUP DUPCLUSTERS TRIVIALQ HARDQ MAXCALLS NUMSHARDS OUT

if command -v parallel >/dev/null 2>&1; then
  seq 0 $((NUMSHARDS - 1)) | parallel -j "$NUMSHARDS" run_shard {}
else
  echo "GNU parallel not found; falling back to xargs"
  seq 0 $((NUMSHARDS - 1)) | xargs -P "$NUMSHARDS" -I{} bash -c 'run_shard "$@"' _ {}
fi

echo "== merging =="
"$PY" "$SCRIPT" --merge --out "$OUT"
echo "== done; results at ${OUT}_results.csv / _summary.csv =="
