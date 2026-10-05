#!/usr/bin/env bash
# Section 6.6 follow-up: dirty-pool curriculum ablation — sharded cluster run.
#
# Mirrors run_lavagap_curriculum.sh. Each shard recomputes the (deterministic)
# seed-bank scoring + contaminated pool + 4-way subsets, runs its assigned training
# jobs, writes <OUT>_shard<i>of<N>.csv. After all shards finish, --merge aggregates.
#
# Usage (on bruinml, inside ~/temp-prl):
#   source ~/anaconda3/etc/profile.d/conda.sh && conda activate llm_gs
#   bash scripts/curriculum/cluster/run_dirty_pool_curriculum.sh
#
# Env overrides: GEN POOL K HELDOUT SIZE NITER KNB TRAINSEEDS DRAWS BANDLO BANDHI
#   FRACTRIVIAL FRACHARD FRACDUP DUPCLUSTERS TRIVIALQ HARDQ PROBESEEDS NUMSHARDS OUT
set -euo pipefail

GEN=${GEN:-2000}
POOL=${POOL:-1000}
K=${K:-40}
HELDOUT=${HELDOUT:-80}
SIZE=${SIZE:-13}
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
PROBESEEDS=${PROBESEEDS:-2}
NUMSHARDS=${NUMSHARDS:-20}
OUT=${OUT:-data/dirty_pool_curriculum}

PY=$(command -v python)
SCRIPT=scripts/curriculum/dirty_pool_curriculum.py

echo "== Dirty-pool curriculum ablation =="
echo "gen=$GEN pool=$POOL k=$K heldout=$HELDOUT size=$SIZE n_iter=$NITER k_nb=$KNB"
echo "train_seeds=$TRAINSEEDS draws=$DRAWS band=[$BANDLO,$BANDHI] shards=$NUMSHARDS out=$OUT"
echo "contamination: trivial=$FRACTRIVIAL hard=$FRACHARD dup=$FRACDUP clusters=$DUPCLUSTERS q=[$TRIVIALQ,$HARDQ]"
mkdir -p "$(dirname "$OUT")" logs

run_shard() {
  local s=$1
  "$PY" "$SCRIPT" \
    --gen-size "$GEN" --pool-size "$POOL" --k "$K" --heldout "$HELDOUT" --size "$SIZE" \
    --n-iter "$NITER" --k-neighbors "$KNB" \
    --train-seeds "$TRAINSEEDS" --full-train-seeds "$FULLTRAINSEEDS" --draws "$DRAWS" \
    --band-lo "$BANDLO" --band-hi "$BANDHI" \
    --frac-trivial "$FRACTRIVIAL" --frac-hard "$FRACHARD" --frac-dup "$FRACDUP" \
    --dup-clusters "$DUPCLUSTERS" --trivial-q "$TRIVIALQ" --hard-q "$HARDQ" \
    --probe-seeds "$PROBESEEDS" \
    --shard "$s" --num-shards "$NUMSHARDS" --out "$OUT" \
    > "logs/dirty_pool_shard${s}.log" 2>&1
  echo "shard $s done"
}
export -f run_shard
export PY SCRIPT GEN POOL K HELDOUT SIZE NITER KNB TRAINSEEDS FULLTRAINSEEDS DRAWS BANDLO BANDHI
export FRACTRIVIAL FRACHARD FRACDUP DUPCLUSTERS TRIVIALQ HARDQ PROBESEEDS NUMSHARDS OUT

if command -v parallel >/dev/null 2>&1; then
  seq 0 $((NUMSHARDS - 1)) | parallel -j "$NUMSHARDS" run_shard {}
else
  echo "GNU parallel not found; falling back to xargs"
  seq 0 $((NUMSHARDS - 1)) | xargs -P "$NUMSHARDS" -I{} bash -c 'run_shard "$@"' _ {}
fi

echo "== merging =="
"$PY" "$SCRIPT" --merge --out "$OUT"
echo "== done; results at ${OUT}_results.csv / _summary.csv =="
