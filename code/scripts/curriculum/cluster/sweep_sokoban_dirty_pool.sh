#!/usr/bin/env bash
# Sweep the Sokoban dirty-pool curation ablation across k.
# Calls run_sokoban_dirty_pool.sh once per k for a k-curve of
# random vs band vs coverage vs both on the CLEAN held-out test.
#
# Usage (on bruinml, inside ~/temp-prl):
#   source ~/anaconda3/etc/profile.d/conda.sh && conda activate llm_gs
#   bash scripts/curriculum/cluster/sweep_sokoban_dirty_pool.sh
#
# Env overrides: KS NUNF NMED NHARD POOL HELDOUT PROBEBUDGET NITER KNB
#   TRAINSEEDS FULLTRAINSEEDS DRAWS NUMSHARDS OUTBASE
set -euo pipefail

KS=${KS:-"15 30 50 80"}
NUNF=${NUNF:-700}
NMED=${NMED:-500}
NHARD=${NHARD:-200}
POOL=${POOL:-1000}
HELDOUT=${HELDOUT:-120}
PROBEBUDGET=${PROBEBUDGET:-150}
NITER=${NITER:-250}
KNB=${KNB:-128}
TRAINSEEDS=${TRAINSEEDS:-6}
FULLTRAINSEEDS=${FULLTRAINSEEDS:-2}
DRAWS=${DRAWS:-6}
NUMSHARDS=${NUMSHARDS:-20}
OUTBASE=${OUTBASE:-data/sokoban_dirty}

DRIVER=scripts/curriculum/cluster/run_sokoban_dirty_pool.sh

echo "== Sokoban dirty-pool k-sweep: ks=[$KS] pool=$POOL bank=($NUNF,$NMED,$NHARD) =="
for k in $KS; do
  echo ""
  echo "###### k=$k ######"
  K="$k" NUNF="$NUNF" NMED="$NMED" NHARD="$NHARD" POOL="$POOL" HELDOUT="$HELDOUT" \
    PROBEBUDGET="$PROBEBUDGET" NITER="$NITER" KNB="$KNB" TRAINSEEDS="$TRAINSEEDS" \
    FULLTRAINSEEDS="$FULLTRAINSEEDS" DRAWS="$DRAWS" NUMSHARDS="$NUMSHARDS" \
    OUT="${OUTBASE}_k${k}" \
    bash "$DRIVER"
done
echo ""
echo "== sweep done; per-k results at ${OUTBASE}_k<K>_results.csv =="
