#!/usr/bin/env bash
# Sweep the dirty-pool curriculum ablation across k (and optionally grid size).
# Calls run_dirty_pool_curriculum.sh once per k. Reports a k-curve of
# random vs band vs coverage vs both on the CLEAN held-out test.
#
# Usage (on bruinml, inside ~/temp-prl):
#   source ~/anaconda3/etc/profile.d/conda.sh && conda activate llm_gs
#   bash scripts/curriculum/cluster/sweep_dirty_pool_curriculum.sh
#
# Env overrides: KS SIZE GEN POOL HELDOUT NITER KNB TRAINSEEDS DRAWS NUMSHARDS OUTBASE
set -euo pipefail

KS=${KS:-"10 20 30 50 80"}
SIZE=${SIZE:-13}
GEN=${GEN:-2000}
POOL=${POOL:-1000}
HELDOUT=${HELDOUT:-80}
NITER=${NITER:-250}
KNB=${KNB:-128}
TRAINSEEDS=${TRAINSEEDS:-6}
DRAWS=${DRAWS:-6}
NUMSHARDS=${NUMSHARDS:-20}
OUTBASE=${OUTBASE:-data/dirty_pool}

DRIVER=scripts/curriculum/cluster/run_dirty_pool_curriculum.sh

echo "== Dirty-pool k-sweep: ks=[$KS] size=$SIZE gen=$GEN pool=$POOL =="
for k in $KS; do
  echo ""
  echo "###### k=$k ######"
  K="$k" SIZE="$SIZE" GEN="$GEN" POOL="$POOL" HELDOUT="$HELDOUT" \
    NITER="$NITER" KNB="$KNB" TRAINSEEDS="$TRAINSEEDS" DRAWS="$DRAWS" \
    NUMSHARDS="$NUMSHARDS" OUT="${OUTBASE}_k${k}" \
    bash "$DRIVER"
done
echo ""
echo "== sweep done; per-k results at ${OUTBASE}_k<K>_results.csv =="
