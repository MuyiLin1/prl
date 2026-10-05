#!/usr/bin/env bash
# Section 6.6 LavaGap curriculum prefilter — k-curve sweep with the FAIR ablation.
#
# Runs run_lavagap_curriculum.sh once per k in KS, each with a curated ENSEMBLE
# (CURATEDDRAWS subsets) so curated and random have matched n and the
# curated-vs-random comparison is statistically fair at every k. Produces the
# full curve in one shot: data/lavagap_curriculum_fair_k<k>_{results,summary}.csv
#
# Usage (on bruinml, inside ~/temp-prl):
#   source ~/anaconda3/etc/profile.d/conda.sh && conda activate llm_gs
#   bash scripts/curriculum/cluster/sweep_lavagap_curriculum.sh
#
# Env overrides: KS POOL HELDOUT NITER KNB TRAINSEEDS RANDOMDRAWS CURATEDDRAWS \
#                POOLSUBSAMPLE PROBESEEDS NUMSHARDS OUTBASE
set -euo pipefail

KS=${KS:-"10 15 20 30 50"}
export POOL=${POOL:-150}
export HELDOUT=${HELDOUT:-80}
export NITER=${NITER:-250}
export KNB=${KNB:-128}
export TRAINSEEDS=${TRAINSEEDS:-6}
export RANDOMDRAWS=${RANDOMDRAWS:-6}
export CURATEDDRAWS=${CURATEDDRAWS:-6}
export POOLSUBSAMPLE=${POOLSUBSAMPLE:-0.85}
export PROBESEEDS=${PROBESEEDS:-3}
export NUMSHARDS=${NUMSHARDS:-20}
OUTBASE=${OUTBASE:-data/lavagap_curriculum_fair}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DRIVER="$HERE/run_lavagap_curriculum.sh"

echo "== LavaGap curriculum FAIR k-curve sweep =="
echo "ks=[$KS] pool=$POOL heldout=$HELDOUT curated_draws=$CURATEDDRAWS random_draws=$RANDOMDRAWS"
echo

for k in $KS; do
  echo "######## k=$k ########"
  K="$k" OUT="${OUTBASE}_k${k}" bash "$DRIVER"
  echo
done

echo "== sweep complete; summaries: ${OUTBASE}_k*_summary.csv =="
