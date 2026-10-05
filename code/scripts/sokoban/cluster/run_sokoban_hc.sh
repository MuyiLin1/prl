#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Sokoban DSL + Hill Climbing difficulty labelling — cluster run script.
#
# Run this from the REPO ROOT on the cluster after installing the bundle
# (see README.md). It labels every level listed in the levels CSV using the
# real Sokoban DSL + bounded random-restart Hill Climbing, identical in form to
# the Karel / MiniGrid reference solver.
#
# The job is embarrassingly parallel across levels: use --shard / --num-shards
# to split it across array-job tasks or `parallel` workers, then concatenate
# the per-shard CSVs at the end.
# ---------------------------------------------------------------------------
set -euo pipefail

# ---- configuration (override via environment) -----------------------------
CONDA_ENV="${CONDA_ENV:-llm_gs}"
LABELS_CSV="${LABELS_CSV:-data/sokoban_fastpath_labels.csv}"   # level_id,tier
NUM_SEEDS="${NUM_SEEDS:-12}"        # R
BUDGET="${BUDGET:-4000}"            # program-eval budget per seed
K="${K:-250}"                       # HC neighbours per step
SIGMA="${SIGMA:-0.25}"
MAX_CALLS="${MAX_CALLS:-2000}"      # max env steps per program rollout
W_ON_TARGET="${W_ON_TARGET:-0.8}"   # shaping weight: boxes actually on target
W_PROGRESS="${W_PROGRESS:-0.2}"     # shaping weight: matching-distance progress
OUT_PREFIX="${OUT_PREFIX:-data/sokoban_hc_labels}"
EXTRA_ARGS="${EXTRA_ARGS:-}"        # e.g. "--shard 0 --num-shards 8"

# ---- activate environment -------------------------------------------------
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

echo "=== Sokoban DSL+HC labelling ==="
echo "labels=${LABELS_CSV} R=${NUM_SEEDS} budget=${BUDGET} k=${K} sigma=${SIGMA} max_calls=${MAX_CALLS}"
echo "shaping: w_on_target=${W_ON_TARGET} w_progress=${W_PROGRESS}"
echo "out=${OUT_PREFIX} extra=${EXTRA_ARGS}"

python scripts/sokoban/hc_label.py \
    --labels "${LABELS_CSV}" \
    --num-seeds "${NUM_SEEDS}" \
    --budget "${BUDGET}" \
    --k "${K}" \
    --sigma "${SIGMA}" \
    --max-calls "${MAX_CALLS}" \
    --w-on-target "${W_ON_TARGET}" \
    --w-progress "${W_PROGRESS}" \
    --out "${OUT_PREFIX}" \
    --no-progress \
    ${EXTRA_ARGS}

echo "=== done -> ${OUT_PREFIX}_labels.csv ==="
