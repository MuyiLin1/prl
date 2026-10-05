#!/usr/bin/env bash
# Part B: RedBlueDoor shaped difficulty — GRID-SHARDED cluster run.
#
# The default run_minigrid_difficulty.sh parallelizes ACROSS tasks (1 job per
# task), so RedBlueDoor ends up grinding all GRIDS grids serially on a single
# core while PutNear (which converges fast) finishes early and frees its core.
# RedBlueDoor's staged shaping is far more expensive, so the lone serial worker
# takes many hours. This script splits the GRIDS grids into SHARDS contiguous
# slices and runs them in parallel, then merges the raw per-grid shards into the
# usual one-row summary (reliability + geometry CV computed over the full set).
#
# Outputs:
#   <OUT>/raw/redbluedoor_shard_<i>.csv        per-shard raw (grid_seed, shaped_*, features)
#   <OUT>/minigrid_RedBlueDoor_summary.csv      merged one-row summary
#   logs/redbluedoor_shard_<i>.log
#
# Usage (on bruinml, inside ~/temp-prl):
#   source ~/anaconda3/etc/profile.d/conda.sh && conda activate llm_gs
#   bash scripts/cluster/run_redbluedoor_sharded.sh
#
# Env overrides: GRIDS REPS SIZE NITER OUT SHARDS
set -euo pipefail

GRIDS=${GRIDS:-200}
REPS=${REPS:-12}
SIZE=${SIZE:-6}
NITER=${NITER:-2000}
OUT=${OUT:-data/minigrid_difficulty}
SHARDS=${SHARDS:-16}

PY=$(command -v python)
SCRIPT=scripts/minigrid_shaped_extra.py
RAW="$OUT/raw"

mkdir -p "$RAW" logs

echo "== RedBlueDoor sharded difficulty =="
echo "grids=$GRIDS reps=$REPS size=$SIZE n_iter=$NITER shards=$SHARDS out=$OUT"

# Contiguous slice boundaries: shard i covers [start, end).
run_shard() {
  local i=$1
  local per=$(( (GRIDS + SHARDS - 1) / SHARDS ))
  local start=$(( i * per ))
  local end=$(( start + per ))
  if [ "$end" -gt "$GRIDS" ]; then end=$GRIDS; fi
  if [ "$start" -ge "$GRIDS" ]; then return 0; fi
  "$PY" "$SCRIPT" \
    --tasks RedBlueDoor \
    --grid-start "$start" --grid-end "$end" \
    --reps "$REPS" --size "$SIZE" --n-iter "$NITER" \
    --raw-output "$RAW/redbluedoor_shard_${i}.csv" \
    > "logs/redbluedoor_shard_${i}.log" 2>&1
  echo "shard $i [$start,$end) done"
}
export -f run_shard
export PY SCRIPT GRIDS REPS SIZE NITER RAW SHARDS

if command -v parallel >/dev/null 2>&1; then
  seq 0 $(( SHARDS - 1 )) | parallel -j "$SHARDS" run_shard {}
else
  echo "GNU parallel not found; falling back to xargs"
  seq 0 $(( SHARDS - 1 )) | xargs -P "$SHARDS" -I{} bash -c 'run_shard "$@"' _ {}
fi

echo "== merging RedBlueDoor shards =="
# shellcheck disable=SC2086
"$PY" "$SCRIPT" \
  --merge-raw "$RAW"/redbluedoor_shard_*.csv \
  --merge-task RedBlueDoor \
  --output "$OUT/minigrid_RedBlueDoor_summary.csv"

echo "== rebuilding combined MiniGrid summary (PutNear + RedBlueDoor) =="
"$PY" - "$OUT" <<'PYEOF'
import sys, glob, os
import pandas as pd
out = sys.argv[1]
parts = sorted(glob.glob(os.path.join(out, "minigrid_*_summary.csv")))
parts = [p for p in parts if not p.endswith("difficulty_summary.csv")]
if not parts:
    print("no summary files found"); sys.exit(1)
df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
dest = os.path.join(out, "minigrid_difficulty_summary.csv")
df.to_csv(dest, index=False)
print(df.to_string(index=False))
print(f"\nwrote {dest}")
PYEOF
