#!/usr/bin/env bash
# Part B: comprehensive Karel difficulty measurement — sharded cluster run.
#
# One task per parallel job (the natural shard unit). Each job runs the
# R-seed-averaged HC difficulty pipeline (averaged_label_pilot.py) at full
# N=GRIDS x R=REPS, writing:
#   <OUT>/karel_<task>_grids.csv     (per-grid mean difficulty + features)
#   <OUT>/karel_<task>_summary.csv   (one-row per-task summary: reliability,
#                                     best_single_rho, rf_cv_rho, SNR, ...)
#   logs/karel_<task>.log            (full stdout incl. per-grid trace)
# After all jobs finish, the summary CSVs are concatenated into
#   <OUT>/karel_difficulty_summary.csv
#
# Usage (on bruinml, inside ~/temp-prl):
#   source ~/anaconda3/etc/profile.d/conda.sh && conda activate llm_gs
#   bash scripts/cluster/run_karel_difficulty.sh
#
# Env overrides: GRIDS REPS MAXPROG TASKS OUT JOBS
set -euo pipefail

GRIDS=${GRIDS:-200}
REPS=${REPS:-12}
MAXPROG=${MAXPROG:-5000}
TASKS=${TASKS:-"StairClimber Maze FourCorners CleanHouse DoorKey OneStroke WallAvoider PathFollow"}
OUT=${OUT:-data/karel_difficulty}
JOBS=${JOBS:-8}

PY=$(command -v python)
SCRIPT=scripts/averaged_label_pilot.py

mkdir -p "$OUT" logs

echo "== Karel comprehensive difficulty =="
echo "grids=$GRIDS reps=$REPS max_programs=$MAXPROG jobs=$JOBS out=$OUT"
echo "tasks: $TASKS"

run_task() {
  local t=$1
  "$PY" "$SCRIPT" \
    --tasks "$t" \
    --grids "$GRIDS" --reps "$REPS" --max-programs "$MAXPROG" \
    --output "$OUT/karel_${t}_grids.csv" \
    --summary-output "$OUT/karel_${t}_summary.csv" \
    > "logs/karel_${t}.log" 2>&1
  echo "task $t done"
}
export -f run_task
export PY SCRIPT GRIDS REPS MAXPROG OUT

if command -v parallel >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  printf '%s\n' $TASKS | parallel -j "$JOBS" run_task {}
else
  echo "GNU parallel not found; falling back to xargs"
  # shellcheck disable=SC2086
  printf '%s\n' $TASKS | xargs -P "$JOBS" -I{} bash -c 'run_task "$@"' _ {}
fi

echo "== merging summaries =="
"$PY" - "$OUT" <<'PYEOF'
import sys, glob, os
import pandas as pd
out = sys.argv[1]
parts = sorted(glob.glob(os.path.join(out, "karel_*_summary.csv")))
if not parts:
    print("no summary files found"); sys.exit(1)
df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
dest = os.path.join(out, "karel_difficulty_summary.csv")
df.to_csv(dest, index=False)
print(df.to_string(index=False))
print(f"\nwrote {dest}")
PYEOF
