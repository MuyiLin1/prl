#!/usr/bin/env bash
# Part B: MiniGrid PutNear + RedBlueDoor difficulty (shaped reward) — cluster run.
#
# Raw MiniGrid reward is sparse/staged => flat HC landscape => degenerate
# difficulty. minigrid_shaped_extra.py adds distance/subgoal shaping so HC has a
# gradient. One task per parallel job at full N=GRIDS x R=REPS, writing:
#   <OUT>/minigrid_<task>_summary.csv  (one-row per-task summary)
#   /tmp/minigrid_<task>.csv           (per-grid difficulty + features; script default)
#   logs/minigrid_<task>.log
# After all jobs finish the summaries are concatenated into
#   <OUT>/minigrid_difficulty_summary.csv
#
# Usage (on bruinml, inside ~/temp-prl):
#   source ~/anaconda3/etc/profile.d/conda.sh && conda activate llm_gs
#   bash scripts/cluster/run_minigrid_difficulty.sh
#
# Env overrides: GRIDS REPS SIZE NITER TASKS OUT JOBS
set -euo pipefail

GRIDS=${GRIDS:-200}
REPS=${REPS:-12}
SIZE=${SIZE:-6}
NITER=${NITER:-2000}
TASKS=${TASKS:-"PutNear RedBlueDoor"}
OUT=${OUT:-data/minigrid_difficulty}
JOBS=${JOBS:-2}

PY=$(command -v python)
SCRIPT=scripts/minigrid_shaped_extra.py

mkdir -p "$OUT" logs

echo "== MiniGrid shaped difficulty (PutNear + RedBlueDoor) =="
echo "grids=$GRIDS reps=$REPS size=$SIZE n_iter=$NITER jobs=$JOBS out=$OUT"
echo "tasks: $TASKS"

run_task() {
  local t=$1
  "$PY" "$SCRIPT" \
    --tasks "$t" \
    --grids "$GRIDS" --reps "$REPS" --size "$SIZE" --n-iter "$NITER" \
    --output "$OUT/minigrid_${t}_summary.csv" \
    > "logs/minigrid_${t}.log" 2>&1
  echo "task $t done"
}
export -f run_task
export PY SCRIPT GRIDS REPS SIZE NITER OUT

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
