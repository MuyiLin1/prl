#!/bin/bash
# Addendum C: harder structured mazes. Pilot -> pre-registered rule picks p_max -> main run, unattended.
cd "$(dirname "$0")/.."
PY=/Users/linmuyi/miniconda3/envs/llm_gs/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CAP=${CAP:-12}
busy() { pgrep -f "python -u (maze|curriculum|sfl)_harness.py" | wc -l; }
wait_slot() { while [ "$(busy)" -ge "$CAP" ]; do sleep 15; done; }
launch() {  # out p cond seed
  [ -f $1/$3_seed$4.json ] && return
  wait_slot
  nice -n 10 $PY -u maze_harness.py --perfect --p-max $2 --calib-p 0.5 --condition $3 --seed $4 --sizes 19 \
      --n-seeds 90 --eval-n-seeds 36 --total-steps 1500000 --eval-every 50000 \
      --results-dir $1 > logs_maze/c_$(basename $1)_$3_seed$4.log 2>&1 &
  sleep 2
}
mkdir -p logs_maze
echo "pilot start $(date)"
for p in 0.25 0.1; do
  for s in 0 1 2 3; do launch results_maze_c_pilot/p$p $p flat $s; done
  for s in 0 1; do launch results_maze_c_pilot/p$p $p wall_count $s; done
done
wait
CHOSEN=$($PY - <<'EOF'
import json, glob
for p in ["0.25", "0.1"]:
    f = [json.load(open(x))["final_held_success"] for x in glob.glob(f"results_maze_c_pilot/p{p}/flat_seed*.json")]
    w = [json.load(open(x))["final_held_success"] for x in glob.glob(f"results_maze_c_pilot/p{p}/wall_count_seed*.json")]
    ok = len(f) == 4 and len(w) == 2 and sum(f) / 4 <= 0.15 and sum(w) / 2 >= 0.05
    print(f"p_max {p}: flat {[round(x, 2) for x in f]} wall_count {[round(x, 2) for x in w]} -> {'QUALIFIES' if ok else 'no'}", file=__import__('sys').stderr)
    if ok:
        print(p); break
EOF
)
echo "pilot done $(date); chosen p_max = '${CHOSEN}'"
if [ -z "$CHOSEN" ]; then echo "NO CURRICULUM REGIME FOUND - main run not launched (Addendum C rule)"; exit 0; fi
OUT=results_maze_s19_p$CHOSEN
for s in $(seq 0 19); do
  for c in grader random_curriculum wall_count path_len flat; do launch $OUT $CHOSEN $c $s; done
done
wait
echo "MAZE C MAIN DONE $(date)"
