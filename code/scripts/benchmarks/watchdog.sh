#!/bin/bash
# Load guard for the experiment queues. Every 20 s:
#  * logs free memory %, load and our total RSS to logs_queue/watchdog.log
#  * PAUSES (SIGSTOP) the newest running experiment process if free memory < 30%
#  * RESUMES paused processes when free memory > 35%
#  * KILLS any single experiment process above 4 GB RSS, and everything ours if free memory < 15%
cd "$(dirname "$0")/../.."
PAT="curriculum_harness.py|sfl_harness.py|llm_judge_mlx.py|maze_harness.py|feature_ablation.py"
mkdir -p logs_queue
while true; do
  free=$(memory_pressure | awk '/free percentage/ {gsub("%","",$5); print $5}')
  swap=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}')
  pids=$(pgrep -f "$PAT")
  rss=0; for p in $pids; do r=$(ps -o rss= -p $p 2>/dev/null || echo 0); rss=$((rss + r));
    [ "$r" -gt 4194304 ] && { echo "$(date +%T) KILL $p rss=${r}KB" ; kill $p; }; done
  echo "$(date +%T) free=${free}% swap=${swap}MB load=$(sysctl -n vm.loadavg | awk '{print $2}') ours=$((rss/1024))MB n=$(echo $pids | wc -w)"
  if [ "${free:-100}" -lt 15 ]; then echo "$(date +%T) EMERGENCY: killing all experiment processes"; pkill -f "$PAT"; touch logs_queue/EMERGENCY_STOP
  elif [ "${free:-100}" -lt 30 ]; then
    # swap is not a trigger: macOS keeps swap allocated long after pressure ends, which froze every job
    newest=$(ps -axo pid=,stat=,command= | awk '$2 !~ /T/' | grep -E "$PAT" | grep -v grep | awk '{print $1}' | sort -n | awk 'END{print}')
    [ -n "$newest" ] && { kill -STOP $newest; echo "$(date +%T) PAUSE $newest"; }
  elif [ "${free:-100}" -gt 35 ]; then
    for p in $(ps -axo pid=,stat=,command= | awk '$2 ~ /T/' | grep -E "$PAT" | awk '{print $1}'); do kill -CONT $p; echo "$(date +%T) RESUME $p"; done
  fi
  sleep 20
done >> logs_queue/watchdog.log 2>&1
