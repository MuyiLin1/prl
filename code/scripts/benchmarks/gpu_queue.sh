#!/bin/bash
# Always-on GPU queue: score every prompt file in data/benchmarks/llm_prompts with the local Qwen judge.
# Picks up new *.jsonl files as they appear. Stop with: touch data/benchmarks/llm_prompts/STOP
cd "$(dirname "$0")/../.."
D=data/benchmarks/llm_prompts
while [ ! -f $D/STOP ]; do
  todo=""
  for f in $(ls -Sr $D/*.jsonl); do                       # smallest first -> early results
    out=${f%.jsonl}_llm_qwen7b.csv
    n=$(wc -l < $f); m=$([ -f $out ] && echo $(( $(wc -l < $out) - 1 )) || echo 0)
    [ "$m" -lt "$n" ] && todo="$todo $f"
  done
  if [ -n "$todo" ]; then ~/.venvs/mlx/bin/python -u scripts/benchmarks/llm_judge_mlx.py $todo; else sleep 60; fi
done
