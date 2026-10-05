"""LLM-as-judge difficulty baseline (local Qwen2.5-7B-Instruct, 4-bit, Apple GPU via MLX).

Input : JSONL files with {"id", "domain", "prompt"} (one per environment instance).
Output: CSV next to the input with id, llm_expected (E[digit] under the model's next-token
        distribution over 0..9 -> graded score), llm_argmax, p0..p9.

Resumable: ids already in the output CSV are skipped, so the queue runner can restart it freely.
No sampling, no chain-of-thought: one forward pass per instance (zero-shot, deterministic).
usage: ~/.venvs/mlx/bin/python llm_judge_mlx.py <in.jsonl> [<in2.jsonl> ...]
"""
import csv
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

MODEL = str(Path.home() / ".venvs" / "qwen25-7b-instruct-4bit")
SYSTEM = ("You are an expert in reinforcement learning and puzzle design. You estimate how hard an "
          "environment instance is for a learning agent, judging only from its description and layout.")


def main(paths):
    # Cap MLX GPU memory: without this MLX keeps a buffer cache that grows with every new prompt length.
    mx.set_memory_limit(8 * 1024 ** 3)
    mx.set_cache_limit(1 * 1024 ** 3)
    model, tok = load(MODEL)
    digit_ids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)]
    for p in paths:
        p = Path(p)
        out = p.with_name(p.stem + "_llm_qwen7b.csv")
        done = set()
        if out.exists():
            done = {r["id"] for r in csv.DictReader(open(out))}
        items = [json.loads(l) for l in open(p)]
        items = [it for it in items if it["id"] not in done]
        print(f"[{p.name}] {len(items)} to score ({len(done)} already done)", flush=True)
        new = not out.exists()
        with open(out, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["id", "llm_expected", "llm_argmax"] + [f"p{d}" for d in range(10)])
            t0 = time.time()
            for k, it in enumerate(items):
                msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": it["prompt"]}]
                text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
                ids = mx.array(tok.encode(text))[None]
                logits = model(ids)[0, -1].astype(mx.float32)
                lp = np.array(logits[mx.array(digit_ids)])
                pr = np.exp(lp - lp.max())
                pr /= pr.sum()
                w.writerow([it["id"], float((pr * np.arange(10)).sum()), int(pr.argmax())]
                           + [f"{v:.4f}" for v in pr])
                if k % 50 == 0:
                    mx.clear_cache()
                if k % 100 == 0:
                    fh.flush()
                    print(f"  {k}/{len(items)}  {(time.time() - t0) / max(1, k):.2f}s/item", flush=True)
        print(f"[{p.name}] done -> {out.name}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
