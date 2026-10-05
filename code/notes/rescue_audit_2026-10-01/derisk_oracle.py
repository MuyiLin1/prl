"""De-risk: is per-layout LavaCrossing difficulty (as measured by a flat-trained reference PPO)
reliable across reference seeds, overall and WITHIN (size, N) cells?

Trains one reference PPO (flat, whole pool, same config as curriculum_harness.oracle_scores)
and records, per training-pool layout and per eval episode, success and BFS-progress
((d_start - d_min) / d_start, lava-blocked BFS distance to goal). Run once per ref seed.

usage: python derisk_oracle.py <tag> <ref_seed> <steps> <sizes...>
"""
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "/Users/linmuyi/code/temp-prl/scripts/curriculum")
import numpy as np
import torch

torch.set_num_threads(1)
import warnings

warnings.filterwarnings("ignore")
import curriculum_harness as H
from difficulty_scoring import LAVA, WALL, _grid_objects, enumerate_pool

tag, ref_seed, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
sizes = [int(s) for s in sys.argv[4:]]
OUT = Path(__file__).resolve().parent / "derisk_out"
OUT.mkdir(exist_ok=True)
EPISODES = 20


def dist_map(img, goal):
    w, h = img.shape
    d = np.full((w, h), -1, dtype=int)
    d[goal] = 0
    q = deque([goal])
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and d[nx, ny] < 0 and img[nx, ny] not in (WALL, LAVA):
                d[nx, ny] = d[x, y] + 1
                q.append((nx, ny))
    return d


pool = enumerate_pool(sizes, [1, 2, 3], 80, 10)
t0 = time.time()
env = H.CurriculumPoolEnv(True, rng_seed=777 + ref_seed)
env.set_active(pool)
model = H.make_ppo(env, seed=777 + ref_seed)
model.learn(total_timesteps=steps)
t_train = time.time() - t0

rows = []
for lo in pool:
    img, start, goal = _grid_objects(lo.size, lo.n, lo.seed)
    dm = dist_map(img, goal)
    d0 = dm[start]
    e = H.wrap(lo.size, lo.n, True)
    succ, prog = [], []
    for _ in range(EPISODES):
        obs, _ = e.reset(seed=lo.seed)
        done, total, dmin = False, 0.0, d0
        while not done:
            a, _ = model.predict(obs, deterministic=False)
            obs, r, term, trunc, _ = e.step(int(a))
            total += r
            p = tuple(int(v) for v in e.unwrapped.agent_pos)
            if dm[p] >= 0:
                dmin = min(dmin, dm[p])
            done = term or trunc
        succ.append(int(total > 0))
        prog.append(1.0 if total > 0 else (d0 - dmin) / d0)
    e.close()
    rows.append(dict(size=lo.size, n=lo.n, seed=lo.seed, sig=json.dumps(lo.sig), succ=succ, prog=prog,
                     detour=lo.features["detour"], n_lava=lo.features["n_lava"]))

out = OUT / f"{tag}_ref{ref_seed}.json"
out.write_text(json.dumps(dict(tag=tag, ref_seed=ref_seed, steps=steps, sizes=sizes, episodes=EPISODES,
                               train_s=round(t_train, 1), total_s=round(time.time() - t0, 1), rows=rows)))
print(f"done {out} train {t_train:.0f}s total {time.time() - t0:.0f}s")
