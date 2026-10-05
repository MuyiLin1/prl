"""SFL (Sampling For Learnability; Rutherford et al., NeurIPS 2024) on LavaCrossing, with and without
our cheap difficulty grader. Same PPO learner, pool generator, held-out set and budget as
curriculum_harness.py.

SFL: every `refresh` env-steps, draw C fresh candidate layouts from the generator, roll the CURRENT
policy out k times on each to estimate success p, keep the B layouts with the highest learnability
p(1-p) as a buffer, and train on buffer + an equal number of fresh random layouts.

Conditions
  sfl          C=100 candidates scored by rollouts (standard SFL; expensive scoring).
  sfl_small    C=20 candidates scored by rollouts (same scoring budget as sfl_grader).
  sfl_grader   C=100 candidates; only M=20 random ones are rolled out, to fit a 1-D logistic map
               "grader score -> current success". Every candidate's p is then PREDICTED from its grader
               score (free), and the B with highest predicted p(1-p) form the buffer.
Scoring rollout env-steps are logged separately from training env-steps.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from curriculum_harness import (CACHE_DIR, GRADER_CALIB, CurriculumPoolEnv, _code_fingerprint,
                                enumerate_pool, make_ppo, tuple_key, wrap)
from difficulty_scoring import Layout, fit_grader, geometric_features

HERE = Path(__file__).resolve().parent


def rollout_stats(model, layouts, episodes, restrict=True):
    """Per-layout success rate and the env-steps spent measuring it."""
    rates, steps = [], 0
    for lo in layouts:
        env = wrap(lo.size, lo.n, restrict)
        succ = 0
        for _ in range(episodes):
            obs, _ = env.reset(seed=lo.seed)
            done, total = False, 0.0
            while not done:
                a, _ = model.predict(obs, deterministic=False)
                obs, r, term, trunc, _ = env.step(int(a))
                total += r
                steps += 1
                done = term or trunc
            succ += total > 0
        env.close()
        rates.append(succ / episodes)
    return np.array(rates), steps


def fit_logistic_1d(x, p):
    """Weighted logistic fit of success rate p on score x (gradient descent; tiny problem)."""
    xm, xs = x.mean(), x.std() + 1e-9
    z = (x - xm) / xs
    a, b = 0.0, 0.0
    for _ in range(2000):
        q = 1 / (1 + np.exp(-(a + b * z)))
        a -= 0.5 * np.mean(q - p)
        b -= 0.5 * np.mean((q - p) * z)
    return lambda xn: 1 / (1 + np.exp(-(a + b * (xn - xm) / xs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["sfl", "sfl_small", "sfl_grader"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sizes", type=int, nargs="+", default=[9, 13, 17])
    ap.add_argument("--tiers", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--total-steps", type=int, default=1_500_000)
    ap.add_argument("--eval-every", type=int, default=50_000)
    ap.add_argument("--refresh", type=int, default=250_000)
    ap.add_argument("--candidates", type=int, default=100)
    ap.add_argument("--calib", type=int, default=20, help="rolled-out candidates (sfl_small / sfl_grader)")
    ap.add_argument("--k", type=int, default=4, help="rollouts per scored candidate")
    ap.add_argument("--buffer", type=int, default=30)
    ap.add_argument("--gen-seed-range", type=int, default=50_000, help="generator seeds for candidates")
    ap.add_argument("--eval-seed-start", type=int, default=100_000)
    ap.add_argument("--eval-episodes", type=int, default=4)
    ap.add_argument("--results-dir", default="results_v2_sfl_lc91317")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.sizes, args.total_steps, args.eval_every, args.refresh = [7, 9], 40_000, 10_000, 20_000
        args.candidates, args.calib, args.buffer, args.k = 20, 6, 6, 2

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    # candidate generator: fresh distinct layouts drawn from a large seed range of every (size, N) cell
    # held-out set: identical to curriculum_harness (seeds >= 100k); candidates come from seeds < 100k
    eval_pool = enumerate_pool(args.sizes, args.tiers, 200, 4, seed_start=args.eval_seed_start)
    eval_sigs = {lo.sig for lo in eval_pool}
    print(f"[{args.condition}] held-out={len(eval_pool)}", flush=True)

    predict = None
    if args.condition == "sfl_grader":
        calib = {tuple_key(k): v for k, v in json.loads((CACHE_DIR / GRADER_CALIB).read_text()).items()}
        cp = enumerate_pool([7, 9, 11], [1, 2, 3], 80, 10)
        predict = fit_grader(cp, calib, seed=0)

    def draw(n):
        """Sample n fresh distinct solvable layouts from the generator (uniform cell, uniform seed)."""
        out, seen = [], set()
        while len(out) < n:
            size, tier = int(rng.choice(args.sizes)), int(rng.choice(args.tiers))
            seed = int(rng.integers(args.gen_seed_range))
            res = geometric_features(size, tier, seed)
            if res is None or res[1] in seen or res[1] in eval_sigs:
                continue
            seen.add(res[1])
            out.append(Layout(n=tier, seed=seed, size=size, sig=res[1], features=res[0]))
        return out

    env = CurriculumPoolEnv(True, rng_seed=args.seed)
    env.set_active(draw(args.buffer * 2))
    model = make_ppo(env, seed=args.seed)
    curve, refreshes, score_steps, next_refresh = [], [], 0, 0
    n_ck = args.total_steps // args.eval_every
    for ck in range(1, n_ck + 1):
        env_steps_before = (ck - 1) * args.eval_every
        if env_steps_before >= next_refresh:
            cands = draw(args.candidates)
            tr = time.time()
            if args.condition == "sfl":
                p, st = rollout_stats(model, cands, args.k)
                pick = [cands[i] for i in np.argsort(-(p * (1 - p)), kind="stable")[:args.buffer]]
                info = dict(rolled=len(cands), mean_p=float(p.mean()))
            elif args.condition == "sfl_small":
                sub = cands[:args.calib]
                p, st = rollout_stats(model, sub, args.k)
                pick = [sub[i] for i in np.argsort(-(p * (1 - p)), kind="stable")[:args.buffer]]
                info = dict(rolled=len(sub), mean_p=float(p.mean()))
            else:
                g = predict(cands)
                sub_idx = np.arange(args.calib)
                p, st = rollout_stats(model, [cands[i] for i in sub_idx], args.k)
                if p.min() == p.max():          # degenerate calibration: all fail / all succeed
                    order = np.argsort(g) if p.max() == 0 else np.argsort(-g)  # easiest / hardest first
                else:
                    ph = fit_logistic_1d(g[sub_idx], p)(g)
                    order = np.argsort(-(ph * (1 - ph)), kind="stable")
                pick = [cands[i] for i in order[:args.buffer]]
                info = dict(rolled=args.calib, mean_p=float(p.mean()), degenerate=bool(p.min() == p.max()))
            score_steps += st
            fresh = draw(args.buffer)
            env.set_active(pick + fresh)
            refreshes.append(dict(env_steps=env_steps_before, scoring_steps=st,
                                  scoring_wall_s=round(time.time() - tr, 1), **info,
                                  buffer_cells=sorted(f"s{lo.size}N{lo.n}" for lo in pick)))
            next_refresh += args.refresh
        model.learn(total_timesteps=args.eval_every, reset_num_timesteps=(ck == 1))
        held_r, _ = rollout_stats(model, eval_pool, args.eval_episodes)
        held = float(held_r.mean())
        curve.append(dict(env_steps=ck * args.eval_every, held_success=round(held, 4),
                          scoring_steps_cum=score_steps, wall_s=round(time.time() - t0, 1)))
        print(f"  steps={ck * args.eval_every:>7d} held={held:.2f} scoring_steps={score_steps} "
              f"({time.time() - t0:.0f}s)", flush=True)

    res = dict(condition=args.condition, seed=args.seed, final_held_success=curve[-1]["held_success"],
               training_steps=args.total_steps, scoring_steps=score_steps, wall_s=round(time.time() - t0, 1),
               curve=curve, refreshes=refreshes, args=vars(args), code=_code_fingerprint())
    out = HERE / args.results_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.condition}_seed{args.seed}.json").write_text(json.dumps(res, indent=2))
    print(f"[{args.condition}] DONE final={res['final_held_success']:.2f} scoring_steps={score_steps}", flush=True)


if __name__ == "__main__":
    main()
