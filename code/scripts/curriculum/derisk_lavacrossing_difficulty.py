"""De-risk check for LavaCrossing: do the built-in N=1/2/3 tiers (and per-seed layout
variation) give a PPO learner a real easy->hard difficulty gradient, and does a cheap
geometric quantity track it?

Why LavaCrossing (vs LavaGap)
-----------------------------
LavaGap difficulty is ~1D (a single gap; the optimal path length is constant across gap
positions), which gives almost no easy->hard spread. LavaCrossing places N lava rivers,
each with a crossing, so:
  * N (num_crossings) is a built-in difficulty tier (free external ordering, like Boxoban).
  * Within a tier, the BFS optimal-path length (detour) genuinely VARIES with where the
    crossings land -> a real geometric difficulty signal.

What it does
------------
1. For each tier N in {1,2,3}: enumerate distinct layouts over a seed range and compute the
   BFS shortest-path length start->goal avoiding lava (the geometric difficulty proxy).
2. Train ONE PPO policy (small CNN, 3 nav actions -- the LavaGap lesson) across all tiers
   under the native SPARSE reward.
3. At checkpoints, measure per-layout success and report: success by tier (does N1>N2>N3?)
   and rho(detour, difficulty) within the pooled layouts.

NO LLM. Standard gymnasium MiniGrid CrossingEnv (Version B).
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import numpy as np

WALL, LAVA, GOAL = 2, 9, 8


def make_raw_env(size: int, n: int):
    from minigrid.envs import CrossingEnv

    return CrossingEnv(size=size, num_crossings=n, max_steps=4 * size * size)


def maybe_restrict_actions(env, restrict: bool):
    """Collapse MiniGrid's 7 actions to the 3 navigation actions {left,right,forward}."""
    if not restrict:
        return env
    import gymnasium as gym

    class NavActions(gym.ActionWrapper):
        def __init__(self, e):
            super().__init__(e)
            self.action_space = gym.spaces.Discrete(3)

        def action(self, a):
            return int(a)

    return NavActions(env)


def bfs_detour(size: int, n: int, seed: int):
    """Return (signature, path_len, solvable). path_len = BFS shortest start->goal avoiding lava."""
    from minigrid.wrappers import FullyObsWrapper

    env = FullyObsWrapper(make_raw_env(size, n))
    obs, _ = env.reset(seed=seed)
    img = obs["image"][:, :, 0]  # [x, y] object-id
    w, h = img.shape
    start = env.unwrapped.agent_pos
    goal_cells = list(zip(*np.where(img == GOAL)))
    lava = tuple(sorted(zip(*np.where(img == LAVA))))
    env.close()
    if not goal_cells:
        return None, None, False
    gx, gy = goal_cells[0]
    sx, sy = int(start[0]), int(start[1])

    blocked = np.zeros((w, h), dtype=bool)
    for x in range(w):
        for y in range(h):
            if img[x, y] in (WALL, LAVA):
                blocked[x, y] = True
    blocked[sx, sy] = False
    blocked[gx, gy] = False

    dist = -np.ones((w, h), dtype=int)
    dist[sx, sy] = 0
    q = deque([(sx, sy)])
    while q:
        x, y = q.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < h and not blocked[nx, ny] and dist[nx, ny] < 0:
                dist[nx, ny] = dist[x, y] + 1
                q.append((nx, ny))
    path_len = int(dist[gx, gy])
    solvable = path_len >= 0
    sig = (n, lava, (sx, sy), (gx, gy))
    return sig, path_len, solvable


def enumerate_tiers(size: int, tiers, n_seeds: int):
    """{sig: {'seeds':[...], 'n':N, 'path_len':int}} across all tiers."""
    groups: dict = {}
    for n in tiers:
        for s in range(n_seeds):
            sig, path_len, ok = bfs_detour(size, n, s)
            if sig is None or not ok:
                continue
            if sig not in groups:
                groups[sig] = {"seeds": [], "n": n, "path_len": path_len}
            groups[sig]["seeds"].append(s)
    return groups


def build_small_cnn():
    import torch as th
    import torch.nn as nn
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class SmallCNN(BaseFeaturesExtractor):
        def __init__(self, obs_space, features_dim: int = 128):
            super().__init__(obs_space, features_dim)
            n_in = obs_space.shape[0]
            self.cnn = nn.Sequential(
                nn.Conv2d(n_in, 16, 2, 1), nn.ReLU(),
                nn.Conv2d(16, 32, 2, 1), nn.ReLU(),
                nn.Conv2d(32, 64, 2, 1), nn.ReLU(),
                nn.Flatten(),
            )
            with th.no_grad():
                n_flat = self.cnn(th.zeros(1, *obs_space.shape)).shape[1]
            self.lin = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

        def forward(self, x):
            return self.lin(self.cnn(x))

    return SmallCNN


def make_pool_env(size: int, pool, rng_seed: int, restrict: bool):
    """pool = list of (n, seed). On reset, pick one and build that tier's layout."""
    import gymnasium as gym
    from minigrid.wrappers import ImgObsWrapper

    class PoolEnv(gym.Wrapper):
        def __init__(self):
            # all tiers share obs/action space; base env tier is swapped per reset
            self._size = size
            self._pool = list(pool)
            self._rng = np.random.default_rng(rng_seed)
            self._restrict = restrict
            n0, _ = self._pool[0]
            super().__init__(maybe_restrict_actions(ImgObsWrapper(make_raw_env(size, n0)), restrict))
            self.current = None

        def reset(self, **kwargs):
            kwargs.pop("seed", None)
            n, s = self._pool[self._rng.integers(len(self._pool))]
            self.env = maybe_restrict_actions(
                ImgObsWrapper(make_raw_env(self._size, n)), self._restrict)
            self.current = (n, s)
            return self.env.reset(seed=int(s), **kwargs)

    return PoolEnv()


def eval_groups(model, size: int, groups, episodes: int, restrict: bool):
    from minigrid.wrappers import ImgObsWrapper

    rates = {}
    for sig, info in groups.items():
        n, seed = info["n"], info["seeds"][0]
        env = maybe_restrict_actions(ImgObsWrapper(make_raw_env(size, n)), restrict)
        succ = 0
        for _ in range(episodes):
            obs, _ = env.reset(seed=seed)
            total, done = 0.0, False
            while not done:
                a, _ = model.predict(obs, deterministic=False)
                obs, r, term, trunc, _ = env.step(int(a))
                total += r
                done = term or trunc
            if total > 0:
                succ += 1
        env.close()
        rates[sig] = succ / episodes
    return rates


def main():
    import warnings

    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=9)
    ap.add_argument("--tiers", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n-seeds", type=int, default=40)
    ap.add_argument("--total-steps", type=int, default=600_000)
    ap.add_argument("--eval-every", type=int, default=100_000)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--restrict-actions", action="store_true")
    ap.add_argument("--enumerate-only", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.total_steps, args.eval_every, args.eval_episodes, args.n_seeds = 8_000, 4_000, 5, 12

    t0 = time.time()
    print(f"[enumerate] size={args.size} tiers={args.tiers} over {args.n_seeds} seeds/tier", flush=True)
    groups = enumerate_tiers(args.size, args.tiers, args.n_seeds)
    for n in args.tiers:
        sub = {k: v for k, v in groups.items() if v["n"] == n}
        pls = sorted({v["path_len"] for v in sub.values()})
        print(f"  N={n}: {len(sub)} distinct layouts, detour range {min(pls) if pls else '-'}"
              f"..{max(pls) if pls else '-'} (values {pls})")
    if args.enumerate_only:
        print(f"[done] {time.time()-t0:.1f}s")
        return

    from stable_baselines3 import PPO
    from scipy.stats import spearmanr

    pool = [(v["n"], s) for v in groups.values() for s in v["seeds"]]
    env = make_pool_env(args.size, pool, args.seed, args.restrict_actions)
    pk = dict(features_extractor_class=build_small_cnn(),
              features_extractor_kwargs=dict(features_dim=128), net_arch=[64, 64])
    model = PPO("CnnPolicy", env, n_steps=2048, batch_size=256, n_epochs=4, gamma=0.99,
                ent_coef=0.01, learning_rate=3e-4, policy_kwargs=pk, seed=args.seed,
                verbose=0, device="cpu")

    n_ck = max(1, args.total_steps // args.eval_every)
    print(f"\n[train] {args.total_steps} steps, eval every {args.eval_every} ({n_ck} checkpoints)\n",
          flush=True)
    for ck in range(1, n_ck + 1):
        model.learn(total_timesteps=args.eval_every, reset_num_timesteps=(ck == 1))
        rates = eval_groups(model, args.size, groups, args.eval_episodes, args.restrict_actions)
        by_tier = {}
        for sig, sr in rates.items():
            by_tier.setdefault(groups[sig]["n"], []).append(sr)
        tier_str = "  ".join(f"N{n}={np.mean(v):.2f}" for n, v in sorted(by_tier.items()))
        pls = np.array([groups[s]["path_len"] for s in rates])
        srs = np.array(list(rates.values()))
        rho = spearmanr(pls, 1 - srs).correlation if len(set(pls)) > 1 else float("nan")
        print(f"  steps={ck*args.eval_every:>7d}  {tier_str}   "
              f"overall={srs.mean():.2f}  rho(detour,diff)={rho:+.2f}", flush=True)

    # verdict: tier monotonicity + detour correlation
    print("\n[VERDICT]")
    tier_means = {n: np.mean([rates[s] for s in rates if groups[s]['n'] == n]) for n in args.tiers}
    print(f"  final success by tier: {{" + ", ".join(f'N{n}:{m:.2f}' for n, m in tier_means.items()) + "}")
    mono = all(tier_means[args.tiers[i]] >= tier_means[args.tiers[i + 1]] - 0.05
               for i in range(len(args.tiers) - 1))
    spread = max(tier_means.values()) - min(tier_means.values())
    if mono and spread >= 0.2:
        print(f"  -> TIER GRADIENT EXISTS (easy N1 -> hard N{max(args.tiers)}), spread {spread:.2f}. "
              "Curriculum has a real ladder.")
    elif spread >= 0.2:
        print(f"  -> difficulty spread {spread:.2f} exists but not cleanly tier-monotone; usable.")
    else:
        print(f"  -> WEAK tier spread {spread:.2f}; reconsider budget/size.")
    print(f"\n[done] total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
