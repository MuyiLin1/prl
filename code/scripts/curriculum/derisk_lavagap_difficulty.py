"""De-risk check: does LavaGap difficulty actually VARY across gap positions for a
PPO learner, and does the geometric proxy (BFS detour length) track that difficulty?

Why this exists
---------------
The whole neural-curriculum experiment only makes sense if (a) different gap layouts
have genuinely different difficulty for a PPO policy (so an easy->hard ordering has
something to order), and (b) a cheap geometric quantity correlates with that difficulty
(so our predictor can build the ordering). LavaGap difficulty ~ gap position ~ geometry,
which is a near-collapse that killed an earlier curation experiment. Before building the
full 7-condition harness we verify the difficulty gradient exists.

What it does
------------
1. Enumerate distinct LavaGap layouts across a seed range; for each, compute a geometric
   difficulty proxy = BFS shortest-path length (start -> goal) avoiding lava (the detour).
2. Train ONE PPO policy (small CNN) on the pool of layouts under the NATIVE SPARSE reward.
3. At several training checkpoints, measure per-layout success RATE (stochastic rollouts).
4. Report whether per-layout success spreads out (gradient exists) and whether it
   correlates with the geometric proxy (geometry tracks PPO difficulty).

NO LLM anywhere. Standard gymnasium MiniGrid LavaGap (Version B).
"""

from __future__ import annotations

import argparse
import time
from collections import deque, defaultdict

import numpy as np

# MiniGrid object ids (minigrid.core.constants.OBJECT_TO_IDX)
WALL, LAVA, GOAL, AGENT = 2, 9, 8, 10


def make_raw_env(size: int):
    """A LavaGap env at a given grid size (Version B = standard gymnasium MiniGrid)."""
    from minigrid.envs import LavaGapEnv

    return LavaGapEnv(size=size, max_steps=4 * size * size)


def maybe_restrict_actions(env, restrict: bool):
    """Optionally collapse MiniGrid's 7 actions to the 3 navigation actions.

    MiniGrid exposes turn-left(0), turn-right(1), forward(2), pickup(3), drop(4),
    toggle(5), done(6). For LavaGap only the first three matter; the other four only
    enlarge the exploration problem. Restricting to {0,1,2} is standard practice and is
    usually what lets PPO learn MiniGrid navigation under sparse reward.
    """
    if not restrict:
        return env
    import gymnasium as gym

    class NavActions(gym.ActionWrapper):
        def __init__(self, e):
            super().__init__(e)
            self.action_space = gym.spaces.Discrete(3)

        def action(self, a):
            return int(a)  # 0,1,2 already map to left,right,forward

    return NavActions(env)


def measure_layout(size: int, seed: int):
    """Reset a layout and return (signature, geom, solvable).

    signature uniquely identifies the layout (lava cells + agent + goal). geom is the
    GAP-OFFSET geometric difficulty proxy: the lateral distance between the agent's start
    position and the single gap in the lava barrier. (The BFS optimal-path length is
    CONSTANT across gap positions in LavaGap, so it carries no signal; under partial
    observability the gap's offset from the straight route is the exploration-cost proxy.)
    """
    from minigrid.wrappers import FullyObsWrapper

    env = FullyObsWrapper(make_raw_env(size))
    obs, _ = env.reset(seed=seed)
    img = obs["image"][:, :, 0]  # object-id channel, indexed [x, y]

    lava = list(zip(*np.where(img == LAVA)))  # list of (x, y)
    start = env.unwrapped.agent_pos  # (x, y)
    goal_cells = list(zip(*np.where(img == GOAL)))
    env.close()
    if not goal_cells or not lava:
        return None, None, False
    goal = goal_cells[0]
    sx, sy = int(start[0]), int(start[1])
    gx, gy = int(goal[0]), int(goal[1])

    xs = sorted({p[0] for p in lava})
    ys = sorted({p[1] for p in lava})
    # The barrier is a single line; the gap is the missing index along the varying axis.
    if len(ys) == 1:  # horizontal barrier at y=ys[0]; gap varies in x
        line = ys[0]
        present = {p[0] for p in lava}
        full = range(min(xs), max(xs) + 1)
        gap_axis = [c for c in full if c not in present]
        gap = (gap_axis[0] if gap_axis else min(xs) - 1, line)
        geom = abs(gap[0] - sx)  # lateral offset of gap from agent start (along barrier)
    elif len(xs) == 1:  # vertical barrier at x=xs[0]; gap varies in y
        line = xs[0]
        present = {p[1] for p in lava}
        full = range(min(ys), max(ys) + 1)
        gap_axis = [c for c in full if c not in present]
        gap = (line, gap_axis[0] if gap_axis else min(ys) - 1)
        geom = abs(gap[1] - sy)  # lateral offset of gap from agent start (along barrier)
    else:
        gap = None
        geom = None

    if geom is None:
        return None, None, False
    sig = (tuple(sorted(lava)), (sx, sy), (gx, gy))
    return sig, int(geom), True


def enumerate_positions(size: int, n_seeds: int):
    """Group seeds by layout signature; return {sig: {'seeds':[...], 'geom':int}}."""
    groups: dict = {}
    for s in range(n_seeds):
        sig, geom, solvable = measure_layout(size, s)
        if sig is None or not solvable:
            continue
        if sig not in groups:
            groups[sig] = {"seeds": [], "geom": geom}
        groups[sig]["seeds"].append(s)
    return groups


# --------------------------------------------------------------------------- #
# PPO training env: each reset samples a layout seed from a fixed pool.
# --------------------------------------------------------------------------- #
def build_small_cnn():
    import torch as th
    import torch.nn as nn
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class SmallCNN(BaseFeaturesExtractor):
        def __init__(self, obs_space, features_dim: int = 128):
            super().__init__(obs_space, features_dim)
            n_in = obs_space.shape[0]  # channels-first after SB3 transpose
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


def make_seedpool_env(size: int, seed_pool, rng_seed: int, restrict: bool = False):
    import gymnasium as gym
    from minigrid.wrappers import ImgObsWrapper

    class SeedPoolEnv(gym.Wrapper):
        """On each reset, regenerate the layout from a random seed in the pool."""

        def __init__(self):
            super().__init__(maybe_restrict_actions(ImgObsWrapper(make_raw_env(size)), restrict))
            self._pool = list(seed_pool)
            self._rng = np.random.default_rng(rng_seed)
            self.current_seed = None

        def reset(self, **kwargs):
            kwargs.pop("seed", None)
            s = int(self._rng.choice(self._pool))
            self.current_seed = s
            return self.env.reset(seed=s, **kwargs)

    return SeedPoolEnv()


def eval_per_layout(model, size: int, groups, episodes: int, rng_seed: int,
                    restrict: bool = False):
    """Stochastic success rate per layout signature -> dict sig -> success_rate."""
    from minigrid.wrappers import ImgObsWrapper

    env = maybe_restrict_actions(ImgObsWrapper(make_raw_env(size)), restrict)
    rng = np.random.default_rng(rng_seed)
    out = {}
    for sig, info in groups.items():
        seed = info["seeds"][0]  # layout fully determined by signature
        succ = 0
        for _ in range(episodes):
            obs, _ = env.reset(seed=seed)
            total = 0.0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=False)
                obs, r, term, trunc, _ = env.step(int(action))
                total += r
                done = term or trunc
            if total > 0:  # reached goal (lava death / timeout give 0)
                succ += 1
        out[sig] = succ / episodes
    env.close()
    return out


def main():
    import warnings

    warnings.filterwarnings("ignore")

    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=7)
    ap.add_argument("--n-seeds", type=int, default=300, help="seed range to enumerate layouts")
    ap.add_argument("--total-steps", type=int, default=300_000)
    ap.add_argument("--eval-every", type=int, default=50_000)
    ap.add_argument("--eval-episodes", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--enumerate-only", action="store_true")
    ap.add_argument("--restrict-actions", action="store_true",
                    help="collapse to 3 nav actions (standard MiniGrid fix)")
    ap.add_argument("--train-max-offset", type=int, default=None,
                    help="train pool only includes layouts with gap_offset <= this (eval still all)")
    ap.add_argument("--single-seed", type=int, default=None,
                    help="train on exactly one layout seed (unit test of solvability)")
    ap.add_argument("--quick", action="store_true", help="tiny budget smoke test")
    args = ap.parse_args()

    if args.quick:
        args.total_steps = 6_000
        args.eval_every = 3_000
        args.eval_episodes = 6
        args.n_seeds = 60

    t0 = time.time()
    print(f"[enumerate] size={args.size} over {args.n_seeds} seeds ...", flush=True)
    groups = enumerate_positions(args.size, args.n_seeds)
    geoms = sorted({info["geom"] for info in groups.values()})
    print(f"[enumerate] distinct layouts: {len(groups)}")
    print(f"[enumerate] gap-offset (geom) values: {geoms}")
    for sig, info in sorted(groups.items(), key=lambda kv: kv[1]["geom"]):
        print(f"    gap_offset={info['geom']:2d}  n_seeds={len(info['seeds']):3d}  "
              f"example_seed={info['seeds'][0]}")
    if len(groups) < 2:
        print("[VERDICT] <2 distinct layouts -> no difficulty gradient possible at this size. "
              "Try a larger --size.")
        return
    if args.enumerate_only:
        print(f"[done] enumerate-only in {time.time() - t0:.1f}s")
        return

    # ----- train one PPO policy on the whole layout pool ----------------------
    from stable_baselines3 import PPO

    if args.single_seed is not None:
        pool = [args.single_seed]
        print(f"[train-pool] SINGLE layout seed={args.single_seed}")
    elif args.train_max_offset is not None:
        pool = [s for info in groups.values() if info["geom"] <= args.train_max_offset
                for s in info["seeds"]]
        print(f"[train-pool] layouts with gap_offset <= {args.train_max_offset}: "
              f"{len(pool)} seeds")
    else:
        pool = [s for info in groups.values() for s in info["seeds"]]
    train_env = make_seedpool_env(args.size, pool, rng_seed=args.seed,
                                  restrict=args.restrict_actions)
    policy_kwargs = dict(
        features_extractor_class=build_small_cnn(),
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[64, 64],
    )
    model = PPO(
        "CnnPolicy",
        train_env,
        n_steps=2048,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        ent_coef=0.01,  # encourage exploration under sparse reward
        learning_rate=3e-4,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        verbose=0,
        device="cpu",
    )

    from scipy.stats import spearmanr

    n_checkpoints = max(1, args.total_steps // args.eval_every)
    print(f"\n[train] {args.total_steps} steps, eval every {args.eval_every} "
          f"({n_checkpoints} checkpoints)\n", flush=True)
    history = []
    for ck in range(1, n_checkpoints + 1):
        model.learn(total_timesteps=args.eval_every, reset_num_timesteps=(ck == 1),
                    progress_bar=False)
        steps = ck * args.eval_every
        rates = eval_per_layout(model, args.size, groups, args.eval_episodes,
                                rng_seed=1000 + ck, restrict=args.restrict_actions)
        arr = np.array(list(rates.values()))
        gms = np.array([groups[sig]["geom"] for sig in rates])
        # difficulty = 1 - success; correlate with gap-offset
        rho = spearmanr(gms, 1 - arr).correlation if len(set(gms)) > 1 else float("nan")
        history.append((steps, arr.mean(), arr.min(), arr.max(), arr.std(), rho))
        print(f"  steps={steps:>7d}  succ mean={arr.mean():.2f} "
              f"min={arr.min():.2f} max={arr.max():.2f} spread={arr.max()-arr.min():.2f} "
              f"std={arr.std():.2f}  rho(gap_offset,diff)={rho:+.2f}", flush=True)

    # ----- verdict ------------------------------------------------------------
    best_spread = max(h[3] - h[2] for h in history)
    rhos = [h[5] for h in history if not np.isnan(h[5])]
    mean_rho = float(np.mean(rhos)) if rhos else float("nan")
    print("\n[VERDICT]")
    print(f"  max per-layout success spread across checkpoints: {best_spread:.2f}")
    print(f"  mean rho(gap_offset, difficulty): {mean_rho:+.2f}")
    if best_spread >= 0.3:
        print("  -> DIFFICULTY GRADIENT EXISTS (layouts separate). Curriculum has signal.")
    else:
        print("  -> WEAK/NO gradient (layouts ~equally hard). Reconsider size/task before building.")
    if not np.isnan(mean_rho) and abs(mean_rho) >= 0.3:
        print("  -> GEOMETRY (gap-offset) TRACKS PPO difficulty. Predictor can build the ordering.")
    elif not np.isnan(mean_rho):
        print("  -> Geometry weakly tracks PPO difficulty; probe may be needed.")
    print(f"\n[done] total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
