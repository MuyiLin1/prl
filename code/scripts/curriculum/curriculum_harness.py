"""Neural curriculum experiment on LavaCrossing (sizes x tiers).

Tests the paper's thesis operationally: can a CHEAP geometric difficulty estimate build an
easy->hard training curriculum that learns as efficiently as an EXPENSIVE empirical (oracle)
one -- and better than no curriculum (random)?

Conditions (each is just a different way to SCORE every layout's difficulty, then train a
competence-gated easy->hard curriculum on that ordering):

  random          no ordering; train on the whole pool uniformly from the start (baseline).
  oracle          empirical difficulty = 1 - success of a reference PPO (expensive, cached).
  geometry        cheap grid-shape score g(layout) = rivers + size + detour   (the paper's method).
  probe           cheap behavioral score = random-walk closest-distance-to-goal (no network).
  combined        rank-average of geometry + probe.
  adaptive_fine   re-score with the CURRENT policy every checkpoint (difficulty = 1 - success).
  adaptive_coarse re-score with the current policy only when a bin is unlocked.

Headline metric: ENV-STEPS to reach a target success on a DISJOINT held-out layout set
(hardware-independent). Also logs final held-out success at budget and wall-clock.

Curriculum mechanic: start with the easiest bin only; when success on the active (unlocked)
layouts reaches the competence gate, unlock the next bin; repeat. Random unlocks everything
at the start.

NO LLM. Standard gymnasium MiniGrid CrossingEnv (Version B). 3 nav actions (the LavaGap lesson).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from difficulty_scoring import (
    Layout,
    bin_layouts,
    enumerate_pool,
    geometric_score,
    make_raw_env,
    random_walk_probe,
    fit_grader,
    GRADER_FEATURES,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
CONDITIONS = ["random", "oracle", "geometry", "probe", "combined",
              "adaptive_fine", "adaptive_coarse",
              # v2 (2026-10-02, see PREREGISTRATION.md)
              "flat", "size_order", "random_curriculum", "rivers_only", "grader"]
# v2 names for the original conditions (identical behaviour; the old names stay valid so
# existing result folders keep aggregating).
ALIASES = {"flat": "random", "size_order": "geometry"}
FLAT_CONDITIONS = {"random", "flat"}
SCORING_METHOD = {
    "random": "none (flat)", "flat": "none (flat)",
    "geometry": "geometric_score = est_rivers + (size-7)/2 + detour/100 (== size order on 9/13/17)",
    "size_order": "geometric_score = est_rivers + (size-7)/2 + detour/100 (== size order on 9/13/17)",
    "random_curriculum": "uniform random score per layout (seeded), curriculum mechanics on",
    "rivers_only": "tier N (num_crossings) + seeded random tiebreak",
    "grader": "RF(300) on GRADER_FEATURES, fit on cache/grader_calib_p7911_r6.json (sizes 7/9/11)",
    "oracle": "reference PPO 1-success (cached)", "probe": "random-walk probe",
    "combined": "rank(geometric_score)+rank(probe)", "adaptive_fine": "geometric_score then live re-score every ck",
    "adaptive_coarse": "geometric_score then live re-score at unlocks",
}
GRADER_CALIB = "grader_calib_p7911_r6.json"


# --------------------------------------------------------------------------- envs / model

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


def wrap(size: int, n: int, restrict: bool):
    from minigrid.wrappers import ImgObsWrapper

    return maybe_restrict_actions(ImgObsWrapper(make_raw_env(size, n)), restrict)


def build_small_cnn():
    """Small CNN feature extractor (MiniGrid obs is 7x7x3; Atari-size kernels would crash)."""
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


class CurriculumPoolEnv(gym.Env):
    """Single env whose reset() samples a layout from a mutable ACTIVE set.

    The harness mutates the active set (via set_active) as bins unlock; because PPO holds a
    reference to this same object, the change is picked up on the next reset. Sizes/tiers vary
    per layout but the observation is always 7x7x3, so obs/action spaces stay fixed.
    """

    metadata = {"render_modes": []}

    def __init__(self, restrict: bool, rng_seed: int = 0):
        super().__init__()
        self._restrict = restrict
        self._rng = np.random.default_rng(rng_seed)
        self._active: list[Layout] = []
        self.env = wrap(9, 1, restrict)  # placeholder; spaces match all layouts
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.render_mode = None

    def set_active(self, layouts):
        self._active = list(layouts)

    def reset(self, *, seed=None, options=None):
        lo = self._active[int(self._rng.integers(len(self._active)))]
        self.env = wrap(lo.size, lo.n, self._restrict)
        return self.env.reset(seed=int(lo.seed))

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


def make_ppo(env, seed: int):
    from stable_baselines3 import PPO

    pk = dict(features_extractor_class=build_small_cnn(),
              features_extractor_kwargs=dict(features_dim=128), net_arch=[64, 64])
    return PPO("CnnPolicy", env, n_steps=2048, batch_size=256, n_epochs=4, gamma=0.99,
               ent_coef=0.01, learning_rate=3e-4, policy_kwargs=pk, seed=seed,
               verbose=0, device="cpu")


# --------------------------------------------------------------------------- evaluation

def eval_layouts(model, layouts, episodes: int, restrict: bool, deterministic: bool = False):
    """Per-layout success rate (success = episode reaches the goal, i.e. positive return)."""
    rates = []
    for lo in layouts:
        env = wrap(lo.size, lo.n, restrict)
        succ = 0
        for _ in range(episodes):
            obs, _ = env.reset(seed=lo.seed)
            done, total = False, 0.0
            while not done:
                a, _ = model.predict(obs, deterministic=deterministic)
                obs, r, term, trunc, _ = env.step(int(a))
                total += r
                done = term or trunc
            if total > 0:
                succ += 1
        env.close()
        rates.append(succ / episodes)
    return rates


# --------------------------------------------------------------------------- scoring

def _rank(values):
    """Return ranks (0=smallest) of a list, averaging ties."""
    order = np.argsort(np.argsort(values))
    return order.astype(float)


def _pool_hash(sizes, tiers, n_seeds, cap):
    key = json.dumps([sorted(sizes), sorted(tiers), n_seeds, cap])
    return hashlib.md5(key.encode()).hexdigest()[:10]


def oracle_scores(train_pool, args, restrict):
    """Empirical difficulty = mean over reference PPOs of (1 - success) per layout.

    Averaging over several reference seeds (and many eval episodes) denoises the per-layout
    label so a fitted geometry predictor can recover the true difficulty surface, and so the
    oracle CONDITION orders on signal rather than training luck. Cached (keyed by pool spec +
    ref-seeds + eval-episodes) so all curriculum seeds reuse one expensive build.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = _pool_hash(args.sizes, args.tiers, args.n_seeds, args.cap_per_cell)
    cache = CACHE_DIR / f"oracle_{h}_r{args.oracle_ref_seeds}e{args.oracle_eval_episodes}.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        return {tuple_key(k): v for k, v in d.items()}

    print(f"[oracle] building reference difficulty: {args.oracle_ref_seeds} ref seed(s) x "
          f"{args.oracle_steps} steps, {args.oracle_eval_episodes} eval episodes...", flush=True)
    acc = {lo.sig: [] for lo in train_pool}
    for rs in range(args.oracle_ref_seeds):
        env = CurriculumPoolEnv(restrict, rng_seed=12345 + rs)
        env.set_active(train_pool)
        model = make_ppo(env, seed=12345 + rs)
        model.learn(total_timesteps=args.oracle_steps, reset_num_timesteps=True)
        rates = eval_layouts(model, train_pool, args.oracle_eval_episodes, restrict)
        for lo, r in zip(train_pool, rates):
            acc[lo.sig].append(1.0 - r)
        print(f"[oracle]   ref seed {rs} done", flush=True)
    scores = {sig: float(np.mean(v)) for sig, v in acc.items()}
    cache.write_text(json.dumps({str_key(k): v for k, v in scores.items()}))
    print(f"[oracle] cached -> {cache.name}", flush=True)
    return scores


def str_key(sig):
    return json.dumps(sig)


def tuple_key(s):
    def to_tup(x):
        return tuple(to_tup(i) for i in x) if isinstance(x, list) else x
    return to_tup(json.loads(s))


def initial_scores(condition, train_pool, args, restrict):
    """Difficulty score per layout signature (higher = harder) for the chosen condition.

    Geometry/combined/adaptive use the cheap, policy-free, ORACLE-FREE `geometric_score`
    (rivers + size + detour, read straight off the grid). Adaptive conditions start from that
    and then refine via the live policy during training. Only the `oracle` condition runs
    reference PPOs; every other condition is oracle-free.
    """
    condition = ALIASES.get(condition, condition)
    if condition in ("random", "random_curriculum"):
        rng = np.random.default_rng(args.seed)
        return {lo.sig: float(rng.random()) for lo in train_pool}
    if condition == "rivers_only":
        return {lo.sig: float(lo.n) for lo in train_pool}
    if condition == "grader":
        calib = {tuple_key(k): v for k, v in json.loads((CACHE_DIR / GRADER_CALIB).read_text()).items()}
        calib_pool = enumerate_pool([7, 9, 11], [1, 2, 3], 80, 10)
        assert all(lo.sig in calib for lo in calib_pool), "calibration pool / labels mismatch"
        predict = fit_grader(calib_pool, calib, seed=0)
        return {lo.sig: float(p) for lo, p in zip(train_pool, predict(train_pool))}
    if condition in ("geometry", "adaptive_fine", "adaptive_coarse"):
        return {lo.sig: float(geometric_score(lo)) for lo in train_pool}
    if condition == "probe":
        return {lo.sig: random_walk_probe(lo, k=args.probe_k, seed=args.seed) for lo in train_pool}
    if condition == "combined":
        g = np.array([geometric_score(lo) for lo in train_pool])
        p = np.array([random_walk_probe(lo, k=args.probe_k, seed=args.seed) for lo in train_pool])
        comb = _rank(g) + _rank(p)
        return {lo.sig: float(c) for lo, c in zip(train_pool, comb)}
    if condition == "oracle":
        return oracle_scores(train_pool, args, restrict)
    raise ValueError(condition)


def adaptive_rescore(model, train_pool, args, restrict):
    """Re-score difficulty with the CURRENT policy: harder = lower current success.

    A tiny geometry term breaks the many ties when most layouts are still at 0 success early on.
    """
    rates = eval_layouts(model, train_pool, args.adapt_episodes, restrict)
    return {lo.sig: float((1.0 - r) + 1e-3 * geometric_score(lo))
            for lo, r in zip(train_pool, rates)}


def bins_from_scores(train_pool, scores, n_bins, tiebreak_seed=None):
    """Quantile bins by score. With tiebreak_seed, ties are broken by a seeded random key
    instead of Python's stable sort (which silently falls back to pool-generation order)."""
    if tiebreak_seed is None:
        return bin_layouts(train_pool, key=lambda lo: scores[lo.sig], n_bins=n_bins)
    rng = np.random.default_rng(10_007 + tiebreak_seed)
    tb = {lo.sig: float(rng.random()) for lo in train_pool}
    return bin_layouts(train_pool, key=lambda lo: (scores[lo.sig], tb[lo.sig]), n_bins=n_bins)


def bin_composition(bins):
    return [{f"s{k[0]}N{k[1]}": v for k, v in sorted(
        __import__("collections").Counter((lo.size, lo.n) for lo in b).items())} for b in bins]


def tie_fraction(scores):
    v = np.round(np.array(list(scores.values()), float), 9)
    _, counts = np.unique(v, return_counts=True)
    return float(counts[counts > 1].sum() / len(v))


# --------------------------------------------------------------------------- experiment

def run_condition(condition, args):
    import warnings

    warnings.filterwarnings("ignore")
    restrict = not args.no_restrict_actions
    t0 = time.time()

    # ---- pools: train + DISJOINT held-out (same cells, far-away seeds) ----
    train_pool = enumerate_pool(args.sizes, args.tiers, args.n_seeds, args.cap_per_cell)
    train_sigs = {lo.sig for lo in train_pool}
    eval_pool = enumerate_pool(args.sizes, args.tiers, args.eval_n_seeds,
                               args.eval_cap_per_cell, seed_start=args.eval_seed_start,
                               exclude_sigs=train_sigs)
    print(f"[{condition}] train={len(train_pool)} layouts  held-out={len(eval_pool)} layouts",
          flush=True)

    # ---- difficulty scores + bins ----
    scores = initial_scores(condition, train_pool, args, restrict)
    v2 = condition in ("flat", "size_order", "random_curriculum", "rivers_only", "grader")
    tb_seed = args.seed if v2 else None
    bins = bins_from_scores(train_pool, scores, args.n_bins, tb_seed)
    init_bins = bin_composition(bins)
    init_ties = tie_fraction(scores)
    unlocks = []

    # ---- env + model ----
    env = CurriculumPoolEnv(restrict, rng_seed=args.seed)
    is_curriculum = condition not in FLAT_CONDITIONS
    frontier = 0 if is_curriculum else args.n_bins - 1   # random unlocks everything
    active = [lo for b in bins[:frontier + 1] for lo in b]
    env.set_active(active)
    model = make_ppo(env, seed=args.seed)

    n_ck = max(1, args.total_steps // args.eval_every)
    print(f"[{condition}] {args.total_steps} steps, {n_ck} checkpoints, "
          f"{'curriculum' if is_curriculum else 'flat (no curriculum)'}\n", flush=True)

    curve = []
    steps_to_target = None
    frontier_start = 0  # env_steps at which the current frontier bin became active
    # a frontier is force-unlocked once it has been active this long (guarantees the curriculum
    # reaches the FULL distribution before the budget ends -> fair ordering test, not data-restriction)
    bin_deadline = int(args.bin_deadline_frac * args.total_steps)
    for ck in range(1, n_ck + 1):
        model.learn(total_timesteps=args.eval_every, reset_num_timesteps=(ck == 1))
        env_steps = ck * args.eval_every

        # adaptive re-scoring (fine: every checkpoint)
        if condition == "adaptive_fine":
            scores = adaptive_rescore(model, train_pool, args, restrict)
            bins = bins_from_scores(train_pool, scores, args.n_bins)

        # held-out success (headline) + per-cell breakdown (exposes the size-cliff: does the
        # curriculum crack the big grids that uniform sampling never reaches?)
        per_layout = eval_layouts(model, eval_pool, args.eval_episodes, restrict)
        held = float(np.mean(per_layout))
        _cell = {}
        for lo, r in zip(eval_pool, per_layout):
            _cell.setdefault((lo.size, lo.n), []).append(r)
        held_by_cell = {f"s{k[0]}N{k[1]}": round(float(np.mean(v)), 3)
                        for k, v in sorted(_cell.items())}

        # unlock next bin when EITHER the learner is competent on the active layouts (gate) OR the
        # current frontier has been active past its deadline (forced coverage).
        promoted = False
        unlock_kind = ""
        if is_curriculum and frontier < args.n_bins - 1:
            gate_set = active if len(active) <= args.gate_layouts else \
                list(np.random.default_rng(ck).choice(active, args.gate_layouts, replace=False))
            gate_succ = float(np.mean(eval_layouts(model, gate_set, args.gate_episodes, restrict)))
            competent = gate_succ >= args.gate
            overdue = (env_steps - frontier_start) >= bin_deadline
            if competent or overdue:
                frontier += 1
                frontier_start = env_steps
                promoted = True
                unlock_kind = "gate" if competent else "deadline"
                unlocks.append(dict(env_steps=env_steps, to_frontier=frontier, kind=unlock_kind,
                                    gate_success=round(gate_succ, 4)))
                if condition == "adaptive_coarse":
                    scores = adaptive_rescore(model, train_pool, args, restrict)
                    bins = bins_from_scores(train_pool, scores, args.n_bins)
        else:
            gate_succ = held

        # refresh active set (bins may have been re-scored, or frontier advanced)
        active = [lo for b in bins[:frontier + 1] for lo in b]
        env.set_active(active)

        if steps_to_target is None and held >= args.target:
            steps_to_target = env_steps

        wall = time.time() - t0
        curve.append(dict(env_steps=env_steps, held_success=round(held, 4),
                          gate_success=round(gate_succ, 4), frontier=frontier,
                          n_active=len(active), wall_s=round(wall, 1),
                          held_by_cell=held_by_cell))
        print(f"  steps={env_steps:>7d}  held={held:.2f}  gate={gate_succ:.2f}  "
              f"bins={frontier + 1}/{args.n_bins}  active={len(active):3d}"
              f"{('  [UNLOCK:' + unlock_kind + ']') if promoted else ''}  ({wall:.0f}s)", flush=True)

    final = curve[-1]["held_success"]
    result = dict(
        condition=condition, seed=args.seed, sizes=args.sizes, tiers=args.tiers,
        total_steps=args.total_steps, n_bins=args.n_bins, target=args.target,
        steps_to_target=steps_to_target, final_held_success=final,
        wall_s=round(time.time() - t0, 1), n_train=len(train_pool), n_eval=len(eval_pool),
        curve=curve,
        scoring_method=SCORING_METHOD.get(condition, condition),
        args=vars(args), code=_code_fingerprint(), initial_bins=init_bins,
        score_tie_fraction=init_ties, unlocks=unlocks,
    )

    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"{condition}_seed{args.seed}.json"
    out.write_text(json.dumps(result, indent=2))
    stt = f"{steps_to_target:,}" if steps_to_target is not None else "NOT REACHED"
    cellstr = "  ".join(f"{k}={v:.2f}" for k, v in curve[-1]["held_by_cell"].items())
    print(f"\n[{condition}] DONE  final held={final:.2f}  steps_to_{args.target}={stt}  "
          f"wall={result['wall_s']:.0f}s  -> {out.name}", flush=True)
    print(f"[{condition}] per-cell held: {cellstr}", flush=True)
    return result


def _code_fingerprint():
    import subprocess
    here = Path(__file__).resolve().parent
    fp = {f: hashlib.md5((here / f).read_bytes()).hexdigest() for f in
          ("curriculum_harness.py", "difficulty_scoring.py")}
    try:
        fp["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here,
                                                 text=True).strip()
    except Exception:
        fp["git_head"] = None
    return fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=CONDITIONS + ["all"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results-dir", default=None,
                    help="output folder (default: scripts/curriculum/results); use a NEW one per run family")
    # pool
    ap.add_argument("--sizes", type=int, nargs="+", default=[7, 9, 11])
    ap.add_argument("--tiers", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n-seeds", type=int, default=80)
    ap.add_argument("--cap-per-cell", type=int, default=10)
    ap.add_argument("--n-bins", type=int, default=3)
    # held-out eval pool
    ap.add_argument("--eval-seed-start", type=int, default=100_000)
    ap.add_argument("--eval-n-seeds", type=int, default=200)
    ap.add_argument("--eval-cap-per-cell", type=int, default=4)
    ap.add_argument("--eval-episodes", type=int, default=4)
    # curriculum gate
    ap.add_argument("--gate", type=float, default=0.55)
    ap.add_argument("--gate-episodes", type=int, default=4)
    ap.add_argument("--gate-layouts", type=int, default=18)
    ap.add_argument("--bin-deadline-frac", type=float, default=0.25,
                    help="force-unlock a bin after it has been active this fraction of total steps "
                         "(guarantees full-distribution coverage -> fair ordering test)")
    ap.add_argument("--target", type=float, default=0.30)
    # training budget
    ap.add_argument("--total-steps", type=int, default=600_000)
    ap.add_argument("--eval-every", type=int, default=50_000)
    ap.add_argument("--no-restrict-actions", action="store_true")
    # condition-specific
    ap.add_argument("--probe-k", type=int, default=8)
    ap.add_argument("--adapt-episodes", type=int, default=3)
    ap.add_argument("--oracle-steps", type=int, default=600_000)
    ap.add_argument("--oracle-ref-seeds", type=int, default=3)
    ap.add_argument("--oracle-eval-episodes", type=int, default=20)
    # build the shared oracle cache once (for parallel multi-seed orchestration), then exit
    ap.add_argument("--build-oracle-only", action="store_true")
    # smoke
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.sizes = [7, 9]
        args.tiers = [1, 2, 3]
        args.n_seeds, args.cap_per_cell = 40, 4
        args.total_steps, args.eval_every = 40_000, 10_000
        args.eval_n_seeds, args.eval_cap_per_cell, args.eval_episodes = 100, 2, 3
        args.oracle_steps, args.oracle_ref_seeds, args.oracle_eval_episodes = 40_000, 1, 6
        args.gate_layouts, args.gate_episodes = 8, 3

    if args.build_oracle_only:
        import warnings
        warnings.filterwarnings("ignore")
        restrict = not args.no_restrict_actions
        train_pool = enumerate_pool(args.sizes, args.tiers, args.n_seeds, args.cap_per_cell)
        oracle_scores(train_pool, args, restrict)
        print("[build-oracle-only] done", flush=True)
        return

    conditions = CONDITIONS if args.condition == "all" else [args.condition]
    for c in conditions:
        run_condition(c, args)


if __name__ == "__main__":
    main()
