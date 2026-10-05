#!/usr/bin/env python3
"""
Cheap Rollout Features for difficulty prediction.

IDEA: Instead of just looking at grid structure, run a few simple/random programs
and observe the reward. This captures the interaction between grid layout AND
reward function — which is what structural features miss.

Simple programs ("probes"):
  - "just move forward"            → does forward motion help at all?
  - "move + pickMarker loop"       → does collecting markers give reward?
  - "move + putMarker loop"        → does placing markers give reward?
  - "turn + move random walk"      → what does random exploration achieve?
  - "while(frontIsClear) move"     → how far can you go in a line?
  - Random programs from ProgrammaticSpace (sample N, take stats)

For each probe, record: reward, crashed?, steps taken
Then use these as features for difficulty prediction.

This is ~100x cheaper than HC search (seconds vs minutes per instance).
"""

import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'leaps'))

# Resolve circular import
from prog_policies.utils import get_env_name  # noqa: F401
from prog_policies.karel_tasks import get_task_cls
from prog_policies.karel import KarelDSL
from prog_policies.search_space import ProgrammaticSpace
from prog_policies.base import dsl_nodes


def get_env_args(task_name):
    env_args = {
        "env_height": 8, "env_width": 8,
        "crashable": False, "leaps_behaviour": True, "max_calls": 10000,
    }
    if task_name in ("StairClimberSparse", "TopOff", "FourCorners"):
        env_args["env_height"] = 12
        env_args["env_width"] = 12
    elif task_name == "CleanHouse":
        env_args["env_height"] = 14
        env_args["env_width"] = 22
    elif task_name == "WallAvoider":
        env_args["env_height"] = 8
        env_args["env_width"] = 5
    return env_args


def build_program(body_node):
    """Wrap a statement node in a Program."""
    prog = dsl_nodes.Program()
    prog.children = [body_node]
    return prog


def build_action(name):
    return dsl_nodes.Action(name)


def build_concat(*actions):
    """Build a left-nested concatenation of actions."""
    if len(actions) == 1:
        return actions[0]
    result = dsl_nodes.Concatenate()
    result.children = [actions[0], build_concat(*actions[1:])]
    return result


def build_repeat(n, body):
    """Repeat body n times."""
    rep = dsl_nodes.Repeat()
    int_node = dsl_nodes.ConstInt(n)
    rep.children = [int_node, body]
    return rep


def build_while_front_clear(body):
    """while(frontIsClear) { body }"""
    w = dsl_nodes.While()
    cond = dsl_nodes.BoolFeature('frontIsClear')
    w.children = [cond, body]
    return w


def get_probe_programs():
    """Build a set of simple "probe" programs that test different aspects."""
    probes = {}
    
    # 1. Just move forward 20 times
    probes['move_20'] = build_program(build_repeat(20, build_action('move')))
    
    # 2. Move + pickMarker repeated
    probes['move_pick_10'] = build_program(
        build_repeat(10, build_concat(build_action('move'), build_action('pickMarker')))
    )
    
    # 3. Move + putMarker repeated
    probes['move_put_10'] = build_program(
        build_repeat(10, build_concat(build_action('move'), build_action('putMarker')))
    )
    
    # 4. Turn left + move (spiral-ish)
    probes['turn_move_20'] = build_program(
        build_repeat(20, build_concat(build_action('turnLeft'), build_action('move')))
    )
    
    # 5. while(frontIsClear) move — go as far as possible in one direction
    probes['while_front_move'] = build_program(
        build_while_front_clear(build_action('move'))
    )
    
    # 6. while(frontIsClear) { move; pickMarker }
    probes['while_front_move_pick'] = build_program(
        build_while_front_clear(
            build_concat(build_action('move'), build_action('pickMarker'))
        )
    )
    
    # 7. Repeat 5: { while(frontIsClear) move; turnLeft }
    probes['zigzag'] = build_program(
        build_repeat(5, build_concat(
            build_while_front_clear(build_action('move')),
            build_action('turnLeft')
        ))
    )
    
    # 8. Repeat 5: { while(frontIsClear) move; turnRight }
    probes['zigzag_right'] = build_program(
        build_repeat(5, build_concat(
            build_while_front_clear(build_action('move')),
            build_action('turnRight')
        ))
    )
    
    # 9. Pick marker 20 times (in place)
    probes['pick_20'] = build_program(build_repeat(20, build_action('pickMarker')))
    
    # 10. Put marker 20 times (in place)
    probes['put_20'] = build_program(build_repeat(20, build_action('putMarker')))
    
    return probes


def evaluate_probe(task, program):
    """Evaluate a probe program, return reward."""
    task.reset_environment()
    reward = 0.
    step_count = 0
    crashed = False
    for _ in program.run_generator(task.environment):
        terminated, instant_reward = task.get_reward(task.environment)
        reward += instant_reward
        step_count += 1
        if terminated or task.environment.is_crashed():
            crashed = task.environment.is_crashed()
            break
    return reward, crashed, step_count


def evaluate_random_programs(task, search_space, n_programs=50, seed=0):
    """Sample n random programs and evaluate them. Return stats."""
    search_space.set_seed(seed + 1000)  # offset to avoid correlation with env seed
    rewards = []
    crashes = 0
    for _ in range(n_programs):
        _, prog = search_space.initialize_individual()
        r, crashed, _ = evaluate_probe(task, prog)
        rewards.append(r)
        if crashed:
            crashes += 1
    
    rewards = np.array(rewards)
    return {
        'random_mean_reward': np.mean(rewards),
        'random_max_reward': np.max(rewards),
        'random_median_reward': np.median(rewards),
        'random_std_reward': np.std(rewards),
        'random_pct_positive': float(np.mean(rewards > 0)),
        'random_pct_perfect': float(np.mean(rewards >= 1.0)),
        'random_crash_rate': crashes / n_programs,
    }


def extract_rollout_features(task_name, seed, n_random=50):
    """Extract cheap rollout features for one (task, seed) instance."""
    task_cls = get_task_cls(task_name)
    env_args = get_env_args(task_name)
    task = task_cls(env_args, seed)
    
    dsl = KarelDSL()
    search_space = ProgrammaticSpace(dsl, sigma=0.1)
    
    features = {}
    
    # Run probe programs
    probes = get_probe_programs()
    for probe_name, prog in probes.items():
        task.reset_environment()
        reward, crashed, steps = evaluate_probe(task, prog)
        features['probe_{}_reward'.format(probe_name)] = reward
        features['probe_{}_crashed'.format(probe_name)] = float(crashed)
        features['probe_{}_steps'.format(probe_name)] = float(steps)
    
    # Run random programs
    random_feats = evaluate_random_programs(task, search_space, n_programs=n_random, seed=seed)
    features.update(random_feats)
    
    # Derived: best probe reward
    probe_rewards = [v for k, v in features.items() if k.endswith('_reward') and 'random' not in k]
    features['best_probe_reward'] = max(probe_rewards) if probe_rewards else 0.0
    features['any_probe_positive'] = float(any(r > 0 for r in probe_rewards))
    
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='multiseed_dataset_large.csv',
                       help='Existing dataset to augment with rollout features')
    parser.add_argument('--n-random', type=int, default=50,
                       help='Number of random programs to sample per instance')
    parser.add_argument('--output', default='multiseed_with_rollout.csv')
    args = parser.parse_args()
    
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.dataset)
    df = pd.read_csv(dataset_path)
    print('Loaded {} samples'.format(len(df)))
    
    # Extract rollout features for each instance
    all_rollout_features = []
    t0 = time.time()
    
    for idx, row in df.iterrows():
        task_name = row['task_name']
        seed = int(row['seed'])
        
        feats = extract_rollout_features(task_name, seed, n_random=args.n_random)
        all_rollout_features.append(feats)
        
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = elapsed / (idx + 1)
            remaining = rate * (len(df) - idx - 1)
            print('  {}/{} done ({:.1f}s elapsed, ~{:.0f}s remaining)'.format(
                idx + 1, len(df), elapsed, remaining))
    
    # Merge with original dataset
    rollout_df = pd.DataFrame(all_rollout_features)
    combined = pd.concat([df.reset_index(drop=True), rollout_df], axis=1)
    
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.output)
    combined.to_csv(output_path, index=False)
    print('\nSaved {} with {} new rollout features'.format(output_path, len(rollout_df.columns)))
    
    # Quick evaluation
    print('\n--- Quick Correlation Check ---')
    rollout_cols = rollout_df.columns.tolist()
    y = df['difficulty'].values
    
    print('{:<35} {:>12}'.format('Feature', 'Spearman'))
    print('-' * 50)
    correlations = []
    for col in sorted(rollout_cols):
        x = rollout_df[col].values
        if np.std(x) > 1e-10:
            sr, p = spearmanr(y, x)
            correlations.append((col, sr, p))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for col, sr, p in correlations[:20]:
        sig = '*' if p < 0.05 else ' '
        print('{:<35} {:>10.4f} {}'.format(col, sr, sig))
    
    # Overall: use rollout features alone
    print('\n--- Model: Rollout features → difficulty ---')
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    
    np.random.seed(42)
    train_idx, test_idx = [], []
    for task, group in df.groupby('task_name'):
        indices = group.index.values.copy()
        np.random.shuffle(indices)
        n_train = max(1, int(len(indices) * 0.75))
        train_idx.extend(indices[:n_train])
        test_idx.extend(indices[n_train:])
    
    y_train = df.loc[train_idx, 'difficulty'].values
    y_test = df.loc[test_idx, 'difficulty'].values
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(rollout_df.loc[train_idx].values)
    X_test = scaler.transform(rollout_df.loc[test_idx].values)
    
    for name, model in [
        ('Ridge', Ridge(alpha=1.0)),
        ('GBR(100,d3)', GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)),
        ('RF(200,d5)', RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)),
    ]:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        sr, _ = spearmanr(y_test, pred)
        r2 = r2_score(y_test, pred)
        print('  {:<20} Spearman={:.4f}  R2={:.4f}'.format(name, sr, r2))
    
    # LOTO evaluation
    print('\n--- LOTO: Can rollout features generalize to new tasks? ---')
    tasks = df['task_name'].unique()
    all_true, all_pred = [], []
    for held_out in tasks:
        train_mask = df['task_name'] != held_out
        test_mask = df['task_name'] == held_out
        X_tr = scaler.fit_transform(rollout_df[train_mask].values)
        X_te = scaler.transform(rollout_df[test_mask].values)
        y_tr = df.loc[train_mask, 'difficulty'].values
        y_te = df.loc[test_mask, 'difficulty'].values
        m = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        m.fit(X_tr, y_tr)
        all_true.extend(y_te)
        all_pred.extend(m.predict(X_te))
    
    sr_loto, _ = spearmanr(all_true, all_pred)
    print('  LOTO Pooled Spearman = {:.4f}'.format(sr_loto))
    print('  (Compare: structural features LOTO = -0.45, task-mean = 0.48)')


if __name__ == '__main__':
    main()
