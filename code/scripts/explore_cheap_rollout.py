#!/usr/bin/env python3
"""
Deep exploration of cheap rollout features for difficulty prediction.

Variations tested:
1. Scaling: How does quality improve as we sample more random programs?
2. Mini HC: Run a very short HC search (10-20 iters) as a feature
3. Smarter probes: Programs with conditionals, while loops with different conditions
4. Feature analysis: Which probe signals matter most?
5. Cost-accuracy tradeoff: What's the Pareto frontier?
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'leaps'))

from prog_policies.utils import get_env_name  # noqa: F401
from prog_policies.karel_tasks import get_task_cls
from prog_policies.karel import KarelDSL
from prog_policies.search_space import ProgrammaticSpace
from prog_policies.search_methods import HillClimbing
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
    prog = dsl_nodes.Program()
    prog.children = [body_node]
    return prog

def build_action(name):
    return dsl_nodes.Action(name)

def build_concat(*actions):
    if len(actions) == 1:
        return actions[0]
    result = dsl_nodes.Concatenate()
    result.children = [actions[0], build_concat(*actions[1:])]
    return result

def build_repeat(n, body):
    rep = dsl_nodes.Repeat()
    int_node = dsl_nodes.ConstInt(n)
    rep.children = [int_node, body]
    return rep

def build_while(cond_name, body, negate=False):
    w = dsl_nodes.While()
    cond = dsl_nodes.BoolFeature(cond_name)
    if negate:
        not_node = dsl_nodes.Not()
        not_node.children = [cond]
        w.children = [not_node, body]
    else:
        w.children = [cond, body]
    return w

def build_if(cond_name, body, negate=False):
    ite = dsl_nodes.If()
    cond = dsl_nodes.BoolFeature(cond_name)
    if negate:
        not_node = dsl_nodes.Not()
        not_node.children = [cond]
        ite.children = [not_node, body]
    else:
        ite.children = [cond, body]
    return ite

def build_ite(cond_name, if_body, else_body, negate=False):
    ite = dsl_nodes.ITE()
    cond = dsl_nodes.BoolFeature(cond_name)
    if negate:
        not_node = dsl_nodes.Not()
        not_node.children = [cond]
        ite.children = [not_node, if_body, else_body]
    else:
        ite.children = [cond, if_body, else_body]
    return ite


def get_extended_probes():
    """Extended set of probe programs testing various strategies."""
    probes = {}
    
    # === Basic movement ===
    probes['move_5'] = build_program(build_repeat(5, build_action('move')))
    probes['move_20'] = build_program(build_repeat(20, build_action('move')))
    probes['move_50'] = build_program(build_repeat(50, build_action('move')))
    
    # === Marker interaction ===
    probes['pick_20'] = build_program(build_repeat(20, build_action('pickMarker')))
    probes['put_20'] = build_program(build_repeat(20, build_action('putMarker')))
    probes['move_pick'] = build_program(build_repeat(15, build_concat(build_action('move'), build_action('pickMarker'))))
    probes['move_put'] = build_program(build_repeat(15, build_concat(build_action('move'), build_action('putMarker'))))
    
    # === Turning patterns ===
    probes['turn_move_L'] = build_program(build_repeat(20, build_concat(build_action('turnLeft'), build_action('move'))))
    probes['turn_move_R'] = build_program(build_repeat(20, build_concat(build_action('turnRight'), build_action('move'))))
    
    # === While-based navigation ===
    probes['while_front_move'] = build_program(build_while('frontIsClear', build_action('move')))
    probes['while_front_move_pick'] = build_program(
        build_while('frontIsClear', build_concat(build_action('move'), build_action('pickMarker')))
    )
    probes['while_front_move_put'] = build_program(
        build_while('frontIsClear', build_concat(build_action('move'), build_action('putMarker')))
    )
    
    # === Zigzag patterns (while + turn) ===
    probes['zigzag_L'] = build_program(build_repeat(5, build_concat(
        build_while('frontIsClear', build_action('move')), build_action('turnLeft')
    )))
    probes['zigzag_R'] = build_program(build_repeat(5, build_concat(
        build_while('frontIsClear', build_action('move')), build_action('turnRight')
    )))
    probes['zigzag_LR'] = build_program(build_repeat(5, build_concat(
        build_while('frontIsClear', build_action('move')),
        build_action('turnLeft'),
        build_while('frontIsClear', build_action('move')),
        build_action('turnRight')
    )))
    
    # === Conditional movement (if frontIsClear move else turn) ===
    probes['if_front_move_else_turnL'] = build_program(build_repeat(30,
        build_ite('frontIsClear', build_action('move'), build_action('turnLeft'))
    ))
    probes['if_front_move_else_turnR'] = build_program(build_repeat(30,
        build_ite('frontIsClear', build_action('move'), build_action('turnRight'))
    ))
    
    # === Marker-conditional programs ===
    probes['if_marker_pick_else_move'] = build_program(build_repeat(30,
        build_ite('markersPresent', build_action('pickMarker'), build_action('move'))
    ))
    probes['while_marker_pick'] = build_program(
        build_while('markersPresent', build_action('pickMarker'))
    )
    
    # === Wall-following (left-hand rule approximation) ===
    # Repeat: if(frontIsClear) move else turnRight
    probes['wall_follow_R'] = build_program(build_repeat(50,
        build_ite('frontIsClear', build_action('move'), build_action('turnRight'))
    ))
    # Repeat: if(leftIsClear) { turnLeft; move } else { if(frontIsClear) move else turnRight }
    probes['wall_follow_L'] = build_program(build_repeat(50,
        build_ite('leftIsClear',
            build_concat(build_action('turnLeft'), build_action('move')),
            build_ite('frontIsClear', build_action('move'), build_action('turnRight'))
        )
    ))
    
    # === Exploration (move + pick + turn combinations) ===
    probes['explore_pick'] = build_program(build_repeat(20,
        build_concat(
            build_ite('frontIsClear', build_action('move'), build_action('turnLeft')),
            build_action('pickMarker')
        )
    ))
    probes['explore_put'] = build_program(build_repeat(20,
        build_concat(
            build_ite('frontIsClear', build_action('move'), build_action('turnLeft')),
            build_action('putMarker')
        )
    ))
    
    # === Spiral (turn same direction periodically) ===
    probes['spiral_L'] = build_program(build_repeat(10, build_concat(
        build_repeat(3, build_action('move')),
        build_action('turnLeft')
    )))
    probes['spiral_R'] = build_program(build_repeat(10, build_concat(
        build_repeat(3, build_action('move')),
        build_action('turnRight')
    )))
    
    return probes


def evaluate_probe(task, program):
    """Evaluate a probe, return reward, crashed, steps."""
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


def run_mini_hc(task_name, seed, n_iterations=10):
    """Run a very short HC search. Returns best reward found."""
    task_cls = get_task_cls(task_name)
    env_args = get_env_args(task_name)
    task_envs = [task_cls(env_args, seed)]
    
    dsl = KarelDSL()
    search_space = ProgrammaticSpace(dsl, sigma=0.1)
    search_space.set_seed(seed)
    
    search_method = HillClimbing(k=250, e=2)
    progs, rewards = search_method.search(search_space, task_envs, seed=seed, n_iterations=n_iterations)
    
    return max(rewards) if rewards else 0.0


def evaluate_random_programs(task, search_space, n_programs, seed):
    """Sample random programs and collect detailed stats."""
    search_space.set_seed(seed + 10000)
    rewards = []
    crashes = 0
    steps_list = []
    
    for _ in range(n_programs):
        _, prog = search_space.initialize_individual()
        reward, crashed, steps = evaluate_probe(task, prog)
        rewards.append(reward)
        steps_list.append(steps)
        if crashed:
            crashes += 1
    
    rewards = np.array(rewards)
    steps_arr = np.array(steps_list)
    
    feats = {
        'mean_reward': np.mean(rewards),
        'max_reward': np.max(rewards),
        'median_reward': np.median(rewards),
        'std_reward': np.std(rewards),
        'q75_reward': np.percentile(rewards, 75),
        'q90_reward': np.percentile(rewards, 90),
        'pct_positive': float(np.mean(rewards > 0)),
        'pct_above_half': float(np.mean(rewards > 0.5)),
        'pct_perfect': float(np.mean(rewards >= 1.0)),
        'pct_negative': float(np.mean(rewards < 0)),
        'crash_rate': crashes / n_programs,
        'mean_steps': np.mean(steps_arr),
        'std_steps': np.std(steps_arr),
    }
    return feats, rewards


def main():
    # Load existing dataset for ground truth
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'multiseed_dataset_large.csv')
    df = pd.read_csv(dataset_path)
    print('Loaded {} samples'.format(len(df)))
    
    dsl = KarelDSL()
    
    # ================================================================
    # PART 1: Extended probes
    # ================================================================
    print('\n' + '='*70)
    print('PART 1: Extended Probe Programs (25 probes)')
    print('='*70)
    
    probes = get_extended_probes()
    print('  {} probe programs defined'.format(len(probes)))
    
    probe_results = defaultdict(list)  # probe_name -> list of rewards
    t0 = time.time()
    
    for idx, row in df.iterrows():
        task_cls = get_task_cls(row['task_name'])
        env_args = get_env_args(row['task_name'])
        task = task_cls(env_args, int(row['seed']))
        
        for probe_name, prog in probes.items():
            reward, crashed, steps = evaluate_probe(task, prog)
            probe_results[probe_name].append(reward)
    
    elapsed = time.time() - t0
    print('  Probe evaluation: {:.1f}s for {} instances'.format(elapsed, len(df)))
    
    # Correlation of each probe with difficulty
    y = df['difficulty'].values
    print('\n  Per-probe Spearman correlation with difficulty:')
    print('  {:<35} {:>10} {:>10}'.format('Probe', 'Spearman', 'p-value'))
    print('  ' + '-' * 58)
    
    probe_correlations = []
    for probe_name in sorted(probes.keys()):
        rewards = np.array(probe_results[probe_name])
        if np.std(rewards) > 1e-10:
            sr, p = spearmanr(y, rewards)
            probe_correlations.append((probe_name, sr, p))
    
    probe_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, sr, p in probe_correlations:
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print('  {:<35} {:>10.4f} {:>10.4f} {}'.format(name, sr, p, sig))
    
    # ================================================================
    # PART 2: Scaling analysis — how many random programs do we need?
    # ================================================================
    print('\n' + '='*70)
    print('PART 2: Scaling Analysis (N random programs vs quality)')
    print('='*70)
    
    # Generate 200 random programs for each instance, then subsample
    max_n = 200
    all_random_rewards = []  # shape: [n_instances, max_n]
    
    t0 = time.time()
    for idx, row in df.iterrows():
        task_cls = get_task_cls(row['task_name'])
        env_args = get_env_args(row['task_name'])
        task = task_cls(env_args, int(row['seed']))
        search_space = ProgrammaticSpace(dsl, sigma=0.1)
        search_space.set_seed(int(row['seed']) + 10000)
        
        instance_rewards = []
        for _ in range(max_n):
            _, prog = search_space.initialize_individual()
            reward, _, _ = evaluate_probe(task, prog)
            instance_rewards.append(reward)
        all_random_rewards.append(instance_rewards)
        
        if (idx + 1) % 100 == 0:
            print('  {}/{} done ({:.1f}s)'.format(idx + 1, len(df), time.time() - t0))
    
    all_random_rewards = np.array(all_random_rewards)  # [550, 200]
    elapsed = time.time() - t0
    print('  Total: {:.1f}s for {} x {} evaluations'.format(elapsed, len(df), max_n))
    
    # Now compute features at different N and measure correlation
    print('\n  N_random → Spearman ρ (using max_reward as feature):')
    print('  {:<8} {:>10} {:>12} {:>12} {:>12}'.format('N', 'max_rwd', 'mean_rwd', 'pct_pos', 'LOTO_max'))
    print('  ' + '-' * 58)
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    
    tasks = df['task_name'].unique()
    
    for n in [5, 10, 20, 50, 100, 150, 200]:
        sub_rewards = all_random_rewards[:, :n]
        max_r = sub_rewards.max(axis=1)
        mean_r = sub_rewards.mean(axis=1)
        pct_pos = (sub_rewards > 0).mean(axis=1)
        
        sr_max, _ = spearmanr(y, max_r)
        sr_mean, _ = spearmanr(y, mean_r)
        sr_pct, _ = spearmanr(y, pct_pos)
        
        # Quick LOTO with just max_reward
        all_true, all_pred = [], []
        for held_out in tasks:
            train_mask = df['task_name'] != held_out
            test_mask = df['task_name'] == held_out
            X_tr = max_r[train_mask].reshape(-1, 1)
            X_te = max_r[test_mask].reshape(-1, 1)
            y_tr = y[train_mask]
            y_te = y[test_mask]
            from sklearn.linear_model import Ridge
            m = Ridge(alpha=1.0)
            m.fit(X_tr, y_tr)
            all_true.extend(y_te)
            all_pred.extend(m.predict(X_te))
        sr_loto, _ = spearmanr(all_true, all_pred)
        
        print('  {:<8} {:>10.4f} {:>12.4f} {:>12.4f} {:>12.4f}'.format(n, sr_max, sr_mean, sr_pct, sr_loto))
    
    # ================================================================
    # PART 3: Mini HC as a feature
    # ================================================================
    print('\n' + '='*70)
    print('PART 3: Mini HC (varying budget)')
    print('='*70)
    print('How good is a very short HC search as a predictor?')
    
    for n_iter in [5, 10, 20, 50]:
        t0 = time.time()
        mini_hc_rewards = []
        for idx, row in df.iterrows():
            r = run_mini_hc(row['task_name'], int(row['seed']), n_iterations=n_iter)
            mini_hc_rewards.append(r)
        elapsed = time.time() - t0
        
        mini_hc_rewards = np.array(mini_hc_rewards)
        sr, _ = spearmanr(y, mini_hc_rewards)
        
        # LOTO
        all_true, all_pred = [], []
        for held_out in tasks:
            train_mask = df['task_name'] != held_out
            test_mask = df['task_name'] == held_out
            X_tr = mini_hc_rewards[train_mask].reshape(-1, 1)
            X_te = mini_hc_rewards[test_mask].reshape(-1, 1)
            y_tr = y[train_mask]
            y_te = y[test_mask]
            m = Ridge(alpha=1.0)
            m.fit(X_tr, y_tr)
            all_true.extend(y_te)
            all_pred.extend(m.predict(X_te))
        sr_loto, _ = spearmanr(all_true, all_pred)
        
        cost_per_instance = elapsed / len(df)
        print('  HC(n_iter={:<3}): Spearman={:.4f}, LOTO={:.4f}, cost={:.3f}s/instance, total={:.1f}s'.format(
            n_iter, sr, sr_loto, cost_per_instance, elapsed))
    
    # ================================================================
    # PART 4: Combined model — probes + random + mini HC
    # ================================================================
    print('\n' + '='*70)
    print('PART 4: Combined Model (probes + 200 random + mini HC)')
    print('='*70)
    
    # Build feature matrix
    # Probes
    probe_df = pd.DataFrame({name: probe_results[name] for name in probes.keys()})
    probe_df.columns = ['probe_' + c for c in probe_df.columns]
    
    # Random stats at N=200
    random_feats = pd.DataFrame({
        'rand_max': all_random_rewards.max(axis=1),
        'rand_mean': all_random_rewards.mean(axis=1),
        'rand_median': np.median(all_random_rewards, axis=1),
        'rand_std': all_random_rewards.std(axis=1),
        'rand_q90': np.percentile(all_random_rewards, 90, axis=1),
        'rand_pct_pos': (all_random_rewards > 0).mean(axis=1),
        'rand_pct_neg': (all_random_rewards < 0).mean(axis=1),
        'rand_pct_perfect': (all_random_rewards >= 1.0).mean(axis=1),
    })
    
    # Mini HC (n=10 is fast)
    mini_hc_10 = []
    for idx, row in df.iterrows():
        r = run_mini_hc(row['task_name'], int(row['seed']), n_iterations=10)
        mini_hc_10.append(r)
    mini_hc_df = pd.DataFrame({'mini_hc_10': mini_hc_10})
    
    # Combine all
    all_features = pd.concat([probe_df, random_feats, mini_hc_df], axis=1)
    feat_cols = all_features.columns.tolist()
    
    print('  Total features: {}'.format(len(feat_cols)))
    
    # Standard split evaluation
    np.random.seed(42)
    train_idx, test_idx = [], []
    for task, group in df.groupby('task_name'):
        indices = group.index.values.copy()
        np.random.shuffle(indices)
        n_train = max(1, int(len(indices) * 0.75))
        train_idx.extend(indices[:n_train])
        test_idx.extend(indices[n_train:])
    
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(all_features.loc[train_idx].values)
    X_test = scaler.transform(all_features.loc[test_idx].values)
    
    for name, model in [
        ('Ridge', Ridge(alpha=1.0)),
        ('RF(200,d5)', RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)),
        ('RF(300,d7)', RandomForestRegressor(n_estimators=300, max_depth=7, random_state=42)),
    ]:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        sr, _ = spearmanr(y_test, pred)
        print('  Within-distrib {:<15} Spearman = {:.4f}'.format(name, sr))
    
    # LOTO
    print()
    for name, model_cls_args in [
        ('Ridge', (Ridge, {'alpha': 1.0})),
        ('RF(200,d5)', (RandomForestRegressor, {'n_estimators': 200, 'max_depth': 5, 'random_state': 42})),
    ]:
        all_true, all_pred = [], []
        for held_out in tasks:
            train_mask = df['task_name'] != held_out
            test_mask = df['task_name'] == held_out
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(all_features[train_mask].values)
            X_te = scaler.transform(all_features[test_mask].values)
            y_tr = y[train_mask]
            y_te = y[test_mask]
            cls, kwargs = model_cls_args
            m = cls(**kwargs)
            m.fit(X_tr, y_tr)
            all_true.extend(y_te)
            all_pred.extend(m.predict(X_te))
        sr, _ = spearmanr(all_true, all_pred)
        print('  LOTO {:<20} Spearman = {:.4f}'.format(name, sr))
    
    # Feature importance
    model_rf = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    scaler = StandardScaler()
    X_all = scaler.fit_transform(all_features.values)
    model_rf.fit(X_all, y)
    importances = model_rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    print('\n  Top 15 feature importances:')
    for i in top_idx:
        print('    {:<35} {:.4f}'.format(feat_cols[i], importances[i]))
    
    # ================================================================
    # PART 5: Cost-Accuracy Pareto Frontier
    # ================================================================
    print('\n' + '='*70)
    print('PART 5: Cost-Accuracy Tradeoff')
    print('='*70)
    print('  {:<40} {:>10} {:>12} {:>10}'.format('Method', 'Spearman', 'LOTO', 'Cost(s)'))
    print('  ' + '-' * 75)
    
    # We already computed these — gather results
    # 1. Just max of N=10 random
    r10 = all_random_rewards[:, :10].max(axis=1)
    sr10, _ = spearmanr(y, r10)
    # cost: 10 evals * ~0.001s = 0.01s
    
    r50 = all_random_rewards[:, :50].max(axis=1)
    sr50, _ = spearmanr(y, r50)
    
    r200 = all_random_rewards[:, :200].max(axis=1)
    sr200, _ = spearmanr(y, r200)
    
    # Best probe
    best_probe = probe_df.max(axis=1).values
    sr_bp, _ = spearmanr(y, best_probe)
    
    # Probes + random(50) model
    feat_small = pd.concat([probe_df, pd.DataFrame({
        'rand_max_50': all_random_rewards[:, :50].max(axis=1),
        'rand_mean_50': all_random_rewards[:, :50].mean(axis=1),
        'rand_pct_pos_50': (all_random_rewards[:, :50] > 0).mean(axis=1),
    })], axis=1)
    
    # LOTO for feat_small
    all_true_s, all_pred_s = [], []
    for held_out in tasks:
        train_mask = df['task_name'] != held_out
        test_mask = df['task_name'] == held_out
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(feat_small[train_mask].values)
        X_te = scaler.transform(feat_small[test_mask].values)
        y_tr = y[train_mask]
        y_te = y[test_mask]
        m = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
        m.fit(X_tr, y_tr)
        all_true_s.extend(y_te)
        all_pred_s.extend(m.predict(X_te))
    sr_small_loto, _ = spearmanr(all_true_s, all_pred_s)
    
    # Simple within-distrib for all
    results_summary = [
        ('max(10 random)', sr10, '-', '~0.01'),
        ('max(50 random)', sr50, '-', '~0.03'),
        ('max(200 random)', sr200, '-', '~0.12'),
        ('best probe (25 probes)', sr_bp, '-', '~0.01'),
        ('probes + 50 random (RF, LOTO)', '-', '{:.4f}'.format(sr_small_loto), '~0.04'),
    ]
    
    for name, sr_val, loto_val, cost in results_summary:
        sr_str = '{:.4f}'.format(sr_val) if isinstance(sr_val, float) else sr_val
        print('  {:<40} {:>10} {:>12} {:>10}'.format(name, sr_str, loto_val, cost))


if __name__ == '__main__':
    main()
