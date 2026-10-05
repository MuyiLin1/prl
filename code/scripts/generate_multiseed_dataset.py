#!/usr/bin/env python3
"""
Generate a multi-seed dataset for training difficulty predictors.

For each (task, seed) pair:
  1. Instantiate the environment
  2. Extract features: raw grid flattened + hand-crafted features
  3. Run a short HC search to get ground-truth reward (difficulty label)

Outputs: multiseed_dataset.csv with columns:
  task_name, seed, hc_best_reward, difficulty, <46 hand-crafted features>, <flattened grid>
"""

import sys
import os
import csv
import time
import argparse
import numpy as np
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'leaps'))

# Import order matters: utils must be imported before karel_tasks to avoid circular import
from prog_policies.utils import get_env_name  # noqa: F401 - resolves circular import
from prog_policies.karel_tasks import get_task_cls
from prog_policies.karel import KarelDSL
from prog_policies.search_space import ProgrammaticSpace
from prog_policies.search_methods import HillClimbing


# ---- Feature extraction (adapted from karel_curriculum_difficulty_predictor.py) ----

def build_grid_graph(free_cells):
    free_set = set(map(tuple, free_cells))
    graph = {tuple(c): [] for c in free_cells}
    for y, x in free_cells:
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            nb = (y+dy, x+dx)
            if nb in free_set:
                graph[(y,x)].append(nb)
    return graph


def bfs_distances(graph, start):
    if start not in graph:
        return {}
    dist = {start: 0}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in graph[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def extract_features_from_state(state, task_name):
    """Extract hand-crafted features from a Karel environment state tensor."""
    # state shape: [16, H, W]
    H, W = state.shape[1], state.shape[2]
    walls = state[4]  # boolean wall channel
    
    # Find free cells
    free_mask = ~walls
    free_cells = list(zip(*np.where(free_mask)))
    wall_cells = list(zip(*np.where(walls)))
    
    n_free = len(free_cells)
    n_walls = len(wall_cells)
    area = H * W
    
    if n_free == 0:
        # Degenerate case
        return None
    
    # Build graph
    graph = build_grid_graph(free_cells)
    
    # Degrees
    degrees = np.array([len(graph[c]) for c in graph], dtype=float)
    
    # Connected components
    seen = set()
    comps = []
    for node in graph:
        if node in seen:
            continue
        dist = bfs_distances(graph, node)
        comp = list(dist.keys())
        seen.update(comp)
        comps.append(comp)
    
    largest_comp = max(len(c) for c in comps)
    
    # Diameter and mean pair distance
    diameter = 0
    mean_pair_accum = 0.0
    mean_pair_count = 0
    for node in graph:
        d = bfs_distances(graph, node)
        if d:
            diameter = max(diameter, max(d.values()))
            mean_pair_accum += sum(d.values())
            mean_pair_count += len(d)
    mean_pair_dist = mean_pair_accum / max(1, mean_pair_count)
    
    # Agent position (from direction channels 0-3)
    agent_dirs = state[0:4]
    agent_positions = np.where(agent_dirs.any(axis=0))
    if len(agent_positions[0]) > 0:
        agent_y, agent_x = int(agent_positions[0][0]), int(agent_positions[1][0])
        start = (agent_y, agent_x)
        start_dist = bfs_distances(graph, start)
        start_reachable = len(start_dist)
        start_ecc = max(start_dist.values()) if start_dist else 0
    else:
        start_reachable = n_free
        start_ecc = diameter
    
    # Spectral features
    nodes = list(graph.keys())
    n = len(nodes)
    k = 6
    if n > 1:
        idx = {node: i for i, node in enumerate(nodes)}
        A = np.zeros((n, n), dtype=float)
        for u, nbrs in graph.items():
            for v in nbrs:
                A[idx[u], idx[v]] = 1.0
        deg = A.sum(axis=1)
        D_inv_sqrt = np.zeros_like(deg)
        mask = deg > 0
        D_inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])
        L = np.eye(n) - (D_inv_sqrt[:, None] * A * D_inv_sqrt[None, :])
        vals = np.sort(np.real(np.linalg.eigvalsh(L)))[:k]
        if len(vals) < k:
            vals = np.pad(vals, (0, k - len(vals)))
        spectral = [float(x) for x in vals]
    else:
        spectral = [0.0] * k
    
    # Marker features (from state channels 5+)
    markers = state[5:]  # channels 5-15 represent marker counts
    has_markers = markers.any(axis=0).sum()
    
    # Reward-type heuristics based on task name
    tn = task_name.lower()
    reward_pick = float('pick' in tn or 'harvest' in tn or 'clean' in tn or 'path' in tn)
    reward_put = float('put' in tn or 'seed' in tn or 'top' in tn or 'corner' in tn or 'wall' in tn)
    reward_goal = float('maze' in tn or 'stair' in tn or 'door' in tn or 'snake' in tn)
    reward_visit = float('stroke' in tn or 'snake' in tn)
    reward_sparse = float('sparse' in tn or 'maze' in tn or 'stair' in tn or 'snake' in tn)
    reward_crash = float('clean' in tn or 'stair' in tn or 'stroke' in tn or 'snake' in tn or 
                         'wall' in tn or 'path' in tn or 'top' in tn)
    
    features = {
        'grid_height': float(H),
        'grid_width': float(W),
        'grid_area': float(area),
        'graph_num_free_cells': float(n_free),
        'graph_num_wall_cells': float(n_walls),
        'graph_wall_fraction': float(n_walls / area),
        'graph_num_edges': float(sum(len(v) for v in graph.values()) / 2.0),
        'graph_num_components': float(len(comps)),
        'graph_largest_component_frac': float(largest_comp / max(1, n_free)),
        'graph_mean_degree': float(np.mean(degrees)) if len(degrees) > 0 else 0.0,
        'graph_min_degree': float(np.min(degrees)) if len(degrees) > 0 else 0.0,
        'graph_max_degree': float(np.max(degrees)) if len(degrees) > 0 else 0.0,
        'graph_num_deadends': float(np.sum(degrees <= 1)),
        'graph_num_corridor_cells': float(np.sum(degrees == 2)),
        'graph_num_branch_cells': float(np.sum(degrees >= 3)),
        'graph_diameter': float(diameter),
        'graph_mean_pair_distance': float(mean_pair_dist),
        'graph_start_reachable_frac': float(start_reachable / max(1, n_free)),
        'graph_start_eccentricity': float(start_ecc),
        'graph_num_markers': float(has_markers),
        'spectral_lambda_0': spectral[0],
        'spectral_lambda_1': spectral[1],
        'spectral_lambda_2': spectral[2],
        'spectral_lambda_3': spectral[3],
        'spectral_lambda_4': spectral[4],
        'spectral_lambda_5': spectral[5],
        'reward_pick_markers': reward_pick,
        'reward_put_markers': reward_put,
        'reward_reach_goal': reward_goal,
        'reward_visit_cells': reward_visit,
        'reward_sparse': reward_sparse,
        'reward_crash_penalty_used': reward_crash,
    }
    return features


def get_env_args(task_name):
    """Get env_args for a task (mirrors scripts/main.py defaults)."""
    env_args = {
        "env_height": 8,
        "env_width": 8,
        "crashable": False,
        "leaps_behaviour": True,
        "max_calls": 10000,
    }
    if task_name in (
        "StairClimber", "StairClimberSparse",
        "TopOff", "TopOffSparse",
        "FourCorners", "FourCornersSparse",
    ):
        env_args["env_height"] = 12
        env_args["env_width"] = 12
    elif task_name in ("CleanHouse", "CleanHouseSparse"):
        env_args["env_height"] = 14
        env_args["env_width"] = 22
    elif task_name == "WallAvoider":
        env_args["env_height"] = 8
        env_args["env_width"] = 5
    return env_args


def run_hc_single(task_name, seed, max_programs=5000):
    """Run HC on a single (task, seed) and return best reward."""
    dsl = KarelDSL()
    task_cls = get_task_cls(task_name)
    env_args = get_env_args(task_name)
    
    # Single env instance with this seed
    task_envs = [task_cls(env_args, seed)]
    
    search_space = ProgrammaticSpace(dsl, sigma=0.1)
    search_space.set_seed(seed)
    
    search_method = HillClimbing(k=250, e=2)
    progs, rewards = search_method.search(search_space, task_envs, seed=seed, n_iterations=max_programs)
    
    best_reward = max(rewards) if rewards else 0.0
    return best_reward


def flatten_grid(state, max_h=14, max_w=22):
    """Flatten grid state to fixed-size vector. Pad smaller grids."""
    # Use channels: walls (4), markers (5+)
    walls = state[4].astype(float)
    markers_present = state[5:].any(axis=0).astype(float)
    agent_dir = state[0:4].argmax(axis=0).astype(float) * state[0:4].any(axis=0).astype(float)
    
    H, W = state.shape[1], state.shape[2]
    
    # Pad to max size
    walls_padded = np.ones((max_h, max_w), dtype=float)  # pad with walls
    walls_padded[:H, :W] = walls
    
    markers_padded = np.zeros((max_h, max_w), dtype=float)
    markers_padded[:H, :W] = markers_present
    
    agent_padded = np.zeros((max_h, max_w), dtype=float)
    agent_padded[:H, :W] = agent_dir
    
    # Stack and flatten: 3 channels × max_h × max_w
    grid_flat = np.concatenate([
        walls_padded.flatten(),
        markers_padded.flatten(),
        agent_padded.flatten(),
    ])
    return grid_flat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=30, help='Seeds per task')
    parser.add_argument('--max-programs', type=int, default=5000, help='HC budget per instance')
    parser.add_argument('--output', default='multiseed_dataset.csv')
    parser.add_argument('--tasks', nargs='+', default=[
        'MazeSparse', 'StairClimberSparse', 'FourCorners', 'TopOff',
        'Harvester', 'DoorKey', 'OneStroke', 'Seeder',
        'Snake', 'WallAvoider', 'PathFollow'
    ])
    parser.add_argument('--skip-hc', action='store_true', help='Skip HC, only extract features')
    args = parser.parse_args()
    
    # Determine grid columns
    max_h, max_w = 14, 22
    grid_size = 3 * max_h * max_w  # 3 channels
    grid_cols = [f'grid_{i}' for i in range(grid_size)]
    
    # Feature columns
    feature_cols = [
        'grid_height', 'grid_width', 'grid_area',
        'graph_num_free_cells', 'graph_num_wall_cells', 'graph_wall_fraction',
        'graph_num_edges', 'graph_num_components', 'graph_largest_component_frac',
        'graph_mean_degree', 'graph_min_degree', 'graph_max_degree',
        'graph_num_deadends', 'graph_num_corridor_cells', 'graph_num_branch_cells',
        'graph_diameter', 'graph_mean_pair_distance',
        'graph_start_reachable_frac', 'graph_start_eccentricity',
        'graph_num_markers',
        'spectral_lambda_0', 'spectral_lambda_1', 'spectral_lambda_2',
        'spectral_lambda_3', 'spectral_lambda_4', 'spectral_lambda_5',
        'reward_pick_markers', 'reward_put_markers', 'reward_reach_goal',
        'reward_visit_cells', 'reward_sparse', 'reward_crash_penalty_used',
    ]
    
    header = ['task_name', 'seed', 'hc_best_reward', 'difficulty'] + feature_cols + grid_cols
    
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.output)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    
    total = len(args.tasks) * args.seeds
    done = 0
    
    for task_name in args.tasks:
        task_cls = get_task_cls(task_name)
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"{'='*60}")
        
        for seed in range(args.seeds):
            done += 1
            t0 = time.time()
            
            # Generate environment
            env_args = get_env_args(task_name)
            try:
                task = task_cls(env_args, seed)
                env = task.initial_environment
                state = env.state
            except Exception as e:
                print(f"  Seed {seed}: SKIP (env error: {e})")
                continue
            
            # Extract features
            features = extract_features_from_state(state, task_name)
            if features is None:
                print(f"  Seed {seed}: SKIP (no free cells)")
                continue
            
            # Flatten grid
            grid_flat = flatten_grid(state, max_h, max_w)
            
            # Run HC
            if args.skip_hc:
                best_reward = 0.0
                difficulty = 1.0
            else:
                best_reward = run_hc_single(task_name, seed, args.max_programs)
                difficulty = 1.0 - best_reward
            
            elapsed = time.time() - t0
            print(f"  Seed {seed}: reward={best_reward:.4f}, difficulty={difficulty:.4f}, time={elapsed:.1f}s  [{done}/{total}]")
            
            # Write row
            row = [task_name, seed, best_reward, difficulty]
            row += [features[col] for col in feature_cols]
            row += grid_flat.tolist()
            
            with open(output_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
    
    print(f"\nDone! Wrote {output_path}")


if __name__ == '__main__':
    main()
