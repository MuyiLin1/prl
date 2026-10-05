#!/usr/bin/env python3
"""
LLM zero-shot difficulty scorer for Karel tasks.

Renders each Karel task's world_map as ASCII and prompts an LLM to predict
difficulty on a 0-1 scale. This is a "can an LLM eyeball difficulty?" baseline.

Supports:
  - Local model via HuggingFace transformers (default: Qwen/Qwen3-1.7B)
  - OpenAI API (if OPENAI_API_KEY is set and --use-openai flag)

Usage:
  python scripts/llm_difficulty_baseline.py --output llm_difficulty_predictions.csv
  python scripts/llm_difficulty_baseline.py --use-openai --model gpt-4o-mini
"""

import sys
import os
import re
import json
import csv
from pathlib import Path
from argparse import ArgumentParser

sys.path.append(".")
sys.path.append("./leaps")

import numpy as np

# Task files and their world_map locations
TASK_DIR = Path("prog_policies/karel_tasks")

TASK_DESCRIPTIONS = {
    "clean_house": "Pick up all markers scattered across rooms in a house-like grid.",
    "maze": "Navigate through a maze to reach the goal marker.",
    "four_corners": "Place a marker in each of the four corners of the grid.",
    "top_off": "Visit each marker stack and add markers to bring them all to the same height.",
    "harvester": "Pick up all markers from the grid.",
    "stair_climber": "Climb a staircase pattern and reach the marker at the top.",
    "door_key": "Find the key, open the door, and reach the goal.",
    "one_stroke": "Visit every free cell exactly once (Hamiltonian path).",
    "seeder": "Place exactly one marker on every free cell.",
    "snake": "Navigate collecting markers while growing (like the snake game).",
    "wall_avoider": "Navigate a narrow corridor without crashing into walls.",
    "path_follow": "Follow a path of markers from start to end.",
}


def extract_world_map_from_file(filepath: Path) -> list:
    """Extract world_map from a Karel task .py file."""
    import ast
    source = filepath.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "world_map":
                    try:
                        return ast.literal_eval(node.value)
                    except:
                        pass
    return None


def world_map_to_ascii(world_map: list) -> str:
    """Convert world_map to a readable ASCII grid."""
    lines = []
    for row in world_map:
        line = ""
        for cell in row:
            if cell == '-':
                line += '#'  # wall
            elif cell == 0:
                line += '.'  # free space
            else:
                line += str(cell)  # marker count
        lines.append(line)
    return '\n'.join(lines)


def build_prompt(task_name: str, ascii_grid: str, task_desc: str, grid_h: int, grid_w: int) -> str:
    """Build the zero-shot difficulty prediction prompt."""
    return f"""You are an expert at evaluating the difficulty of grid-world navigation tasks for a simple robot agent.

The robot can: move forward, turn left, turn right, pick up markers, and put down markers.
The robot has limited perception: it can only check if front/left/right is clear and if markers are present.
The robot executes a fixed program (no learning during execution).

Here is a {grid_h}x{grid_w} grid world:
- '#' = wall (impassable)
- '.' = free space
- Numbers = markers

```
{ascii_grid}
```

Task name: {task_name}
Task description: {task_desc}

On a scale from 0.0 (trivial) to 1.0 (nearly impossible), how difficult is this task for the robot?

Consider:
- How complex is the navigation required?
- How many subgoals must be completed?
- Are there dead ends or bottlenecks that could trap the robot?
- How much of the grid must be explored?

Respond with ONLY a single decimal number between 0.0 and 1.0. Nothing else."""


def score_with_local_model(prompts: list, model_name: str = "Qwen/Qwen3-1.7B") -> list:
    """Score difficulty using a local HuggingFace model."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers")
        sys.exit(1)

    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cpu"
    )
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        do_sample=False, max_new_tokens=20
    )

    scores = []
    for i, prompt in enumerate(prompts):
        print(f"  Scoring task {i+1}/{len(prompts)}...", end="\r")
        output = pipe(prompt)[0]["generated_text"]
        # Extract the generated part (after prompt)
        generated = output[len(prompt):].strip()
        score = parse_score(generated)
        scores.append(score)

    print()
    return scores


def score_with_openai(prompts: list, model_name: str = "gpt-4o-mini") -> list:
    """Score difficulty using OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai not installed. Run: pip install openai")
        sys.exit(1)

    client = OpenAI()
    scores = []
    for i, prompt in enumerate(prompts):
        print(f"  Scoring task {i+1}/{len(prompts)}...", end="\r")
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20,
        )
        generated = response.choices[0].message.content.strip()
        score = parse_score(generated)
        scores.append(score)

    print()
    return scores


def parse_score(text: str) -> float:
    """Extract a float score from LLM output."""
    # Try to find a decimal number between 0 and 1
    matches = re.findall(r'(0\.\d+|1\.0|0|1)', text)
    if matches:
        return float(matches[0])
    # Try any float
    matches = re.findall(r'(\d+\.?\d*)', text)
    if matches:
        val = float(matches[0])
        return min(max(val, 0.0), 1.0)
    return 0.5  # default if parsing fails


def score_with_mock(prompts: list, task_names: list, features_csv: str) -> list:
    """
    Mock LLM scoring using a simple rule-based heuristic on the ASCII grid.
    Useful for testing the pipeline without GPU/API access.
    Scores based on: grid size, wall density, number of rooms.
    """
    scores = []
    for i, (prompt, name) in enumerate(zip(prompts, task_names)):
        # Extract grid from prompt
        lines = prompt.split("```\n")
        if len(lines) >= 2:
            grid_text = lines[1].split("\n```")[0]
            grid_lines = [l for l in grid_text.strip().split("\n") if l]
        else:
            grid_lines = []

        if not grid_lines:
            scores.append(0.5)
            continue

        h = len(grid_lines)
        w = max(len(l) for l in grid_lines) if grid_lines else 0
        total = h * w
        walls = sum(c == '#' for line in grid_lines for c in line)
        wall_frac = walls / max(1, total)

        # Count connected regions of free space (rough room count)
        # Simple heuristic: more walls + bigger grid = harder
        size_factor = min(1.0, (h * w) / 300.0)  # normalize by CleanHouse size
        complexity = 0.3 * wall_frac + 0.3 * size_factor

        # Task-specific adjustments
        if "one_stroke" in name or "hamiltonian" in name.lower():
            complexity += 0.3  # Hamiltonian path is NP-hard
        elif "clean_house" in name:
            complexity += 0.2  # Large grid + many subgoals
        elif "snake" in name:
            complexity += 0.15
        elif "door_key" in name:
            complexity += 0.1

        scores.append(min(1.0, max(0.0, complexity)))

    return scores


def main():
    parser = ArgumentParser()
    parser.add_argument("--output", default="llm_difficulty_predictions.csv")
    parser.add_argument("--use-openai", action="store_true")
    parser.add_argument("--mock", action="store_true", help="Use rule-based mock (no model needed)")
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument("--features-csv", default="prog_policies/karel_tasks/all_karel_task_features.csv")
    args = parser.parse_args()

    task_files = sorted(TASK_DIR.glob("*.py"))
    task_files = [f for f in task_files if f.stem not in (
        "__init__", "karel_curriculum_difficulty_predictor",
        "karel_curriculum_difficulty_predictor_backup"
    )]

    prompts = []
    task_names = []

    # Try to generate grids from actual task environments
    try:
        from prog_policies.utils import get_env_name
        from prog_policies.karel_tasks import get_task_cls as get_karel_task_cls
        from prog_policies.karel import KarelEnvironment
        USE_ENV = True
    except ImportError:
        USE_ENV = False

    KAREL_TASKS_CONFIG = {
        "CleanHouse": {"env_height": 14, "env_width": 22},
        "StairClimberSparse": {"env_height": 12, "env_width": 12},
        "FourCorners": {"env_height": 12, "env_width": 12},
        "TopOff": {"env_height": 12, "env_width": 12},
        "MazeSparse": {"env_height": 8, "env_width": 8},
        "Harvester": {"env_height": 8, "env_width": 8},
        "DoorKey": {"env_height": 8, "env_width": 8},
        "OneStroke": {"env_height": 8, "env_width": 8},
        "Seeder": {"env_height": 8, "env_width": 8},
        "Snake": {"env_height": 8, "env_width": 8},
        "WallAvoider": {"env_height": 8, "env_width": 5},
        "PathFollow": {"env_height": 8, "env_width": 8},
    }

    if USE_ENV:
        for task_name, cfg in KAREL_TASKS_CONFIG.items():
            env_args = {
                "env_height": cfg["env_height"],
                "env_width": cfg["env_width"],
                "crashable": False,
                "leaps_behaviour": True,
                "max_calls": 10000,
            }
            try:
                task_cls = get_karel_task_cls(task_name)
            except (AssertionError, KeyError):
                print(f"  Skipping {task_name}: task class not found")
                continue

            # Generate a representative grid (seed 0)
            task_env = task_cls(env_args, 0)
            env = task_env.initial_environment
            state = env.state
            h, w = state.shape[1], state.shape[2]

            # Convert state tensor to ASCII
            ascii_lines = []
            for y in range(h):
                line = ""
                for x in range(w):
                    if state[4, y, x]:  # wall
                        line += '#'
                    elif np.any(state[0:4, y, x]):  # agent
                        line += 'A'
                    elif np.any(state[6:, y, x]):  # marker(s)
                        line += 'M'
                    else:
                        line += '.'
                ascii_lines.append(line)
            ascii_grid = '\n'.join(ascii_lines)

            task_desc = TASK_DESCRIPTIONS.get(task_name.lower().replace("sparse", ""),
                        TASK_DESCRIPTIONS.get(task_name.lower(), "Complete the task objective."))
            # Also try matching by converting to underscore form
            for key in TASK_DESCRIPTIONS:
                if key.replace("_", "") == task_name.lower().replace("sparse", ""):
                    task_desc = TASK_DESCRIPTIONS[key]
                    break

            prompt = build_prompt(task_name, ascii_grid, task_desc, h, w)
            prompts.append(prompt)
            task_names.append(task_name.lower().replace("sparse", ""))
    else:
        # Fallback: only use tasks with world_map literals in source
        for tf in task_files:
            world_map = extract_world_map_from_file(tf)
            if world_map is None:
                print(f"  Skipping {tf.stem}: no world_map found")
                continue
            task_name = tf.stem
            task_desc = TASK_DESCRIPTIONS.get(task_name, "Complete the task objective.")
            ascii_grid = world_map_to_ascii(world_map)
            h = len(world_map)
            w = len(world_map[0]) if world_map else 0
            prompt = build_prompt(task_name, ascii_grid, task_desc, h, w)
            prompts.append(prompt)
            task_names.append(task_name)

    print(f"Built prompts for {len(prompts)} tasks")

    if args.mock:
        model = "mock_rule_based"
        print("Using mock rule-based scorer (no model)")
        scores = score_with_mock(prompts, task_names, args.features_csv)
    elif args.use_openai:
        model = args.model or "gpt-4o-mini"
        print(f"Using OpenAI API ({model})")
        scores = score_with_openai(prompts, model)
    else:
        model = args.model or "Qwen/Qwen3-1.7B"
        print(f"Using local model ({model})")
        scores = score_with_local_model(prompts, model)

    # Write output
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_name", "llm_difficulty_score", "model"])
        for name, score in zip(task_names, scores):
            writer.writerow([name, f"{score:.4f}", model])

    print(f"\nWrote {args.output}")
    print("\nLLM difficulty predictions:")
    for name, score in sorted(zip(task_names, scores), key=lambda x: x[1], reverse=True):
        print(f"  {name:20s}  score={score:.4f}")


if __name__ == "__main__":
    main()
