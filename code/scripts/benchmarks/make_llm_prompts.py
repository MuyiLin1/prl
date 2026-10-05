"""Build LLM-judge prompt files (JSONL) for every environment instance we have difficulty labels for.
Same prompt template across domains: task rules + ASCII layout + "answer with one digit 0-9".
usage: python make_llm_prompts.py sokoban|lavagap|lavacrossing|karel
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "benchmarks" / "llm_prompts"
OUT.mkdir(parents=True, exist_ok=True)
ASK = ("\n\nOn a scale from 0 (trivially easy) to 9 (extremely hard), how difficult is this instance "
       "for a learning agent to solve? Answer with a single digit only.")


def prompt(rules, legend, ascii_grid):
    return f"{rules}\n\nLegend: {legend}\n\nLayout:\n{ascii_grid}{ASK}"


SOKOBAN_RULES = ("Game: Sokoban. The player moves up/down/left/right and pushes boxes (cannot pull them, "
                 "cannot push two boxes at once). The level is solved when every box is on a target.")
SOKOBAN_LEGEND = "# wall, space floor, @ player, $ box, . target, * box on target, + player on target"


def sokoban():
    import pandas as pd
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "sokoban")]
    from sokoban_core import iter_levels_in_file
    bench = pd.read_csv(ROOT / "data/benchmarks/boxoban_astar/benchmark_levels.csv")
    with open(OUT / "sokoban_astar_benchmark.jsonl", "w") as fh:
        for _, r in bench.iterrows():
            fh.write(json.dumps(dict(id=r.level_id, domain="sokoban",
                                     prompt=prompt(SOKOBAN_RULES, SOKOBAN_LEGEND, r.ascii))) + "\n")
    pool = pd.read_csv(ROOT / "data/sokoban_fastpath_labels.csv")
    cache = {}
    with open(OUT / "sokoban_rl_pool.jsonl", "w") as fh:
        for lid in pool.level_id:
            tier, f, i = lid.split("/")
            path = ROOT / "data/boxoban" / tier / ("" if tier == "hard" else "train") / f"{f}.txt"
            if path not in cache:
                cache[path] = {l.level_id.split("/")[-1]: l for l in iter_levels_in_file(path, tier)}
            fh.write(json.dumps(dict(id=lid, domain="sokoban", prompt=prompt(
                SOKOBAN_RULES, SOKOBAN_LEGEND, "\n".join(cache[path][i].grid)))) + "\n")


MG_LEGEND = "# wall, . floor, L lava, A agent start, G goal"
LAVAGAP_RULES = ("Game: MiniGrid LavaGap. The agent (A) must reach the goal (G). It can turn left, turn right "
                 "and move forward. Stepping on lava (L) ends the episode with failure. A vertical lava strip "
                 "has a single gap. The agent sees only a small window in front of it.")
LAVACROSS_RULES = ("Game: MiniGrid LavaCrossing. The agent (A) must reach the goal (G). It can turn left, turn "
                   "right and move forward. Stepping on lava (L) ends the episode with failure. Lava rivers each "
                   "have one safe crossing. The agent sees only a small 7x7 window in front of it.")


def _mg_ascii(get, w, h, agent, goal):
    rows = []
    for y in range(h):
        row = ""
        for x in range(w):
            if (x, y) == tuple(agent):
                row += "A"
            elif (x, y) == tuple(goal):
                row += "G"
            else:
                row += get(x, y)
        rows.append(row)
    return "\n".join(rows)


def lavagap():
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "leaps")]
    from prog_policies.utils import get_env_name  # noqa: F401
    from prog_policies.minigrid_tasks.lavagap import LavaGapEnv_modified
    with open(OUT / "lavagap_shaped60.jsonl", "w") as fh:
        for s in range(60):
            env = LavaGapEnv_modified(size=9)
            env.reset(seed=s)
            g = env.grid

            def cell(x, y):
                c = g.get(x, y)
                return "." if c is None else {"wall": "#", "lava": "L", "goal": "G"}.get(c.type, ".")
            a = _mg_ascii(cell, g.width, g.height, env.agent_pos, env.goal_pos)
            fh.write(json.dumps(dict(id=str(s), domain="lavagap", prompt=prompt(LAVAGAP_RULES, MG_LEGEND, a))) + "\n")


def lavacrossing():
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "curriculum")]
    from difficulty_scoring import GOAL, LAVA, WALL, _grid_objects, enumerate_pool
    sym = {WALL: "#", LAVA: "L", GOAL: "G"}
    for tag, sizes in [("lavacrossing_p7911", [7, 9, 11]), ("lavacrossing_p91317", [9, 13, 17])]:
        with open(OUT / f"{tag}.jsonl", "w") as fh:
            for lo in enumerate_pool(sizes, [1, 2, 3], 80, 10):
                img, start, goal = _grid_objects(lo.size, lo.n, lo.seed)
                a = _mg_ascii(lambda x, y: sym.get(int(img[x, y]), "."), img.shape[0], img.shape[1], start, goal)
                fh.write(json.dumps(dict(id=json.dumps(lo.sig), domain="lavacrossing",
                                         prompt=prompt(LAVACROSS_RULES, MG_LEGEND, a))) + "\n")


KAREL_DESC = {
    "CleanHouse": "Pick up all markers scattered across the rooms of a house-like grid.",
    "Maze": "Navigate through a maze to reach the goal marker.",
    "FourCorners": "Place a marker in each of the four corners of the grid.",
    "TopOff": "Visit each marker and put one extra marker on every cell that already has a marker.",
    "Harvester": "Pick up every marker on the grid.",
    "StairClimber": "Climb the staircase and reach the marker at the top.",
    "DoorKey": "Pick up the key marker in the left chamber to open the door, then place it on the target marker in the right chamber.",
    "OneStroke": "Visit every free cell exactly once without revisiting a cell.",
    "Seeder": "Place exactly one marker on every empty cell.",
    "Snake": "Collect markers that appear one at a time; the agent's body grows with each marker and it must not hit itself.",
    "WallAvoider": "Fill the interior with markers while avoiding the cells adjacent to the walls.",
    "PathFollow": "Follow the path of markers from start to end.",
}
KAREL_LEGEND = "# wall, . empty, digits = number of markers on a cell, ^ > v < agent facing north/east/south/west"


def karel():
    import glob
    import pandas as pd
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "leaps")]
    from prog_policies.utils import get_env_name  # noqa: F401
    from prog_policies.karel_tasks import get_task_cls
    from generate_multiseed_dataset import get_env_args
    jobs = {}
    for f in glob.glob(str(ROOT / "data/karel_difficulty/karel_*_grids.csv")):
        d = pd.read_csv(f)
        jobs[d.task.iloc[0]] = sorted(d.grid_seed.unique())
    for f in ["averaged_label_topoff200.csv", "averaged_label_screen_snake_seeder.csv", "averaged_label_pilot.csv"]:
        d = pd.read_csv(ROOT / f)
        for t, g in d.groupby("task"):
            jobs[t] = sorted(set(jobs.get(t, [])) | set(g.grid_seed))
    with open(OUT / "karel_all.jsonl", "w") as fh:
        for task, seeds in sorted(jobs.items()):
            cls, args = get_task_cls(task), get_env_args(task)
            for s in seeds:
                st = cls(args, int(s)).initial_environment.state       # (C, H, W) bool
                C, H, W = st.shape
                rows = []
                for r in range(H):
                    row = ""
                    for c in range(W):
                        if st[4, r, c]:
                            row += "#"
                        elif st[:4, r, c].any():
                            row += "^>v<"[int(np.argmax(st[:4, r, c]))]
                        elif st[6:, r, c].any():
                            row += str(min(9, int(np.argmax(st[6:, r, c])) + 1))
                        else:
                            row += "."
                    rows.append(row)
                rules = (f"Game: Karel grid world. The agent moves forward, turns left/right, and picks up or "
                         f"puts down markers. Task ({task}): {KAREL_DESC[task]}")
                fh.write(json.dumps(dict(id=f"{task}/{int(s)}", domain="karel",
                                         prompt=prompt(rules, KAREL_LEGEND, "\n".join(rows)))) + "\n")


if __name__ == "__main__":
    import numpy as np
    {"sokoban": sokoban, "lavagap": lavagap, "lavacrossing": lavacrossing, "karel": karel}[sys.argv[1]]()
    print("wrote", sorted(p.name for p in OUT.glob("*.jsonl")))
