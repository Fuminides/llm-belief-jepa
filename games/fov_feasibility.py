"""Can the model play from what it sees? One-move questions with a first-person view, read like maze_feasibility.py
(the first token of the answer, either case, with or without a leading space).

The agent sees the cells in its line of sight: walls block the view, and terrain.json can set a radius. The view is
described in words, like a text adventure: what lies along each of the four directions up to the first thing that
blocks it, then the other visible cells that hold something, as offsets from the agent ("2 up and 1 right").
terrain.json holds the map size, the symbol pool and the terrain effects; symbols are assigned to effects at random
per item, so only the history says what a symbol does.

Tasks (the same items for every model):
  first   the goal is in view, 2-5 steps away; any move on a shortest path is correct. Only items whose answer
          follows from the visible cells are kept. Views: the first-person view, and the whole map as coordinates.
  memory  the goal was in view one or two moves ago and is now hidden behind walls. The past turns are chat turns
          (observation and question, then the move). Views: remember (only the past turns say where the goal is),
          told (the observation also gives the hidden goal's offset), none (no past turns: chance).
  ice     two equally short first moves, one onto an ice tile and one onto a harmless tile; slipping carries the
          agent one cell further, onto the tile that hurt it earlier. The history narrates earlier visits to the
          tiles. Conditions, on the same layouts: slip (the ice has slipped; only the harmless tile is correct),
          no_hazard (nothing to slip onto; both are correct), never_slipped (the ice never slipped; both are
          correct). The measure is how often the ice is chosen.

  python -m games.fov_feasibility --model Qwen/Qwen3.5-2B   # writes results/fov_feasibility/<model>.json
  python -m games.fov_feasibility --plot                    # figure from every model run so far
  python -m games.fov_feasibility --show 1                  # print example prompts, no model
"""
import argparse
import glob
import json
import re
from collections import deque
from fractions import Fraction
from pathlib import Path

import numpy as np

from games.maze_feasibility import MOVES, step

CONFIG = json.loads(Path(__file__).with_name("terrain.json").read_text())
ROWS, COLS, RADIUS = CONFIG["rows"], CONFIG["cols"], CONFIG["view_radius"]
TERRAIN = CONFIG["terrain"]
WALL = TERRAIN["wall"]["symbol"]
TASKS = {"first": ("fov", "coords"), "memory": ("remember", "told", "none"),
         "ice": ("slip", "no_hazard", "never_slipped")}
ICE_VISITS = 3  # earlier steps on the ice that the history narrates
OUT = Path("results/fov_feasibility")

INTRO_FOV = ("You are exploring a grid world, seen from where you stand. You see only what is in your line of sight: "
             "walls block the view. Positions are given from you: '2 up and 1 right' is two rows up and one column "
             "to the right. You cannot enter walls or leave the map.\n")
INTRO_MAP = ("You are exploring a grid world seen from above. Row 0 is the top row and column 0 the left column: "
             "moving up decreases the row, down increases it, left decreases the column and right increases it. "
             "You cannot enter walls or leave the map.\n")
QUESTION = ("Which move should you make next to reach the goal by a shortest path without losing health? "
            "Answer with one word: up, down, left or right.")


def inside(cell):
    return 0 <= cell[0] < ROWS and 0 <= cell[1] < COLS


def wall(grid, cell):
    return grid[cell[0]][cell[1]] == WALL


def visible(grid, agent):
    """Cells in the line of sight from the agent's centre. A sample point on a cell border is blocked only if the
    cells on both sides are walls, so the view passes along a wall and through diagonal gaps."""
    seen = {agent}
    for r in range(ROWS):
        for c in range(COLS):
            dr, dc = r - agent[0], c - agent[1]
            if (r, c) == agent or (RADIUS is not None and dr * dr + dc * dc > RADIUS * RADIUS):
                continue
            n = 20 * max(abs(dr), abs(dc))
            clear = True
            for k in range(1, n):
                x, y = agent[0] + Fraction(dr * k, n), agent[1] + Fraction(dc * k, n)
                rows = {int(x - Fraction(1, 2)), int(x + Fraction(1, 2))} if x.denominator == 2 else {round(x)}
                cols = {int(y - Fraction(1, 2)), int(y + Fraction(1, 2))} if y.denominator == 2 else {round(y)}
                cells = {(a, b) for a in rows for b in cols} - {agent, (r, c)}
                if cells and all(wall(grid, cell) for cell in cells):
                    clear = False
                    break
            if clear:
                seen.add((r, c))
    return seen


def shortest_moves(grid, agent, goal, passable):
    """First moves on a shortest path to the goal through cells for which passable(cell) holds (BFS)."""
    dist, queue = {goal: 0}, deque([goal])
    while queue:
        cell = queue.popleft()
        for move in MOVES:
            nxt = step(cell, move)
            if inside(nxt) and nxt not in dist and passable(nxt):
                dist[nxt] = dist[cell] + 1
                queue.append(nxt)
    if agent not in dist:
        return set()
    return {m for m in MOVES if dist.get(step(agent, m), 1e9) == dist[agent] - 1}


def determined(grid, agent, goal, known):
    """The shortest first moves if they are the same whether the unseen cells are walls or free, else None."""
    pessimistic = shortest_moves(grid, agent, goal, lambda c: c in known and not wall(grid, c))
    optimistic = shortest_moves(grid, agent, goal, lambda c: c not in known or not wall(grid, c))
    return sorted(pessimistic) if pessimistic and pessimistic == optimistic else None


def name(content, n=1):
    if content == WALL:
        return "a wall" if n == 1 else f"{n} walls"
    if content == ".":
        return "an empty cell" if n == 1 else f"{n} empty cells"
    if content == "G":
        return "the goal"
    return f"a {content} tile" if n == 1 else f"{n} {content} tiles"


def offset(dr, dc):
    parts = [f"{abs(dr)} {'down' if dr > 0 else 'up'}"] * (dr != 0) + [f"{abs(dc)} {'right' if dc > 0 else 'left'}"] * (dc != 0)
    return " and ".join(parts)


def observe(grid, agent):
    """What the agent sees, in words."""
    seen = visible(grid, agent)
    lines = []
    for move in MOVES:
        contents, cell = [], step(agent, move)
        while inside(cell) and cell in seen:
            contents.append(grid[cell[0]][cell[1]])
            if contents[-1] == WALL:
                break
            cell = step(cell, move)
        runs = []  # consecutive equal contents merged: "3 empty cells"
        for content in contents:
            if runs and runs[-1][0] == content and content != "G":
                runs[-1][1] += 1
            else:
                runs.append([content, 1])
        items = [name(content, n) for content, n in runs]
        if not inside(cell):
            items.append("the edge of the map")
        elif cell not in seen:
            items.append("nothing further in view")
        text = items[-1] if len(items) == 1 else ", ".join(items[:-1]) + ", then " + items[-1]
        lines.append(f"{move.capitalize()}: {text}.")
    off_axis = sorted((c for c in seen if c[0] != agent[0] and c[1] != agent[1] and grid[c[0]][c[1]] != "."),
                      key=lambda c: (abs(c[0] - agent[0]) + abs(c[1] - agent[1]), c))
    things = [f"{name(grid[c[0]][c[1]])} {offset(c[0] - agent[0], c[1] - agent[1])}" for c in off_axis]
    if things:
        lines.append(f"Also in view: {'; '.join(things)}.")
    lines.append("The other cells in view are empty.")
    return "You see:\n" + "\n".join(lines)


def coords(grid, agent):
    """The whole map as coordinates, as in maze_feasibility.py."""
    goal = next((r, c) for r in range(ROWS) for c in range(COLS) if grid[r][c] == "G")
    lines = [f"The map has {ROWS} rows and {COLS} columns. You are at row {agent[0]}, column {agent[1]}. "
             f"The goal is at row {goal[0]}, column {goal[1]}."]
    cells = [f"({r}, {c})" for r in range(ROWS) for c in range(COLS) if grid[r][c] == WALL]
    if cells:
        lines.append(f"Walls: {', '.join(cells)}.")
    return " ".join(lines) + " All other cells are empty."


def random_grid(rng, wall_density):
    return [[WALL if rng.random() < wall_density else "." for _ in range(COLS)] for _ in range(ROWS)]


def random_cell(rng):
    return int(rng.integers(ROWS)), int(rng.integers(COLS))


def make_first(rng):
    while True:
        grid, agent, goal = random_grid(rng, 0.2), random_cell(rng), random_cell(rng)
        if wall(grid, agent) or wall(grid, goal) or not 2 <= abs(goal[0] - agent[0]) + abs(goal[1] - agent[1]) <= 5:
            continue
        seen = visible(grid, agent)
        correct = determined(grid, agent, goal, seen) if goal in seen else None
        if correct:
            grid[goal[0]][goal[1]] = "G"
            return {"grid": grid, "path": [agent], "moves": [], "correct": {v: correct for v in TASKS["first"]}}


def make_memory(rng):
    while True:
        grid, agent, goal = random_grid(rng, 0.25), random_cell(rng), random_cell(rng)
        if wall(grid, agent) or wall(grid, goal) or not 3 <= abs(goal[0] - agent[0]) + abs(goal[1] - agent[1]) <= 6:
            continue
        if goal not in visible(grid, agent):
            continue
        path, moves = [agent], []
        for _ in range(int(rng.integers(1, 3))):
            options = [m for m in MOVES if inside(step(path[-1], m)) and not wall(grid, step(path[-1], m))
                       and step(path[-1], m) != goal]
            if not options:
                break
            moves.append(str(rng.choice(options)))
            path.append(step(path[-1], moves[-1]))
        if not moves or any(goal in visible(grid, p) for p in path[1:]):
            continue
        known = set().union(*(visible(grid, p) for p in path))
        correct = determined(grid, path[-1], goal, known)
        if correct:
            grid[goal[0]][goal[1]] = "G"
            return {"grid": grid, "path": path, "moves": moves, "goal": goal,
                    "correct": {v: correct for v in TASKS["memory"]}}


def make_ice(rng):
    """One open layout; the conditions differ in the cell beyond the ice and in the narrated history."""
    while True:
        agent = random_cell(rng)
        vertical, horizontal = str(rng.choice(["up", "down"])), str(rng.choice(["left", "right"]))
        goal = step(step(step(step(agent, vertical), vertical), horizontal), horizontal)
        if inside(goal):
            break
    ice = str(rng.choice([vertical, horizontal]))
    safe = horizontal if ice == vertical else vertical
    ice_s, fire_s, plain_s = (str(s) for s in rng.choice(list(CONFIG["symbols"]), 3, replace=False))
    base = [["." for _ in range(COLS)] for _ in range(ROWS)]
    ice_cell, safe_cell = step(agent, ice), step(agent, safe)
    landing = ice_cell
    for _ in range(TERRAIN["ice"]["slip_cells"]):
        landing = step(landing, ice)
    base[ice_cell[0]][ice_cell[1]], base[safe_cell[0]][safe_cell[1]] = ice_s, plain_s
    base[goal[0]][goal[1]] = "G"
    # One tile of each kind away from the rectangle between the agent and the goal, so that every symbol is in view.
    box = {(r, c) for r in range(min(agent[0], goal[0]), max(agent[0], goal[0]) + 1)
           for c in range(min(agent[1], goal[1]), max(agent[1], goal[1]) + 1)}
    free = [(r, c) for r in range(ROWS) for c in range(COLS) if (r, c) not in box]
    for symbol in (ice_s, fire_s, plain_s):
        r, c = free.pop(int(rng.integers(len(free))))
        base[r][c] = symbol

    def visits(slips):
        return ([f"you stepped on a {ice_s} tile and slid one cell further in the direction you were moving"] * slips
                + [f"you stepped on a {ice_s} tile and stopped on it"] * (ICE_VISITS - slips))
    p = TERRAIN["ice"]["slip_probability"]
    slips = 0
    while slips == 0:
        slips = int(rng.binomial(ICE_VISITS, p))
    others = [f"you stepped on a {fire_s} tile and lost {TERRAIN['fire']['damage']} health",
              f"you stepped on a {plain_s} tile and nothing happened"]
    variants = {}
    for condition in TASKS["ice"]:
        grid = [row[:] for row in base]
        if condition != "no_hazard":
            grid[landing[0]][landing[1]] = fire_s
        events = visits(0 if condition == "never_slipped" else slips) + others
        rng.shuffle(events)
        variants[condition] = {"grid": grid, "history": f"Earlier in this game, {'; '.join(events)}.\n"}
    return {"variants": variants, "path": [agent], "moves": [], "target": ice, "toward": [vertical, horizontal],
            "correct": {"slip": [safe], "no_hazard": sorted([safe, ice]), "never_slipped": sorted([safe, ice])}}


MAKE = {"first": make_first, "memory": make_memory, "ice": make_ice}


def messages(task, item, view):
    """The chat turns for one question: past turns (observation and question, then the move) and the current one."""
    if task == "ice":
        variant = item["variants"][view]
        return [{"role": "user", "content": INTRO_FOV + variant["history"] + observe(variant["grid"], item["path"][0])
                 + "\n" + QUESTION}]
    grid, path = item["grid"], item["path"]
    if view == "coords":
        return [{"role": "user", "content": INTRO_MAP + coords(grid, path[-1]) + "\n" + QUESTION}]
    if view == "none" or len(path) == 1:
        return [{"role": "user", "content": INTRO_FOV + observe(grid, path[-1]) + "\n" + QUESTION}]
    turns = [{"role": "user", "content": INTRO_FOV + observe(grid, path[0]) + "\n" + QUESTION}]
    for move, cell in zip(item["moves"], path[1:]):
        told = ""
        if view == "told" and cell == path[-1]:
            told = f"\nThe goal is out of sight, {offset(item['goal'][0] - cell[0], item['goal'][1] - cell[1])} from you."
        turns += [{"role": "assistant", "content": move},
                  {"role": "user", "content": f"You moved {move}.\n" + observe(grid, cell) + told + "\n" + QUESTION}]
    return turns


def items_for(task, n_items, rng):
    return [MAKE[task](rng) for _ in range(n_items)]


def run(model_name, n_items, seed):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    # Straight onto the device: on unified memory (GB10) a CPU copy first would hold the weights twice.
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16 if device == "cuda" else torch.float32, device_map=device).eval()
    ids = [[tok(s + w, add_special_tokens=False).input_ids[0] for s in ("", " ") for w in (m, m.capitalize())]
           for m in MOVES]
    flat = [i for vs in ids for i in vs]
    assert len(set(flat)) == len(flat), "move words share a first token"

    rng = np.random.default_rng(seed)
    results = {}
    for task, views in TASKS.items():
        items = items_for(task, n_items, rng)
        for view in views:
            probs, mass = [], []
            for item in items:
                text = tok.apply_chat_template(messages(task, item, view), tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
                with torch.no_grad():
                    logits = model(tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)).logits[0, -1]
                p = torch.softmax(logits.float(), 0)
                p_all = torch.stack([p[v].sum() for v in ids])
                mass.append(p_all.sum().item())
                probs.append((p_all / p_all.sum()).tolist())
            probs, choice = np.array(probs), np.array(probs).argmax(1)
            correct = np.array([[m in item["correct"][view] for m in MOVES] for item in items])
            r = results[f"{task}/{view}"] = {
                "accuracy": float(correct[np.arange(len(items)), choice].mean()),
                "p_correct": float((probs * correct).sum(1).mean()),
                "chance": float(correct.mean(1).mean()),
                "move_mass": float(np.mean(mass)),
                "choices": dict(zip(MOVES, np.bincount(choice, minlength=4).tolist())),
                "n": len(items),
            }
            line = (f"{task:7s} {view:13s} acc {r['accuracy']:.2f} (chance {r['chance']:.2f})  "
                    f"P(correct) {r['p_correct']:.2f}  move-token mass {r['move_mass']:.2f}")
            if task == "ice":  # how often the ice is chosen, and how often a move away from the goal
                target = np.array([list(MOVES).index(item["target"]) for item in items])
                toward = np.array([[m in item["toward"] for m in MOVES] for item in items])
                r["ice_rate"] = float((choice == target).mean())
                r["p_ice"] = float(probs[np.arange(len(items)), target].mean())
                r["away_rate"] = float(1 - toward[np.arange(len(items)), choice].mean())
                line += f"  ice {r['ice_rate']:.2f} (P {r['p_ice']:.2f})  away {r['away_rate']:.2f}"
            print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{model_name.replace('/', '__')}.json"
    path.write_text(json.dumps({"model": model_name, "seed": seed, "config": CONFIG, "results": results}, indent=1))
    print("saved", path)


def show(n, seed):
    rng = np.random.default_rng(seed)
    for task, views in TASKS.items():
        items = items_for(task, n, rng)
        for item in items:
            for view in views:
                print(f"===== {task} / {view}  correct: {item['correct'][view]}")
                for turn in messages(task, item, view):
                    print(f"[{turn['role']}] {turn['content']}")


def plot():
    import matplotlib.pyplot as plt

    from figures import figstyle
    runs = [json.loads(Path(f).read_text()) for f in glob.glob(f"{OUT}/*.json")]
    runs.sort(key=lambda r: (r["model"].split("-")[0], float(re.search(r"-(\d+(?:\.\d+)?)B", r["model"]).group(1))))
    figstyle.apply()
    fig, axes = plt.subplots(1, len(TASKS), figsize=(figstyle.DOUBLE, 2.5))
    colours = [figstyle.BLUE, figstyle.VERMILLION, figstyle.GREEN, figstyle.ORANGE, figstyle.PURPLE, figstyle.SKY]
    width = 0.8 / len(runs)
    for ax, (task, views) in zip(axes, TASKS.items()):
        key = "ice_rate" if task == "ice" else "accuracy"
        x = np.arange(len(views))
        for i, (run_, colour) in enumerate(zip(runs, colours)):
            y = np.array([run_["results"][f"{task}/{v}"][key] for v in views])
            n = np.array([run_["results"][f"{task}/{v}"]["n"] for v in views])
            ax.bar(x - 0.4 + width * (i + 0.5), y, width * 0.92, yerr=1.96 * np.sqrt(y * (1 - y) / n),
                   color=colour, label=run_["model"].split("/")[-1], error_kw={"lw": 0.6, "capsize": 1.5})
        if task == "ice":
            ax.hlines(0.5, -0.45, len(views) - 0.55, color="black", lw=1, ls=":", label="indifferent")
            ax.set_ylabel("Chooses the ice (95% CI)")
        else:
            chance = [runs[0]["results"][f"{task}/{v}"]["chance"] for v in views]
            ax.hlines(chance, x - 0.45, x + 0.45, color="black", lw=1, ls=":", label="chance")
            ax.set_ylabel("Accuracy (95% CI)")
        ax.set_xticks(x, [v.replace("_", " ") for v in views], rotation=90)
        ax.set_title(task, loc="left")
        ax.set_ylim(0, 1)
    legend = {}  # one entry per label across panels: the models, chance, indifferent
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            legend.setdefault(label, handle)
    fig.legend(legend.values(), legend.keys(), loc="outside lower center", ncol=len(legend))
    figstyle.panel_labels(axes)
    figstyle.save(fig, "results/fov_feasibility")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--items", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--show", type=int, default=0, help="print this many example prompts per task and stop")
    args = p.parse_args()
    if args.plot:
        plot()
    elif args.show:
        show(args.show, args.seed)
    else:
        run(args.model, args.items, args.seed)
