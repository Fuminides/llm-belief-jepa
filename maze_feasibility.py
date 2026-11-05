"""Can the model read a top-down map? One-move questions on small grids, scored from the logits of the move tokens.

Tasks (the same items for every model):
  adjacent  the goal is next to the agent
  first     the goal is 2-4 steps away; any move on a shortest path is correct
  detour    a wall blocks the straight move; only moves on a shortest path around it are correct
  avoid     two equally short moves, one onto a symbol that hurt the agent earlier; the other is correct
Views: the map with spaces between cells, without spaces, as a list of coordinates, and (for adjacent and
avoid) as a description of the four neighbouring cells, which is what a junction in a path game would give.
Readouts:
  prefix  the answer starts with "I move" and the next token is read (" up", ...). Qwen3.5-27B puts about 40% of
          that mass on " **" and 10% each on " to" and " from", so the four moves get only 20-40%.
  direct  the first token of the answer itself, either case, with or without a leading space ("up", " Up", ...);
          the question asks for one word.

  python maze_feasibility.py --model Qwen/Qwen3.5-2B --readout direct   # writes results/maze_feasibility/direct/<model>.json
  python maze_feasibility.py --plot --readout direct                    # figure from every model run so far
"""
import argparse
import glob
import json
import re
from collections import deque
from pathlib import Path

import numpy as np

MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
SYMBOLS = "%&$+=*"
SIZE = 5
TASKS = {"adjacent": ("spaced", "compact", "coords", "local"), "first": ("spaced", "compact", "coords"),
         "detour": ("spaced", "compact", "coords"), "avoid": ("spaced", "compact", "coords", "local")}
OUT = Path("results/maze_feasibility")


def step(cell, move):
    return cell[0] + MOVES[move][0], cell[1] + MOVES[move][1]


def inside(cell):
    return 0 <= cell[0] < SIZE and 0 <= cell[1] < SIZE


def distances(grid, goal):
    """Steps to the goal from every cell that is not a wall (BFS)."""
    dist, queue = {goal: 0}, deque([goal])
    while queue:
        cell = queue.popleft()
        for move in MOVES:
            nxt = step(cell, move)
            if inside(nxt) and grid[nxt[0]][nxt[1]] != "#" and nxt not in dist:
                dist[nxt] = dist[cell] + 1
                queue.append(nxt)
    return dist


def shortest_moves(grid, agent, goal):
    dist = distances(grid, goal)
    if agent not in dist:
        return set()
    return {m for m in MOVES if dist.get(step(agent, m), 1e9) == dist[agent] - 1}


def random_grid(rng, wall_density):
    return [["#" if rng.random() < wall_density else "." for _ in range(SIZE)] for _ in range(SIZE)]


def make_item(task, rng):
    """A grid with '@' and 'G', the set of correct moves, and the episode history shown before the map."""
    while True:
        agent = (int(rng.integers(SIZE)), int(rng.integers(SIZE)))
        history = ""
        if task == "adjacent":
            grid = random_grid(rng, 0.2)
            move = str(rng.choice(list(MOVES)))
            goal = step(agent, move)
        elif task == "first":
            grid = random_grid(rng, 0.15)
            goal = (int(rng.integers(SIZE)), int(rng.integers(SIZE)))
            if not 2 <= abs(goal[0] - agent[0]) + abs(goal[1] - agent[1]) <= 4:
                continue
        elif task == "detour":
            grid = random_grid(rng, 0.1)
            move = str(rng.choice(list(MOVES)))
            wall, goal = step(agent, move), step(step(agent, move), move)
            if inside(wall):
                grid[wall[0]][wall[1]] = "#"
        else:  # avoid
            grid = random_grid(rng, 0.0)
            vertical, horizontal = str(rng.choice(["up", "down"])), str(rng.choice(["left", "right"]))
            goal = step(step(step(step(agent, vertical), vertical), horizontal), horizontal)
            bad, good = (str(s) for s in rng.choice(list(SYMBOLS), 2, replace=False))
            safe = str(rng.choice([vertical, horizontal]))
            hurt = horizontal if safe == vertical else vertical
            if not (inside(goal) and inside(step(agent, safe)) and inside(step(agent, hurt))):
                continue
            for cell, symbol in ((step(agent, safe), good), (step(agent, hurt), bad)):
                grid[cell[0]][cell[1]] = symbol
            for _ in range(2):  # a few more tiles of each kind elsewhere
                for symbol in (bad, good):
                    r, c = (int(x) for x in rng.integers(SIZE, size=2))
                    if grid[r][c] == ".":
                        grid[r][c] = symbol
            events = [f"you stepped on a {bad} tile and lost 1 health", f"you stepped on a {good} tile and nothing happened"]
            rng.shuffle(events)
            history = f"Earlier in this game, {events[0]}, and {events[1]}.\n"
        if not inside(goal) or goal == agent:
            continue
        grid[agent[0]][agent[1]], grid[goal[0]][goal[1]] = "@", "G"
        correct = {safe} if task == "avoid" else shortest_moves(grid, agent, goal)
        if correct:
            return {"grid": ["".join(row) for row in grid], "agent": agent, "goal": goal,
                    "correct": sorted(correct), "history": history}


def describe(grid, cell):
    if not inside(cell):
        return "the edge of the map"
    content = grid[cell[0]][cell[1]]
    return {"#": "a wall", ".": "an empty cell", "G": "the goal"}.get(content, f"a {content} tile")


def render(item, view):
    grid, (ar, ac), (gr, gc) = item["grid"], item["agent"], item["goal"]
    if view in ("spaced", "compact"):
        sep = " " if view == "spaced" else ""
        return ("Map ('@' is you, 'G' is the goal, '#' is a wall, '.' is an empty cell, other symbols are tiles):\n"
                + "\n".join(sep.join(row) for row in grid))
    if view == "coords":
        lines = [f"The map has {SIZE} rows and {SIZE} columns. You are at row {ar}, column {ac}. "
                 f"The goal is at row {gr}, column {gc}."]
        for symbol, name in [("#", "Walls")] + [(s, f"{s} tiles") for s in SYMBOLS]:
            cells = [f"({r}, {c})" for r in range(SIZE) for c in range(SIZE) if grid[r][c] == symbol]
            if cells:
                lines.append(f"{name}: {', '.join(cells)}.")
        return " ".join(lines) + " All other cells are empty."
    rows, cols = gr - ar, gc - ac
    where = [f"{abs(rows)} row{'s' * (abs(rows) != 1)} {'down' if rows > 0 else 'up'}"] * (rows != 0) + \
            [f"{abs(cols)} column{'s' * (abs(cols) != 1)} {'right' if cols > 0 else 'left'}"] * (cols != 0)
    around = "; ".join(f"{move} is {describe(grid, step(item['agent'], move))}" for move in MOVES)
    return f"The goal is {' and '.join(where)} from you. Around you: {around}."


def question(item, view):
    return ("You are exploring a grid world seen from above. Row 0 is the top row and column 0 the left column: "
            "moving up decreases the row, down increases it, left decreases the column and right increases it. "
            "You cannot enter walls or leave the map.\n"
            f"{item['history']}{render(item, view)}\n"
            "Which move should you make next to reach the goal by a shortest path without losing health? "
            "Answer with one word: up, down, left or right.")


def run(model_name, n_items, seed, readout):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    # Load straight onto the device: on unified memory (GB10) a CPU copy first would hold the weights twice.
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16 if device == "cuda" else torch.float32, device_map=device).eval()
    # Per move, the first tokens that start it: after "I move" a space and the word; as the answer itself, either
    # case, with or without a leading space (after an ASCII map Qwen3.5-4B answers " right").
    variants = [[" " + m] if readout == "prefix" else [s + w for s in ("", " ") for w in (m, m.capitalize())]
                for m in MOVES]
    ids = [[tok(v, add_special_tokens=False).input_ids[0] for v in vs] for vs in variants]
    flat = [i for vs in ids for i in vs]
    assert len(set(flat)) == len(flat), "move words share a first token"
    prefix = "I move" if readout == "prefix" else ""

    rng = np.random.default_rng(seed)
    results = {}
    for task, views in TASKS.items():
        items = [make_item(task, rng) for _ in range(n_items)]
        for view in views:
            probs, mass = [], []
            for item in items:
                text = tok.apply_chat_template([{"role": "user", "content": question(item, view)}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False) + prefix
                with torch.no_grad():
                    logits = model(tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)).logits[0, -1]
                p = torch.softmax(logits.float(), 0)
                p_all = torch.stack([p[v].sum() for v in ids])
                mass.append(p_all.sum().item())
                probs.append((p_all / p_all.sum()).tolist())
            probs = np.array(probs)
            correct = np.array([[m in item["correct"] for m in MOVES] for item in items])
            results[f"{task}/{view}"] = {
                "accuracy": float(correct[np.arange(len(items)), probs.argmax(1)].mean()),
                "p_correct": float((probs * correct).sum(1).mean()),
                "chance": float(correct.mean(1).mean()),
                "move_mass": float(np.mean(mass)),
                "choices": dict(zip(MOVES, np.bincount(probs.argmax(1), minlength=4).tolist())),
                "n": len(items),
            }
            r = results[f"{task}/{view}"]
            print(f"{task:9s} {view:8s} acc {r['accuracy']:.2f} (chance {r['chance']:.2f})  "
                  f"P(correct) {r['p_correct']:.2f}  move-token mass {r['move_mass']:.2f}  choices {r['choices']}",
                  flush=True)
    (OUT / readout).mkdir(parents=True, exist_ok=True)
    path = OUT / readout / f"{model_name.replace('/', '__')}.json"
    path.write_text(json.dumps({"model": model_name, "seed": seed, "readout": readout, "results": results}, indent=1))
    print("saved", path)


def plot(readout):
    import matplotlib.pyplot as plt

    import figstyle

    runs = [json.loads(Path(f).read_text()) for f in glob.glob(f"{OUT}/{readout}/*.json")]
    # Family, then size, so that Qwen3.5-27B comes after Qwen3.5-9B.
    runs.sort(key=lambda r: (r["model"].split("-")[0], float(re.search(r"-(\d+(?:\.\d+)?)B", r["model"]).group(1))))
    figstyle.apply()
    fig, axes = plt.subplots(1, len(TASKS), figsize=(figstyle.DOUBLE, 2.5), sharey=True)
    colours = [figstyle.BLUE, figstyle.VERMILLION, figstyle.GREEN, figstyle.ORANGE, figstyle.PURPLE, figstyle.SKY]
    width = 0.8 / len(runs)
    for ax, (task, views) in zip(axes, TASKS.items()):
        x = np.arange(len(views))
        for i, (run_, colour) in enumerate(zip(runs, colours)):
            acc = np.array([run_["results"][f"{task}/{v}"]["accuracy"] for v in views])
            n = np.array([run_["results"][f"{task}/{v}"]["n"] for v in views])
            ax.bar(x - 0.4 + width * (i + 0.5), acc, width * 0.92, yerr=1.96 * np.sqrt(acc * (1 - acc) / n),
                   color=colour, label=run_["model"].split("/")[-1], error_kw={"lw": 0.6, "capsize": 1.5})
        chance = [runs[0]["results"][f"{task}/{v}"]["chance"] for v in views]
        ax.hlines(chance, x - 0.45, x + 0.45, color="black", lw=1, ls=":", label="chance")
        ax.set_xticks(x, views, rotation=90)
        ax.set_title(task, loc="left")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy (95% CI)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels))
    figstyle.panel_labels(axes)
    figstyle.save(fig, f"results/maze_feasibility_{readout}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--items", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--readout", choices=["prefix", "direct"], default="direct")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()
    plot(args.readout) if args.plot else run(args.model, args.items, args.seed, args.readout)
