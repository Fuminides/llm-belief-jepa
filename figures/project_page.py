"""Data for the project page (docs/index.html): the clue-game and dungeon demos play back real episodes.

Writes docs/demo/clue_data.js (three episodes of Qwen3.5-2B from data/qwen3.5-2b-memory, the belief after each clue
next to the Bayesian posterior) and docs/demo/dungeon_data.js (saved games played again from their moves, as
games_figure.py does, with the text the model read each turn, the level and the sprites of pixelart.py), and copies
the manuscript's PDF to docs/paper.pdf. The page loads the two files as scripts, so it also works from the disk.

  python -m figures.project_page
"""
import json
import shutil
from pathlib import Path

import numpy as np

from games import game
from games.dungeon import MOVES, Dungeon
from games.pixelart import PALETTE, SPRITES, replay_states

OUT = Path("docs/demo")
PDF = Path("manuscript/working_doc_0/main.pdf")

# The three cases: episode index in data/qwen3.5-2b-memory/shard_000000.npz, title, and what to watch for.
CASES = [
    (20, "The sources agree",
     "Alice speaks twice about the material, Bob three times about the shape, then Alice about the size. Nobody "
     "contradicts anyone. The 2B moves the same way as Bayes after every clue, only less far: it never reaches the "
     "near-certainty that two Alice clues give the posterior."),
    (10, "Alice against Bob",
     "Bob says small, Alice says not small, and the two keep disagreeing on the size. Bayes follows Alice, who is "
     "right 90% of the time; the 2B stays with Bob, the first to speak, and later Alice clues barely move it. "
     "The gap on the size is the largest the game produces."),
    (113, "A negation, three times",
     "After one Bob clue for small, Alice says the object is not small three times in a row. Bayes goes to a 98% "
     "big; the 2B stays near 25% and does not move in the right direction. A negated clue weighs about half an "
     "assertion in every size of the model, and here it is not heard at all."),
]

# The dungeon games: results file, policy, game index, and a one-line label.
GAMES = [
    ("results/dungeon/medium/Qwen__Qwen3.5-27B.json", "argmax", 3, "reaches the exit in 10 turns, one slide on ice"),
    ("results/dungeon/medium/Qwen__Qwen3.5-27B.json", "argmax", 9, "walks into a pit it could see, turn 9"),
    ("results/dungeon/medium/Qwen__Qwen3.5-27B.json", "argmax", 12, "burns twice and the torch runs out"),
    ("results/dungeon/medium/think1500_keep1/Qwen__Qwen3.5-27B.json", "think1500_keep1", 5,
     "thinking, the last thought kept: the exit in 17 turns"),
    ("results/dungeon/medium/Qwen__Qwen3.5-9B.json", "sample", 20, "finds the exit after 24 turns"),
    ("results/dungeon/medium/Qwen__Qwen3.5-2B.json", "sample", 3, "bumps into walls for 22 turns, then a pit"),
    ("results/dungeon/medium/Qwen__Qwen3.5-2B.json", "sample", 6, "stands on fire until its lives are gone"),
]


def clue_cases():
    z = np.load("data/qwen3.5-2b-memory/shard_000000.npz")
    cases = []
    for episode, title, note in CASES:
        acts = z["actions"][episode]
        clues = []
        for a in acts:
            attr, value, negated, source = game.ACTIONS[a]
            clues.append({"source": game.SOURCE_NAMES[source], "text": game.clue_text(a).strip(), "attr": attr,
                          "value": value, "negated": negated})
        cases.append({"episode": int(episode), "title": title, "note": note, "clues": clues,
                      "llm": np.round(z["llm_belief"][episode], 4).tolist(),
                      "bayes": np.round(z["bayes_belief"][episode], 4).tolist()})
    return {"model": "Qwen3.5-2B", "attributes": [[a, *v] for a, v in game.ATTRIBUTES.items()],
            "sources": game.SOURCES, "intro": game.intro(), "cases": cases}


def dungeon_games():
    games = []
    for path, policy, index, label in GAMES:
        r = json.loads(Path(path).read_text())
        g = r["results"][policy]["games"][index]
        d = Dungeon(r["config"])
        level = d.generate(g["seed"])
        moves = [t["move"] for t in g["log"]]
        states, last = replay_states(d, level, moves)
        assert [tuple(s.cell) for s, _ in states] == [tuple(t["cell"]) for t in g["log"]], "the replay left the saved game"
        rules = d.rules()
        turns, play, turn = [], d.game(level), None
        for k, t in enumerate(g["log"]):
            turn = next(play) if k == 0 else play.send((moves[k - 1], {}))
            text = turn["text"]
            if k == 0:
                assert text.startswith(rules)
                text = text[len(rules):].lstrip("\n")
            state = states[k][0]
            turns.append({"cell": list(turn["cell"]), "lives": state.lives, "torch": state.turns_left, "text": text,
                          "seen": sorted(list(c) for c in turn["seen"]), "move": t["move"],
                          "p": [round(x, 4) for x in t["p"]], "events": t["events"]})
        try:
            play.send((moves[-1], {}))
        except StopIteration as end:
            record = end.value
        assert record["outcome"] == g["outcome"] and tuple(record["final"]) == tuple(g["final"])
        games.append({"model": r["model"].split("/")[-1], "policy": policy, "seed": g["seed"], "label": label,
                      "outcome": g["outcome"], "cause": g["cause"], "turns": g["turns"], "shortest": g["shortest"],
                      "lives": g["lives"], "final": list(g["final"]), "end": g["end"], "rules": rules,
                      "grid": ["".join(d.terrain[c]["symbol"] for c in row) for row in level.grid],
                      "start": list(level.start), "exit": list(level.exit), "log": turns})
    config = json.loads(Path(GAMES[0][0]).read_text())["config"]
    symbols = {t["symbol"]: name for name, t in config["terrain"].items()}
    return {"moves": list(MOVES), "symbols": symbols, "lives": config["player"]["lives"],
            "torch": config["player"]["turns"], "palette": PALETTE, "sprites": SPRITES, "games": games}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in (("CLUE_DATA", clue_cases()), ("DUNGEON_DATA", dungeon_games())):
        path = OUT / f"{name.lower()}.js"
        path.write_text(f"window.{name} = {json.dumps(data, separators=(',', ':'))};\n")
        print(f"{path}: {path.stat().st_size / 1024:.0f} KB")
    if PDF.exists():
        shutil.copy(PDF, "docs/paper.pdf")
        print("docs/paper.pdf copied")


if __name__ == "__main__":
    main()
