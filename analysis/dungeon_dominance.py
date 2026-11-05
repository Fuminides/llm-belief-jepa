"""Dominated moves in the saved dungeon games: the probability the player puts on moves no goal is served by.

Bayes does not give the dungeon's optimal policy, but it rules some moves out exactly. Every cell a move can reach
this turn is in view, so the one-step risk of each move is known to a player that has read the rules
(Dungeon.risk: a seen pit kills, fire hurts, ice slides into whatever lies beyond with the stated probability). A
move into a seen wall spends a turn of torch for nothing, and a move that kills with probability one ends the game;
both are dominated by any other move. Each saved game is replayed move for move in the engine, which gives each
turn's state; the per-turn record of every run holds the four moves' probabilities. Reported per run: the mean
probability mass on dominated moves per turn (walls, certain death, and both), the share of the moves played that
were dominated, the excess one-step risk (the risk of the expected move over the least risky move), and the same
mass by turn block. Runs: results/dungeon/<difficulty>/*.json and its subdirectories with the same format (thinking
runs, the stated level), argmax and sample policies.

  python -m analysis.dungeon_dominance                    # results/dungeon/dominance.{json,png,pdf}
"""
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figures import figstyle
from games.dungeon import MOVES, Dungeon
from games.play_dungeon import OUT, Replay

BLOCKS = [(0, 10), (10, 20), (20, 30), (30, 40)]


def turns_of(dungeon, game):
    """Per turn of a saved game: the four moves' probabilities, which moves are dominated (wall, certain death), the
    one-step risk of each, and the move played; from the engine's replay of the game."""
    replay = dungeon.play(dungeon.generate(game["seed"]), Replay([t["move"] for t in game["log"]]))
    out = []
    for saved, t in zip(game["log"], replay["log"]):
        assert list(saved["cell"]) == list(t["cell"]), "the replay left the saved game"
        risk = np.array(t["risk"])  # [4, (die, hurt)]
        out.append({"p": np.array(saved["p"][:4]) / sum(saved["p"][:4]), "wall": np.array([x == "wall" for x in t["truth"]]),
                    "death": risk[:, 0] >= 1, "risk": risk.sum(1), "move": list(MOVES).index(saved["move"])
                    if saved["move"] in MOVES else None})
    return out


def summarise(turns):
    p = np.array([t["p"] for t in turns])
    wall, death = np.array([t["wall"] for t in turns]), np.array([t["death"] for t in turns])
    risk = np.array([t["risk"] for t in turns])
    played = [(i, t["move"]) for i, t in enumerate(turns) if t["move"] is not None]
    return {"turns": len(turns),
            "mass_wall": float((p * wall).sum(1).mean()), "mass_death": float((p * death).sum(1).mean()),
            "mass_dominated": float((p * (wall | death)).sum(1).mean()),
            "played_dominated": float(np.mean([(wall | death)[i, m] for i, m in played])),
            "played_death": float(np.mean([death[i, m] for i, m in played])),
            "excess_risk": float(((p * risk).sum(1) - np.where(wall, np.inf, risk).min(1)).clip(0).mean()),
            "turns_with_death_move": float(death.any(1).mean())}


def main():
    rows = []
    for path in sorted(glob.glob(f"{OUT}/*/**/*.json", recursive=True)):
        r = json.loads(Path(path).read_text())
        if "results" not in r or "config" not in r:
            continue
        dungeon = Dungeon(r["config"])
        for policy, res in r["results"].items():
            games = res["games"] if isinstance(res, dict) else res
            if not games or "p" not in games[0]["log"][0]:
                continue
            turns = [(i, t) for g in games for i, t in enumerate(turns_of(dungeon, g))]
            run = Path(path).relative_to(OUT).with_suffix("")
            row = {"run": str(run.parent), "model": r["model"].split("/")[-1], "policy": policy,
                   **summarise([t for _, t in turns]),
                   "by_block": [summarise([t for i, t in turns if a <= i < b])["mass_dominated"]
                                if any(a <= i < b for i, _ in turns) else None for a, b in BLOCKS]}
            rows.append(row)
            print(f"{row['run']:28s} {row['model']:14s} {policy:12s} dominated mass {row['mass_dominated']:.3f} "
                  f"(wall {row['mass_wall']:.3f}, death {row['mass_death']:.3f}), played {row['played_dominated']:.3f} "
                  f"(death {row['played_death']:.3f}), excess risk {row['excess_risk']:.3f}", flush=True)
    (OUT / "dominance.json").write_text(json.dumps(rows, indent=1))
    plot(rows)


def plot(rows):
    figstyle.apply()
    main_rows = [r for r in rows if r["run"] == "medium"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(figstyle.DOUBLE, 2.6), width_ratios=(1.2, 1))
    labels = [f"{r['model'].replace('Qwen3.5-', '')} {r['policy']}" for r in main_rows]
    y = np.arange(len(main_rows))[::-1]
    ax.barh(y, [r["mass_wall"] for r in main_rows], color=figstyle.GREY, label="into a seen wall")
    ax.barh(y, [r["mass_death"] for r in main_rows], left=[r["mass_wall"] for r in main_rows],
            color=figstyle.VERMILLION, label="certain death")
    ax.set_yticks(y, labels)
    ax.set(xlabel="probability on dominated moves (mean per turn)", title="Medium, no thinking")
    ax.legend(loc="lower right")
    conditions = {"medium": ("no thinking", figstyle.GREY), "medium/think1500": ("thinking 1500", figstyle.ORANGE),
                  "medium/think3000": ("thinking 3000", figstyle.VERMILLION),
                  "medium/think1500_keep1": ("1500, last thought kept", figstyle.GREEN),
                  "medium/think1500_keepall": ("1500, every thought kept", figstyle.BLUE)}
    for r in rows:
        if r["run"] in conditions and r["policy"] != "sample" and r["model"] in ("Qwen3.5-9B", "Qwen3.5-27B"):
            name, colour = conditions[r["run"]]
            bx.plot([(a + b) / 2 for a, b in BLOCKS], [np.nan if v is None else v for v in r["by_block"]], "o-", ms=3,
                    color=colour, ls="-" if r["model"] == "Qwen3.5-27B" else "--",
                    label=f"{r['model'].replace('Qwen3.5-', '')} {name}")
    bx.set(xlabel="turn", ylabel="dominated mass", title="By turn")
    bx.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    figstyle.panel_labels([ax, bx])
    figstyle.save(fig, OUT / "dominance")


if __name__ == "__main__":
    main()
