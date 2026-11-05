"""The dungeon as the paper shows it: one level, what the player has seen by a turn, and the text the LLM reads then.

A saved game is played again from its moves (pixelart.py), so the panels show what that player saw. Default: the
27B keeping its last thought (think_dungeon.py --keep 1) on medium level 5, at turn 8, when the exit first comes
into view with a pit on the straight way to it.

  python dungeon_figure.py [--results FILE] [--game 5] [--turn 8]   # results/dungeon_level.{png,pdf}
"""
import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import figstyle
from dungeon import MOVES, Dungeon
from pixelart import TILES, T, level_image, replay_states, tile

UP = 8  # pixel-art upscale before plotting, so viewers do not smooth the sprites
WIDTH = 4.8  # inches: \linewidth of the manuscript's one-column article
LEGEND = [("door", "door"), ("fire", "fire: −1 life per turn"), ("pit", "pit: death"),
          ("ice", "ice: may slide 1 more"), ("exit", "exit"), ("player", "the player")]


def big(img):
    return np.asarray(img.resize((img.width * UP, img.height * UP), Image.NEAREST))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/dungeon/medium/think1500_keep1/Qwen__Qwen3.5-27B.json")
    p.add_argument("--game", type=int, default=5, help="index of the game in the file")
    p.add_argument("--turn", type=int, default=8, help="the turn shown in panels B and C (0 = the first)")
    p.add_argument("--out", default="results/dungeon_level")
    args = p.parse_args()
    r = json.loads(Path(args.results).read_text())
    policy = next(iter(r["results"]))
    game = r["results"][policy]["games"][args.game]
    d = Dungeon(r["config"])
    level = d.generate(game["seed"])
    moves = [t["move"] for t in game["log"]]
    states, _ = replay_states(d, level, moves)
    assert [tuple(s.cell) for s, _ in states] == [tuple(t["cell"]) for t in game["log"]], "the replay left the saved game"

    # the text of the turn, as the chat gave it (the first turn also states the rules and the goal)
    play = d.game(level)
    turn = next(play)
    for move in moves[:args.turn]:
        turn = play.send((move, {}))
    cell = tuple(game["log"][args.turn]["cell"])
    assert tuple(turn["cell"]) == cell
    seen = set().union(*(d.visible(level, s.cell) for s, _ in states[:args.turn + 1]))
    view = level_image(d, level, d.visible(level, cell), seen)
    view.alpha_composite(TILES["player"], (cell[1] * T, cell[0] * T))
    path = [tuple(t["cell"]) for t in game["log"]] + [tuple(game["final"])]
    p_move = game["log"][args.turn]["p"]
    move = moves[args.turn]

    figstyle.apply()
    plt.rcParams["figure.constrained_layout.use"] = False
    # at the size it is printed: the text width of a one-column article
    fig = plt.figure(figsize=(WIDTH, 5.0))
    a = fig.add_axes([0.0, 0.47, 0.485, 0.45])
    b = fig.add_axes([0.515, 0.47, 0.485, 0.45])
    a.imshow(big(level_image(d, level, path=path)), interpolation="nearest")
    a.set_title(f"The level, never shown whole: {r['model'].split('/')[-1]}'s\n"
                f"path, {game['turns']} moves (shortest {game['shortest']})", fontsize=7)
    b.imshow(big(view), interpolation="nearest")
    b.set_title(f"Turn {args.turn}: in view (lit), seen before\n(dimmed), never seen (dark)", fontsize=7)
    for ax in (a, b):
        ax.axis("off")

    leg = fig.add_axes([0.0, 0.38, 1.0, 0.07])
    leg.axis("off")
    for k, (name, label) in enumerate(LEGEND):
        img = tile(name) if name != "player" else TILES["floor"].copy()
        if name == "player":
            img.alpha_composite(TILES["player"])
        x0, y0 = (k % 3) / 3, 0.52 - 0.52 * (k // 3)
        icon = leg.inset_axes([x0, y0, 0.06, 0.46])
        icon.imshow(big(img), interpolation="nearest")
        icon.axis("off")
        leg.text(x0 + 0.075, y0 + 0.23, label, transform=leg.transAxes, va="center", fontsize=6.5)

    c = fig.add_axes([0.0, 0.0, 1.0, 0.36])
    c.axis("off")
    lines = []
    for line in turn["text"].strip().split("\n"):
        lines += textwrap.wrap(line, 78, subsequent_indent="  ") or [""]
    probs = ", ".join(f"{m} {q:.2f}" for m, q in zip(MOVES, p_move))
    c.text(0.0, 1.0, f"What the LLM reads at turn {args.turn}", transform=c.transAxes, va="top", fontsize=7)
    box = c.text(0.012, 0.86, "\n".join(lines), transform=c.transAxes, va="top", ha="left", family="monospace",
                 fontsize=5.8, linespacing=1.3, bbox=dict(boxstyle="round,pad=0.5", fc="#f4f4f4", ec="#c8c8c8", lw=0.6))
    fig.canvas.draw()
    bottom = c.transAxes.inverted().transform(box.get_bbox_patch().get_window_extent())[0, 1]
    c.text(0.0, bottom - 0.04, f"Its move: {move}  (P: {probs})", transform=c.transAxes, va="top", ha="left",
           fontsize=6.5, color=figstyle.BLUE)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"))
    print("wrote", out.with_suffix(".png"), "and .pdf")


if __name__ == "__main__":
    main()
