"""One level of each difficulty of the dungeon, drawn whole with its shortest safe path, for the paper.

The levels come from Dungeon.generate at each preset of dungeon.json (easy, medium, hard), so the figure shows
what the difficulties change: the rooms, the hazards, the lives and the torch. The path drawn is the shortest one
that avoids every hazard (green: start; red: the exit), which every level is generated to have.

The default seeds give a shortest path at each difficulty's median (8, 15, 17 moves); the medium level is the
one of dungeon_figure.py (seed 5), so the two figures show the same level.

  python -m figures.dungeon_levels [--seeds 0 5 18]   # results/dungeon_levels.{png,pdf}
"""
import argparse
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import ImageDraw

from figures import figstyle
from games.dungeon import MOVES, Dungeon, step
from figures.dungeon_figure import LEGEND, WIDTH, big
from games.pixelart import TILES, level_image, tile

DIFFICULTIES = ("easy", "medium", "hard")


def safe_path(d, level):
    """The cells of a shortest path from the start to the exit through cells that are not hazards (Dungeon.shortest
    gives its length only)."""
    parent, queue = {level.start: None}, deque([level.start])
    while queue:
        c = queue.popleft()
        if c == level.exit:
            break
        for move in MOVES:
            nxt = step(c, move)
            if nxt not in parent and d.inside(nxt) and d.safe(level[nxt]):
                parent[nxt] = c
                queue.append(nxt)
    path, c = [], level.exit
    while c is not None:
        path.append(c)
        c = parent[c]
    return path[::-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs=3, default=[0, 5, 18], help="one level seed per difficulty")
    p.add_argument("--out", default="results/dungeon_levels")
    args = p.parse_args()

    figstyle.apply()
    plt.rcParams["figure.constrained_layout.use"] = False
    fig = plt.figure(figsize=(WIDTH, 2.6))
    dungeons = [Dungeon(difficulty=name) for name in DIFFICULTIES]
    # panels side by side, each as wide as its grid, the tallest grid filling the panel row
    gap, total = 0.025, sum(d.cols for d in dungeons)
    scale = (1 - gap * (len(dungeons) - 1)) / total
    x = 0.0
    for d, seed in zip(dungeons, args.seeds):
        level = d.generate(seed)
        ax = fig.add_axes([x, 0.2, d.cols * scale, 0.66], anchor="N")  # the image hangs from the top of the row
        ax.imshow(big(level_image(d, level, path=safe_path(d, level))), interpolation="nearest")
        ax.axis("off")
        pl, lv = d.config["level"], d.config["level"]["density"]
        fig.text(x, 0.99, f"{d.difficulty.capitalize()}: {pl['rooms'][0]}×{pl['rooms'][1]} rooms, {d.rows}×{d.cols} cells\n"
                 f"fire {lv['fire']:.0%}, pits {lv['pit']:.0%}, ice {lv['ice']:.0%} "
                 f"(slides {d.terrain['ice']['slip_probability']:.0%})\n"
                 f"{d.player['lives']} {'life' if d.player['lives'] == 1 else 'lives'}, torch {d.player['turns']} "
                 f"turns, shortest {d.shortest(level)}", fontsize=5.6, va="top", ha="left", linespacing=1.25)
        x += d.cols * scale + gap

    leg = fig.add_axes([0.0, 0.0, 1.0, 0.15])
    leg.axis("off")
    for k, (name, label) in enumerate(LEGEND[:-1] + [("path", "shortest safe path, from the start")]):
        img = tile(name) if name != "path" else TILES["floor"].copy()
        if name == "path":  # as level_image draws it: the line and the start's green square
            draw = ImageDraw.Draw(img)
            draw.line([(2, 8), (13, 8)], fill=(255, 236, 39, 255), width=2)
            draw.rectangle([1, 6, 5, 10], fill=(0, 228, 54, 255))
        x0, y0 = (k % 3) / 3, 0.52 - 0.52 * (k // 3)
        icon = leg.inset_axes([x0, y0, 0.06, 0.46])
        icon.imshow(big(img), interpolation="nearest")
        icon.axis("off")
        leg.text(x0 + 0.075, y0 + 0.23, label, transform=leg.transAxes, va="center", fontsize=6.5)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"))
    print("wrote", out.with_suffix(".png"), "and .pdf")


if __name__ == "__main__":
    main()
