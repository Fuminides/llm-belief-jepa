"""The two games, one row each, in the same three steps: the game, what the LLM reads, what it answers against the
reference. A: the clue game on one real episode of Qwen3.5-2B (data/qwen3.5-2b-memory), the belief after each clue
next to the Bayesian posterior. B: the dungeon at one turn of a saved game, played again from its moves
(pixelart.py): the level with the player's path, what it has seen so far, the text it reads and its move.

  python -m figures.games_figure [--episode 10] [--game 5] [--turn 8]   # manuscript/working_doc_0/figures/games.{pdf,png}
"""
import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from figures import figstyle
from games import game
from games.dungeon import MOVES, Dungeon
from figures.dungeon_figure import LEGEND, big
from figures.jepa_figure import DIM, INK, ORANGE_FILL
from games.pixelart import TILES, level_image, replay_states, tile

OUT = Path("manuscript/working_doc_0/figures/games")
WIDTH, HEIGHT = 6.75, 5.0  # inches; printed at the text width
PANELS, MARGIN = {"A": "#eaf2f9", "B": "#fdf4e6"}, 8  # each game in a tinted rounded panel; the margin inside it, points  # inches; the width is the manuscript's text width
BADGES = {"Alice": ("A", figstyle.SKY), "Bob": ("B", figstyle.PURPLE)}


def rbox(ax, x0, y0, w, h, fc="white", ec="#aaaaaa", lw=0.6, ls="-", r=3, **kw):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                ls=ls, **kw))


def text(ax, x, y, s, size=6, color=INK, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    return ax.text(x, y, s, fontsize=size, color=color, **kw)


def badge(ax, x, y, source, r=4.3):
    letter, color = BADGES[source]
    ax.add_patch(Circle((x, y), r, fc=color, ec="none"))
    text(ax, x, y, letter, 5.5, "white", fontweight="bold")


def header(ax, x, s):
    text(ax, x, 4, s, 7, INK, va="top", linespacing=1.2)


def bar(ax, x0, y0, w, h, p, ref=None, ruled_out=False):
    """A probability as a filled share of a grey bar; ref, the reference probability, as a black tick."""
    ax.add_patch(Rectangle((x0, y0), w, h, fc="#e6e6e6", ec="none", hatch="////" if ruled_out else None,
                           lw=0 if not ruled_out else 0.3))
    ax.add_patch(Rectangle((x0, y0), w * p, h, fc=figstyle.BLUE, ec="none"))
    if ref is not None:
        ax.add_line(Line2D([x0 + w * ref] * 2, [y0 - 1.5, y0 + h + 1.5], color=INK, lw=1.0, solid_capstyle="butt"))


def clue_panel(ax, episode, turns):
    """Row A in points, 468 wide. episode: (clue texts, llm beliefs, bayes beliefs) over turns 0..turns."""
    clues, llm, bayes = episode
    text(ax, 0, 2, "A", 10, INK, fontweight="bold", ha="left", va="top")
    header(ax, 66, "The game: a secret object\nand two sources")
    header(ax, 222, "The clues")
    header(ax, 392, "LLM beliefs against Bayes-optimal")

    # the game: the object with its four binary properties, the sources with their reliabilities
    x0, y0 = 6, 26
    rbox(ax, x0, y0, 120, 100, "white", "#aaaaaa", 0.8, r=4)
    rbox(ax, x0 + 8, y0 + 8, 22, 19, "white", "#333333", 0.9, r=3)
    text(ax, x0 + 19, y0 + 17.5, "?", 10)
    text(ax, x0 + 19, y0 + 32, "secret object", 5.5, DIM)
    for k, (attr, (v0, v1)) in enumerate(game.ATTRIBUTES.items()):
        y = y0 + 8 + 8 * k
        rbox(ax, x0 + 38, y, 76, 6, "white", "#aaaaaa", 0.5, r=1.5)
        text(ax, x0 + 41, y + 3, f"{attr}: {v0} or {v1}", 5.5, "#444444", ha="left")
    for k, (name, rel) in enumerate(game.SOURCES.items()):
        y = y0 + 52 + 16 * k
        badge(ax, x0 + 14, y, name)
        text(ax, x0 + 23, y - 3, name, 6, ha="left")
        text(ax, x0 + 23, y + 4, f"right {round(100 * rel)}% of the time", 5.5, DIM, ha="left")
    text(ax, x0 + 60, y0 + 92, "both stated in the intro; the object\nitself is never revealed", 5.5, DIM,
         linespacing=1.2)

    # what it reads: the clues in turn, then the question
    rows = [26 + 14 * k for k in range(turns + 1)]  # row k: the state after k clues (row 0: before any)
    for k, (source, s) in enumerate(clues, start=1):
        y = rows[k]
        text(ax, 140, y + 4.5, f"turn {k}", 5.5, DIM, ha="left")
        badge(ax, 170, y + 4.5, source)
        rbox(ax, 177, y, 128, 9.5, "white", "#aaaaaa", 0.5, r=2)
        text(ax, 181, y + 4.75, s, 6, ha="left")
    y = rows[-1] + 18
    rbox(ax, 140, y, 165, 24, ORANGE_FILL, figstyle.ORANGE, 0.8, r=3)
    text(ax, 146, y + 7.5, "Question: Is the object big or small?", 6, ha="left")
    text(ax, 146, y + 17, "Answer: The object is", 6, ha="left")
    ax.add_line(Line2D([218, 232], [y + 20.5, y + 20.5], color=INK, lw=0.6))

    # what it answers: P(first value) per property after each clue, the Bayesian posterior as a tick
    cx0, cw, bw = 362, 27, 24
    for j, (v0, _) in enumerate(game.ATTRIBUTES.values()):
        text(ax, cx0 + cw * j + bw / 2, rows[0] - 5, f"P({v0})", 5.5, DIM)
    for k, y in enumerate(rows):
        text(ax, 356, y + 4.5, "before any clue" if k == 0 else f"after clue {k}", 5.5, DIM, ha="right")
        for j in range(4):
            bar(ax, cx0 + cw * j, y, bw, 9, llm[k, j], bayes[k, j])
    y = rows[-1] + 12
    for x, s in ((cx0, "0"), (cx0 + bw, "1")):
        text(ax, x, y, s, 5, DIM)
    y += 11
    ax.add_patch(Rectangle((318, y - 3), 8, 6, fc=figstyle.BLUE, ec="none"))
    text(ax, 329, y, "Qwen3.5-2B's belief", 5.5, ha="left")
    ax.add_line(Line2D([392, 392], [y - 4, y + 4], color=INK, lw=1.0))
    text(ax, 396, y, "Bayes", 5.5, ha="left")
    text(ax, 392, y + 16, "Bob and Alice disagree on the size:\nBayes follows Alice, the 2B stays with Bob", 5.5, DIM,
         linespacing=1.2)


def dungeon_panel(fig, ax, d, level, game_, states, turn_text, k, model):
    """Row B. ax: an overlay axes over the row, in points (468 wide); the images are axes of their own."""
    def inset(x0, y0, w, h):
        """An axes over the rectangle (x0, y0, w, h) of ax, in its points."""
        (fx0, fy0), (fx1, fy1) = fig.transFigure.inverted().transform(ax.transData.transform([(x0, y0 + h), (x0 + w, y0)]))
        return fig.add_axes([fx0, fy0, fx1 - fx0, fy1 - fy0])

    text(ax, 0, 2, "B", 10, INK, fontweight="bold", ha="left", va="top")
    header(ax, 66, f"The dungeon and\n{model}'s path")
    header(ax, 211, "Subjective point of view")
    header(ax, 379, f"Turn {k} context")

    cell = tuple(game_["log"][k]["cell"])
    seen = set().union(*(d.visible(level, s.cell) for s, _ in states[:k + 1]))
    view = level_image(d, level, d.visible(level, cell), seen)
    view.alpha_composite(TILES["player"], (cell[1] * 16, cell[0] * 16))
    path = [tuple(t["cell"]) for t in game_["log"]] + [tuple(game_["final"])]
    y_img, h_img = 24, 133  # points from the row's top
    for x0, img in ((0, level_image(d, level, path=path)), (145, view)):
        ia = inset(x0, y_img, h_img, h_img)
        ia.imshow(big(img), interpolation="nearest")
        ia.axis("off")
    y = y_img + h_img + 8
    for j, (name, label) in enumerate(LEGEND):
        img = tile(name) if name != "player" else TILES["floor"].copy()
        if name == "player":
            img.alpha_composite(TILES["player"])
        x0, yy = 95 * (j % 3), y + 13 * (j // 3)
        ia = inset(x0, yy, 10, 10)
        ia.imshow(big(img), interpolation="nearest")
        ia.axis("off")
        text(ax, x0 + 14, yy + 5, label, 6, ha="left")

    # the text of the turn and the move
    lines = []
    for line in turn_text.strip().split("\n"):
        lines += textwrap.wrap(line, 54, subsequent_indent="  ") or [""]
    box = text(ax, 296, y_img, "\n".join(lines), 5.2, INK, ha="left", va="top", family="monospace",
               linespacing=1.25, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#c8c8c8", lw=0.6))
    fig.canvas.draw()
    bottom = ax.transData.inverted().transform(box.get_bbox_patch().get_window_extent())[:, 1].max()
    y = bottom + 10
    text(ax, 296, y, "Actions:", 6, ha="left")
    next_cell = {m: level[cell[0] + dr, cell[1] + dc] for m, (dr, dc) in MOVES.items()}  # one step away
    for j, (m, p) in enumerate(zip(MOVES, game_["log"][k]["p"])):
        yy = y + 9 + 12 * j
        text(ax, 322, yy + 4.5, m, 6, ha="right")
        ruled_out = next_cell[m] == "wall"
        bar(ax, 327, yy, 60, 9, p, ruled_out=ruled_out)
        text(ax, 391, yy + 4.5, f"{p:.2f}", 5.5, DIM, ha="left")
        what = {"wall": "a wall: ruled out", "floor": "an empty cell"}.get(next_cell[m], next_cell[m])
        text(ax, 408, yy + 4.5, what, 5.5, DIM, ha="left")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/qwen3.5-2b-memory/shard_000000.npz")
    p.add_argument("--episode", type=int, default=10, help="index of the clue-game episode in the shard")
    p.add_argument("--clues", type=int, default=3, help="clues shown of the episode")
    p.add_argument("--results", default="results/dungeon/medium/think1500_keep1/Qwen__Qwen3.5-27B.json")
    p.add_argument("--game", type=int, default=5, help="index of the game in the file")
    p.add_argument("--turn", type=int, default=8, help="the turn shown (0 = the first)")
    args = p.parse_args()

    z = np.load(args.data)
    acts = z["actions"][args.episode, :args.clues]
    clues = [(game.SOURCE_NAMES[game.ACTIONS[a][3]], game.clue_text(a).strip()) for a in acts]
    episode = (clues, z["llm_belief"][args.episode, :args.clues + 1], z["bayes_belief"][args.episode, :args.clues + 1])

    r = json.loads(Path(args.results).read_text())
    game_ = r["results"][next(iter(r["results"]))]["games"][args.game]
    d = Dungeon(r["config"])
    level = d.generate(game_["seed"])
    moves = [t["move"] for t in game_["log"]]
    states, _ = replay_states(d, level, moves)
    assert [tuple(s.cell) for s, _ in states] == [tuple(t["cell"]) for t in game_["log"]], "the replay left the saved game"
    play = d.game(level)
    turn = next(play)
    for move in moves[:args.turn]:
        turn = play.send((move, {}))
    assert tuple(turn["cell"]) == tuple(game_["log"][args.turn]["cell"])

    figstyle.apply()
    plt.rcParams.update({"figure.constrained_layout.use": False, "hatch.linewidth": 0.5})
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    rows = {"A": (3.0, 2.0), "B": (0.0, 2.9)}  # bottom and height in inches
    axes = {}
    for name, (bottom, height) in rows.items():
        ax = fig.add_axes([0, bottom / HEIGHT, 1, height / HEIGHT])
        w, h = WIDTH * 72, height * 72  # the row in points; the content is drawn inside the margin
        ax.set_xlim(-MARGIN, w - MARGIN)
        ax.set_ylim(h - MARGIN, -MARGIN)
        ax.axis("off")
        rbox(ax, -MARGIN + 0.5, -MARGIN + 0.5, w - 1, h - 1, PANELS[name], "#bbbbbb", 0.6, r=5, zorder=-1)
        axes[name] = ax
    clue_panel(axes["A"], episode, args.clues)
    dungeon_panel(fig, axes["B"], d, level, game_, states, turn["text"], args.turn, r["model"].split("/")[-1])
    figstyle.save(fig, OUT)


if __name__ == "__main__":
    main()
