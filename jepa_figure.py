"""The latent model on the clue game: the context (the clues up to turn t) and the target (one clue more), the LLM's
state after each, the shared encoder, the predictor conditioned on the clue in between, the belief decoder and the two
losses. The belief loss compares the decoded belief with the belief the LLM states, its own answer to the question
asked on the context prompt.

  python jepa_figure.py   # manuscript/working_doc_0/figures/jepa-clue-game.{pdf,png}
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

import figstyle

OUT = Path("manuscript/working_doc_0/figures/jepa-clue-game")
BLUE_FILL, ORANGE_FILL, GREY_FILL = "#e8f1f8", "#fdf3e2", "#f2f2f2"
INK, DIM = "#222222", "#666666"
A, B = ("A", figstyle.SKY), ("B", figstyle.PURPLE)  # the sources' badges


def rbox(ax, x0, y0, w, h, fc="white", ec="#aaaaaa", lw=0.6, ls="-", r=3):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                ls=ls))


def text(ax, x, y, s, size=6, color=INK, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    ax.text(x, y, s, fontsize=size, color=color, **kw)


def arrow(ax, start, end, color=INK, lw=0.7):
    ax.annotate("", end, start, arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, shrinkA=0, shrinkB=0,
                                                mutation_scale=6))


def path(ax, points, color, lw=0.7):
    """Polyline with an arrowhead on its last segment."""
    xs, ys = zip(*points[:-1])
    ax.add_line(Line2D(xs, ys, color=color, lw=lw, solid_capstyle="round"))
    arrow(ax, points[-2], points[-1], color, lw)


def badge(ax, x, y, letter, color, r=4.3):
    ax.add_patch(Circle((x, y), r, fc=color, ec="none"))
    text(ax, x, y, letter, 5.5, "white", fontweight="bold")


def squares(ax, x0, x1, y, n, size, fc, ec, hatch=None):
    pitch = (x1 - x0 - size) / (n - 1)
    for i in range(n):
        ax.add_patch(Rectangle((x0 + i * pitch, y), size, size, fc=fc, ec=ec, lw=0.5, hatch=hatch))


def prompt(ax, cx, y0, clues, last):
    """The clues of the game's prompt so far (the game itself is drawn in games_figure.py). clues: (badge, text)
    rows; last: a clue just read, highlighted, or None for a dashed placeholder."""
    x0 = cx - 70
    rbox(ax, x0, y0, 140, 43, GREY_FILL, "#aaaaaa", 0.8, r=4)
    for k, row in enumerate(clues + [last]):
        y = y0 + 5 + 12 * k
        if row is None:
            rbox(ax, x0 + 16, y, 119, 9.5, "none", "#888888", 0.6, ls=(0, (2, 1.5)), r=2)
            text(ax, x0 + 75.5, y + 4.75, "next clue not seen", 6, DIM)
            continue
        (letter, color), s = row
        new = row is last
        badge(ax, x0 + 9.5, y + 4.75, letter, color)
        rbox(ax, x0 + 16, y, 119, 9.5, ORANGE_FILL if new else "white", figstyle.ORANGE if new else "#aaaaaa",
             0.8 if new else 0.5, r=2)
        text(ax, x0 + 21, y + 4.75, s, 6, ha="left")


def main():
    figstyle.apply()
    plt.rcParams.update({"figure.constrained_layout.use": False, "hatch.linewidth": 0.6})
    xlim, ylim = (-18, 353), (342, 36)  # one unit is one point
    fig = plt.figure(figsize=((xlim[1] - xlim[0]) / 72, (ylim[0] - ylim[1]) / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")

    cx, tx = 90, 263  # column centres: context, target
    mid = (cx + tx) / 2
    clues = [(A, "is red"), (B, "is not big")]
    text(ax, cx, 43, "Context: the game up to turn $t$", 8)
    text(ax, tx, 43, "Target: the game up to turn $t+1$", 8)
    prompt(ax, cx, 52, clues, None)
    prompt(ax, tx, 52, clues, (A, "is not round"))
    text(ax, cx, 103, "the prompt after $t$ clues", 6, DIM)
    text(ax, tx, 103, "the same prompt, one clue more", 6, DIM)
    for x in (cx, tx):
        arrow(ax, (x, 108), (x, 118))
        squares(ax, x - 52, x + 52, 119, 16, 5.5, "#d0d0d0", "#999999")
    text(ax, cx, 132, "LLM state $h_t$", 7)
    text(ax, tx, 132, "LLM state $h_{t+1}$", 7)
    text(ax, mid, 142, "activations of the chosen layer, cached after each turn", 6, DIM)

    # the shared encoder and the latents
    for x, fill, edge, note in ((cx, BLUE_FILL, figstyle.BLUE, "learns by backpropagation"),
                                (tx, ORANGE_FILL, figstyle.ORANGE, "same weights")):
        arrow(ax, (x, 148), (x, 156))
        rbox(ax, x - 59, 156, 118, 23, fill, edge, 1.0, r=4)
        text(ax, x, 163, "Encoder $E$", 7)
        text(ax, x, 172, note, 6, DIM)
        arrow(ax, (x, 179), (x, 189))
        squares(ax, x - 39, x + 39, 190, 8, 8, edge, "none")
    ax.add_line(Line2D([cx + 59, tx - 59], [168, 168], color="#999999", lw=0.6, ls=(0, (3, 2))))
    text(ax, mid, 163, "shared weights", 6, DIM)
    text(ax, cx, 204, "latent $z_t$", 7)
    text(ax, tx, 204, "latent $z_{t+1}$", 7)

    # the action and the predictor
    arrow(ax, (cx, 209), (cx, 219))
    rbox(ax, cx - 60, 219, 120, 17, "white", figstyle.ORANGE, 0.8, r=3)
    text(ax, cx - 55, 227.5, "+ action $a_t$:", 6.5, ha="left")
    badge(ax, cx - 10, 227.5, *A)
    rbox(ax, cx - 3, 222, 60, 11, ORANGE_FILL, figstyle.ORANGE, 0.6, r=2)
    text(ax, cx + 1, 227.5, "is not round", 6, ha="left")
    arrow(ax, (cx, 236), (cx, 246))
    rbox(ax, cx - 56, 246, 112, 23, BLUE_FILL, figstyle.BLUE, 1.0, r=4)
    text(ax, cx, 253, "Predictor $P$", 7)
    text(ax, cx, 262, r"$\hat z_{t+1} = z_t + P(z_t, a_t)$", 6, DIM)
    arrow(ax, (cx, 269), (cx, 279))
    squares(ax, cx - 39, cx + 39, 280, 8, 8, "white", figstyle.BLUE, hatch="////")
    text(ax, cx, 294, r"predicted latent $\hat z_{t+1}$", 7)

    # the losses
    ly = 306
    rbox(ax, -6, ly, 137, 34, "#ebebeb", "#777777", 1.0, r=4)
    text(ax, 62, ly + 11, "Belief loss", 7.5)
    text(ax, 62, ly + 23, r"$\hat b_t = R(z_t)$ vs. the stated belief $b_t$", 6, DIM)
    rbox(ax, 145, ly, 200, 34, "#ebebeb", "#777777", 1.0, r=4)
    text(ax, 245, ly + 11, "Prediction loss", 7.5)
    text(ax, 245, ly + 23, r"$\|\hat z_{t+1} - z_{t+1}\|^2$ over rollouts, plus VICReg", 6, DIM)

    # wires: the prediction against the target, which gives no gradient
    path(ax, [(cx, 299), (cx, 302), (170, 302), (170, ly)], figstyle.BLUE)
    path(ax, [(tx, 209), (tx, ly)], figstyle.ORANGE)
    for d in (-2, 2):
        ax.add_line(Line2D([tx - 3 + d, tx + 3 + d], [263, 257], color=figstyle.ORANGE, lw=0.8))
    text(ax, tx + 7, 260, "stop gradient", 6, DIM, ha="left")
    # the belief decoder reads z_t; its target is the LLM's answer to the question, asked on the context prompt
    path(ax, [(cx - 40, 194), (14, 194), (14, ly)], figstyle.BLUE)
    text(ax, 21, 250, "belief decoder $R$", 6, figstyle.BLUE, rotation=90)
    path(ax, [(cx - 70, 73), (0, 73), (0, ly)], "#888888")
    text(ax, -6, 180, "stated belief $b_t$: the LLM's answer to the question", 6, DIM, rotation=90)

    figstyle.save(fig, OUT)


if __name__ == "__main__":
    main()
