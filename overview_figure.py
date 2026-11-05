"""The scheme of the method: a game in turns, the LLM's state and stated belief at each turn, the Bayesian belief
the game gives, and the latent model trained on the transitions.

Three lanes, two turns. The game lane has the hidden state and the Bayesian observer's belief before and after the
action. The LLM lane has its internal state h and stated belief b before and after it reads the action. The latent
lane has the encoder, the predictor conditioned on the action, and the belief decoder.

  python overview_figure.py   # manuscript/working_doc_0/figures/overview.{pdf,png}
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import figstyle

OUT = Path("manuscript/working_doc_0/figures/overview")
WIDTH = 6.5
GAME, LLM, LATENT = "#efefef", "#cfe3f1", "#fdebd0"
EDGE = "#555555"


def box(ax, cx, cy, w, h, text, fc, fontsize=7, ls="-"):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1",
                                fc=fc, ec=EDGE, lw=0.6, ls=ls))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, linespacing=1.3)


def arrow(ax, start, end, text=None, color="#222222", ls="-", dy=1.6, fontsize=6.5, va="bottom"):
    ax.annotate("", end, start, arrowprops=dict(arrowstyle="-|>", lw=0.7, color=color, ls=ls,
                                                shrinkA=0, shrinkB=0))
    if text:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + dy, text, ha="center", va=va,
                fontsize=fontsize, color="#444444")


def main():
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(WIDTH, 2.35))
    ax.set_xlim(0, 82)
    ax.set_ylim(0, 40)
    ax.axis("off")

    y_game, y_llm, y_lat = 33, 20.5, 7
    x_t, x_t1 = 30, 72
    bw, bh = 18, 7.5

    # lane names
    for y, name in ((y_game, "the game"), (y_llm, "the LLM"), (y_lat, "the latent\nmodel")):
        ax.text(0, y, name, ha="left", va="center", fontsize=7.5, fontweight="bold", color="#333333")

    # turn headers
    ax.text(x_t, 39.5, "turn $t$", ha="center", va="top", fontsize=7.5, color="#444444")
    ax.text(x_t1, 39.5, "turn $t+1$", ha="center", va="top", fontsize=7.5, color="#444444")

    # the game
    box(ax, 13, y_game, 8, 5.5, "hidden\nstate", GAME, fontsize=6.5)
    box(ax, x_t, y_game, bw, bh, "Bayesian belief $b^*_t$\ngiven the clues so far", GAME)
    box(ax, x_t1, y_game, bw, bh, "Bayesian belief $b^*_{t+1}$", GAME)
    arrow(ax, (17, y_game), (x_t - bw / 2, y_game))
    arrow(ax, (x_t + bw / 2, y_game), (x_t1 - bw / 2, y_game), "Bayes update with $a_t$", dy=1.2)

    # the LLM
    box(ax, x_t, y_llm, bw, bh, "LLM state $h_t$\nstated belief $b_t$", LLM)
    box(ax, x_t1, y_llm, bw, bh, "LLM state $h_{t+1}$\nstated belief $b_{t+1}$", LLM)
    arrow(ax, (x_t + bw / 2, y_llm), (x_t1 - bw / 2, y_llm),
          "action $a_t$: a clue, or a move\nand what it shows", dy=1.0, fontsize=6.2)
    # the gap, Bayes against the LLM
    for x in (x_t, x_t1):
        ax.annotate("", (x, y_game - bh / 2), (x, y_llm + bh / 2),
                    arrowprops=dict(arrowstyle="<|-|>", lw=0.6, color=figstyle.VERMILLION, shrinkA=0, shrinkB=0))
    ax.text(x_t + 1, (y_game + y_llm) / 2, "Bayes gap", ha="left", va="center", fontsize=6,
            color=figstyle.VERMILLION)

    # the latent model
    box(ax, x_t, y_lat, bw, bh, "latent $z_t = E(h_t)$", LATENT)
    x_hat = (x_t + x_t1) / 2 + 1
    box(ax, x_hat, y_lat, 10, bh, "$\\hat z_{t+1}$", LATENT, ls="--")
    box(ax, x_t1, y_lat, bw, bh, "latent $z_{t+1} = E(h_{t+1})$", LATENT)
    arrow(ax, (x_t + bw / 2, y_lat), (x_hat - 5, y_lat), "predictor\n$P(z_t, a_t)$", dy=1.2)
    ax.text((x_hat + 5 + x_t1 - bw / 2) / 2, y_lat, "$\\approx$", ha="center", va="center", fontsize=9)
    # encoder and belief decoder
    arrow(ax, (x_t - 4, y_llm - bh / 2), (x_t - 4, y_lat + bh / 2), color=figstyle.BLUE)
    ax.text(x_t - 4.8, (y_llm + y_lat) / 2, "encoder $E$", ha="right", va="center", fontsize=6,
            color=figstyle.BLUE)
    arrow(ax, (x_t + 4, y_lat + bh / 2), (x_t + 4, y_llm - bh / 2), color=figstyle.BLUE)
    ax.text(x_t + 4.8, (y_llm + y_lat) / 2, "decoder $R$", ha="left", va="center", fontsize=6,
            color=figstyle.BLUE)
    arrow(ax, (x_t1, y_llm - bh / 2), (x_t1, y_lat + bh / 2), color=figstyle.ORANGE)
    ax.text(x_t1 + 1.2, (y_llm + y_lat) / 2, "encoder $E$,\nno gradient", ha="left", va="center", fontsize=6,
            color=figstyle.ORANGE)

    for issue in figstyle.overlaps(fig):
        print("WARNING overlap:", issue)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}")
    print("saved", OUT)


if __name__ == "__main__":
    main()
