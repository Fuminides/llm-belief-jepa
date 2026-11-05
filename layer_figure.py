"""One Gated DeltaNet layer of Qwen3.5-2B, and why its recurrent memory S summarises the state that the residual
stream spreads over the tokens.

A: the layer. Its only memory of earlier tokens is S, updated at every token by the gated delta rule from a key
and a value made from that token's residual stream, and read by the next token as Sᵀq. The residual stream at one
token holds only what that token retrieved. B: the LLM's belief read by a linear probe from the residual stream
after each layer (last token or turn mean, results/qwen3.5-2b/probes.json), from the running mean of the turn
means, and from S (results/qwen3.5-2b/memory_probes.json).

  python layer_figure.py   # results/qwen3.5-2b-memory/layer_figure.{png,pdf}
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

import figstyle
from model_figure import LAYER_TYPES, MEM, WIDTH

DELTANET, ATTENTION = "#cfe3f1", figstyle.ORANGE


def box(ax, x0, x1, y0, y1, text, fc="#f2f2f2", fontsize=6.5):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=fc, ec="#555555", lw=0.5))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=fontsize, linespacing=1.3)


def arrow(ax, start, end, elbow=None, color="#222222"):
    """A straight arrow, or an elbow: 'up-right' leaves upwards and arrives from the left, and so on."""
    style = {None: "arc3,rad=0", "up-right": "angle,angleA=90,angleB=0,rad=0",
             "right-down": "angle,angleA=0,angleB=90,rad=0"}[elbow]
    ax.annotate("", end, start, arrowprops=dict(arrowstyle="-|>", lw=0.7, color=color, shrinkA=0, shrinkB=0,
                                                connectionstyle=style))


def layer(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 33)
    ax.axis("off")
    y = 6  # the residual stream
    ax.plot([0, 100], [y, y], color="#222222", lw=1.2, solid_capstyle="butt", zorder=1)
    arrow(ax, (97, y), (100, y))
    for x in (80, 97):
        ax.plot(x, y, marker="o", ms=8, mfc="white", mec="#222222", mew=0.8, zorder=2)
        ax.text(x, y, "+", ha="center", va="center", fontsize=8, zorder=3)
    ax.text(0, y - 2, "residual stream\n(2048 per token)", ha="left", va="top", fontsize=6.5)
    # the token mixer
    box(ax, 7, 15, 12, 18, "RMSNorm")
    box(ax, 18, 27, 12, 18, "q, k, v\nprojections")
    box(ax, 29.5, 41, 12, 18, "causal conv\n4 tokens, SiLU")
    box(ax, 44, 68, 10.5, 19.5, "recurrent memory S\n16 heads × 128 × 128\nS ← α (I − β k kᵀ) S + β k vᵀ\n"
                                "o = Sᵀ q", fc=DELTANET)
    box(ax, 71, 79, 12, 18, "gate, norm,\nout proj")
    box(ax, 18, 34, 22.5, 27.5, "gates: α forgets, β writes")
    arrow(ax, (3, y), (7, 15), "up-right")
    arrow(ax, (15, 15), (18, 15))
    arrow(ax, (27, 15), (29.5, 15))
    arrow(ax, (41, 15), (44, 15))
    arrow(ax, (68, 15), (71, 15))
    arrow(ax, (79, 15), (80, y + 1.1), "right-down")
    arrow(ax, (11.5, 18), (18, 25), "up-right")
    arrow(ax, (34, 25), (50, 19.5), "right-down")
    ax.text(35, 10.5, "conv state:\nthe last 3 tokens", ha="center", va="top", fontsize=6, color="#444444")
    # the MLP
    box(ax, 85, 94, 12, 18, "norm, MLP")
    ax.text(89.5, 10.5, "2048 → 6144\n→ 2048", ha="center", va="top", fontsize=6, color="#444444")
    arrow(ax, (82, y), (85, 15), "up-right")
    arrow(ax, (94, 15), (97, y + 1.1), "right-down")
    # why S summarises and one residual vector does not
    ax.text(56, 29.5, "the layer's only memory of earlier tokens: a decayed sum, over every token so far,\n"
            "of k vᵀ with k and v made from that token's residual stream", ha="center", va="bottom",
            fontsize=6.5, color=figstyle.BLUE, linespacing=1.3)
    arrow(ax, (56, 29.2), (56, 19.6), color=figstyle.BLUE)
    ax.text(60, 3.6, "at one token the residual holds what that token retrieved (Sᵀq; the KV cache in the\n"
            "attention layers) and what the MLP made of it: what a clue wrote stays at the clue's position",
            ha="center", va="top", fontsize=6.5, color=figstyle.VERMILLION, linespacing=1.3)
    arrow(ax, (60, 3.9), (60, y - 0.6), color=figstyle.VERMILLION)


def main():
    probes = json.loads(Path("results/qwen3.5-2b/probes.json").read_text())
    memory = json.loads(Path("results/qwen3.5-2b/memory_probes.json").read_text())
    n = len(LAYER_TYPES)
    after = np.arange(n)  # probes.json index i is the residual after layer i - 1 (index 0: the embeddings)
    last = probes["acts"]["belief"][1:]
    mean = probes["pool"]["belief"][1:]
    mem_layers = memory["memory_layers"]
    states = memory["states"]

    figstyle.apply()
    plt.rcParams["figure.constrained_layout.use"] = False
    fig = plt.figure(figsize=(WIDTH, 4.3))
    a = fig.add_axes([0.0, 0.50, 1.0, 0.46])
    b = fig.add_axes([0.08, 0.09, 0.50, 0.33])
    layer(a)
    a.set_title("One Gated DeltaNet layer of Qwen3.5-2B (18 of its 24 layers; the other 6 are full attention "
                "with a KV cache)", loc="left", x=0.03, fontsize=7.5)

    b.plot(after, last, color=figstyle.GREY, ls=":", marker=".", ms=3, label="residual, last token of the turn")
    b.plot(after, mean, color=figstyle.GREY, ls="--", marker=".", ms=3, label="residual, mean over the turn")
    b.plot(mem_layers, states["residual, mean of turns so far"]["belief"], color=figstyle.VERMILLION, ls="--",
           marker="^", label="residual, running mean of the turns")
    b.plot(mem_layers, states["recurrent memory S"]["belief"], color=figstyle.BLUE, marker="o",
           label="recurrent memory S, one position")
    for i, kind in enumerate(LAYER_TYPES):  # the kind of each layer, along the axis
        b.add_patch(Rectangle((i - 0.5, 0.0), 1, 0.04, fc=ATTENTION if kind == "attention" else DELTANET,
                              ec="white", lw=0.3))
    b.set_xlim(-0.6, n - 0.4)
    b.set_ylim(0, 1)
    b.set_xticks([0, 4, 8, 12, 16, 20, 23])
    b.set_xlabel("after layer")
    b.set_ylabel("belief R²")
    b.set_title("Reading the belief from the residual stream, by depth, and from S", loc="left")
    handles, labels = b.get_legend_handles_labels()
    handles += [Patch(fc=DELTANET, ec="#555555", lw=0.4), Patch(fc=ATTENTION, ec="#555555", lw=0.4)]
    labels += ["Gated DeltaNet layer", "full attention layer"]
    b.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6.5)
    for letter, (x, yy) in zip("AB", ((0.0, 0.965), (0.0, 0.44))):
        fig.text(x, yy, letter, fontsize=10, fontweight="bold", va="bottom")

    for ext in ("png", "pdf"):
        fig.savefig(MEM / f"layer_figure.{ext}")
    print("wrote", MEM / "layer_figure.png", "and .pdf")


if __name__ == "__main__":
    main()
