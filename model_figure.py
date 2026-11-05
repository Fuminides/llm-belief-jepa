"""Where the belief is read from Qwen3.5-2B, where it is written, and why reading works and writing does not.

A: the 24 layers, 18 Gated DeltaNet (each with a recurrent memory S and a convolution state) and 6 full attention
(a KV cache), with what the experiments touch: S read by probes and the latent model, the residual stream shifted
on a clue's value word. B: the belief read by a linear probe from one layer's S or residual stream
(results/qwen3.5-2b/memory_probes.json). C: the share of a clue's effect on the belief recovered when the state
after the clue is written into the model: S of one layer (latent_edit.json), a shift of the residual stream
(residual_edit.json), and the real cache copied part by part (cache_patch.json).

  python model_figure.py   # results/qwen3.5-2b-memory/model_figure.{png,pdf} (B and C) and model_stack.{png,pdf} (A)
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

import figstyle

MEM = Path("results/qwen3.5-2b-memory")
WIDTH = 6.5  # inches: the manuscript's text width
LAYER_TYPES = (["linear"] * 3 + ["attention"]) * 6  # Qwen3.5-2B: full attention every fourth layer
READ_LAYERS = (2, 9, 17, 22)  # S read by the probes
LATENT_LAYER = 14  # S encoded by the anchored latent model
EDIT_LAYER = 3  # residual stream shifted on the clue's value word


def stack(ax):
    """The layers as boxes, bottom to top, with what is read and what is written."""
    ax.set_xlim(-2.6, 4.6)
    ax.set_ylim(-1.6, 25.4)
    ax.axis("off")
    for i, kind in enumerate(LAYER_TYPES):
        fc = figstyle.ORANGE if kind == "attention" else "#cfe3f1"
        ax.add_patch(Rectangle((0, i), 1.6, 0.8, fc=fc, ec="#555555", lw=0.4))
        ax.text(0.8, i + 0.4, f"{i}", ha="center", va="center", fontsize=6, color="#222222")
        ax.text(1.7, i + 0.4, "KV cache" if kind == "attention" else "S, conv", ha="left", va="center",
                fontsize=6, color="#444444")
    ax.text(0.8, -0.9, "the clue, as tokens", ha="center", va="center", fontsize=7)
    ax.text(0.8, 24.7, "answer: P(red | blue)", ha="center", va="center", fontsize=7)
    ax.annotate("", (0.8, 0), (0.8, -0.6), arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#222222"))
    ax.annotate("", (0.8, 24.4), (0.8, 23.8), arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#222222"))
    # reads, on the left
    for i in READ_LAYERS:
        ax.annotate("", (-0.05, i + 0.4), (-0.9, i + 0.4),
                    arrowprops=dict(arrowstyle="<|-", lw=0.7, color=figstyle.BLUE, shrinkA=0, shrinkB=0))
    ax.text(-1.0, READ_LAYERS[-1] + 1.3, "S read by a\nlinear probe", ha="right", va="bottom", fontsize=6,
            color=figstyle.BLUE)
    ax.annotate("", (-0.05, LATENT_LAYER + 0.4), (-0.9, LATENT_LAYER + 0.4),
                arrowprops=dict(arrowstyle="<|-", lw=0.9, color=figstyle.GREEN, shrinkA=0, shrinkB=0))
    ax.text(-1.0, LATENT_LAYER + 0.4, "S → E → z\n(the latent)", ha="right", va="center", fontsize=7,
            color=figstyle.GREEN)
    # the write, on the right
    ax.annotate("", (2.75, EDIT_LAYER + 0.4), (3.3, EDIT_LAYER + 0.4),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=figstyle.VERMILLION, shrinkA=0, shrinkB=0))
    ax.text(2.8, EDIT_LAYER + 1.0, "residual shift on\nthe clue's value word", ha="left", va="bottom",
            fontsize=6.5, color=figstyle.VERMILLION)
    ax.text(0.8, 25.4, "Qwen3.5-2B", ha="center", va="bottom", fontsize=7)
    # the legend of the two kinds
    for k, (label, fc) in enumerate((("Gated DeltaNet (18): S 16×128×128 per layer", "#cfe3f1"),
                                     ("full attention (6): KV of every token", figstyle.ORANGE))):
        ax.add_patch(Rectangle((-2.5, -1.75 - 0.75 * k - 0.0), 0.5, 0.45, fc=fc, ec="#555555", lw=0.4,
                               clip_on=False))
        ax.text(-1.9, -1.55 - 0.75 * k, label, ha="left", va="center", fontsize=6.5, clip_on=False)


def main():
    probes = json.loads(Path("results/qwen3.5-2b/memory_probes.json").read_text())
    edit = json.loads((MEM / "latent_edit.json").read_text())
    resid = json.loads((MEM / "residual_edit.json").read_text())
    patch = json.loads((MEM / "cache_patch.json").read_text())["fraction_of_real"]
    one_layer = max(edit["conditions"][f"oracle, layer {l}"]["fraction_of_real"] for l in edit["layers"])
    rows = [("S of one layer, written (best of 4)", one_layer, figstyle.BLUE),
            (f"residual stream at layer {EDIT_LAYER}, shifted", resid["layers"][str(EDIT_LAYER)]["oracle"]["add_fraction"],
             figstyle.VERMILLION),
            ("S of all 18 layers, copied", patch["S"]["mean"], figstyle.GREY),
            ("conv states of all 18, copied", patch["conv"]["mean"], figstyle.GREY),
            ("KV of the 6 attention layers, copied", patch["KV"]["mean"], figstyle.GREY),
            ("two of the three, best pair", max(patch[k]["mean"] for k in ("S + conv", "S + KV", "conv + KV")),
             figstyle.GREY),
            ("all three: the whole cache", patch["S + conv + KV"]["mean"], "#333333")]

    figstyle.apply()
    plt.rcParams["figure.constrained_layout.use"] = False
    # the layer stack goes to the appendix on its own
    fig = plt.figure(figsize=(3.2, 4.4))
    stack(fig.add_axes([0.0, 0.02, 1.0, 0.96]))
    for ext in ("png", "pdf"):
        fig.savefig(MEM / f"model_stack.{ext}")
    plt.close(fig)

    fig = plt.figure(figsize=(WIDTH, 2.7))
    b = fig.add_axes([0.07, 0.20, 0.36, 0.62])
    c = fig.add_axes([0.72, 0.20, 0.26, 0.62])

    layers = probes["memory_layers"]
    x = np.arange(len(layers))
    s_r2 = probes["states"]["recurrent memory S"]["belief"]
    h_r2 = probes["states"]["residual, current turn"]["belief"]
    b.bar(x - 0.18, s_r2, 0.36, color=figstyle.BLUE, label="recurrent memory S")
    b.bar(x + 0.18, h_r2, 0.36, color="#9fc7e3", label="residual stream, turn mean")
    for xi, v in zip(x, s_r2):
        b.text(xi - 0.18, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
    b.set_xticks(x, [f"layer {l}" for l in layers])
    b.set_ylim(0, 1.5)
    b.set_yticks([0, 0.5, 1.0])
    b.set_ylabel("belief R²")
    b.set_title("Reading: the belief is in any one layer", loc="left")
    b.legend(loc="upper right", fontsize=6.5, ncol=1)

    y = np.arange(len(rows))[::-1]
    c.barh(y, [v for _, v, _ in rows], color=[col for _, _, col in rows])
    for yi, (_, v, _) in zip(y, rows):
        c.text(max(v, 0) + 0.02, yi, f"{v:.2f}", va="center", fontsize=6.5)
    c.set_yticks(y, [label for label, _, _ in rows], fontsize=7)
    c.set_xlim(0, 1.2)
    c.set_xticks([0, 0.5, 1.0])
    c.set_xlabel("share of the clue's effect")
    c.set_title("Writing: the state after a clue, put into the model", loc="left", x=-0.95)
    for letter, x in zip("AB", (0.0, 0.47)):
        fig.text(x, 0.95, letter, fontsize=10, fontweight="bold", va="bottom")

    for ext in ("png", "pdf"):
        fig.savefig(MEM / f"model_figure.{ext}")
    print("wrote", MEM / "model_figure.png", "and model_stack.png, with .pdf")


if __name__ == "__main__":
    main()
