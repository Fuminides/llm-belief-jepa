"""The mirror test, scored: is the player's move distribution equivariant under left-right reflection?

play_dungeon.py --mirror replays each saved argmax game move for move on its level ("original") and, with the moves
mirrored, on the level's mirror image, with the text listing the rays and tied cells reflected ("mirror", the same
text with left and right swapped) or as usual ("mirror_listing"). The generator is mirror-symmetric, so a player
with a symmetric goal gives the mirrored state the mirrored distribution. Per turn: the total variation between the
original's probabilities and the mirror's with left and right swapped back, the share of turns whose most probable
move differs, and the left bias, half the difference between P(left) and P(right) summed over the pair of states
(zero for a symmetric player). The noise floor is the original replay against the saved game's probabilities.

  python dungeon_mirror.py --model Qwen/Qwen3.5-27B    # results/dungeon/medium/mirror/<model>.{txt,png,pdf}
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

import figstyle
from dungeon import MIRROR, MOVES
from play_dungeon import OUT

FLIP = [list(MOVES).index(MIRROR[m]) for m in MOVES]
L, R = list(MOVES).index("left"), list(MOVES).index("right")


def pairs(saved, mirror, key):
    """[turns, 4] arrays: the saved game's, the original replay's and the condition's probabilities (flipped back)."""
    a, b, c, t = [], [], [], []
    for g0, go, gm in zip(saved, mirror["original"], mirror[key]):
        for i, (x, y, z) in enumerate(zip(g0["log"], go["log"], gm["log"])):
            a.append(x["p"][:4]), b.append(y["p"][:4]), c.append(np.array(z["p"][:4])[FLIP]), t.append(i)
    return np.array(a), np.array(b), np.array(c), np.array(t)


def scores(p, q):
    return {"tv": float(0.5 * np.abs(p - q).sum(1).mean()), "argmax_differs": float((p.argmax(1) != q.argmax(1)).mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-27B")
    p.add_argument("--difficulty", default="medium")
    args = p.parse_args()
    name = args.model.replace("/", "__")
    saved = json.loads((OUT / args.difficulty / f"{name}.json").read_text())["results"]["argmax"]["games"]
    mirror = json.loads((OUT / args.difficulty / "mirror" / f"{name}.json").read_text())["results"]
    res, lines = {}, []
    for key in ("mirror", "mirror_listing"):
        s, o, m, t = pairs(saved, mirror, key)
        bias = 0.5 * ((o[:, L] + m[:, R]) - (o[:, R] + m[:, L]))  # m is flipped back: its R is the mirror's left
        res[key] = {"turns": len(t), "noise": scores(s, o), "asymmetry": scores(o, m),
                    "asymmetry_turn0": scores(o[t == 0], m[t == 0]), "left_bias": float(bias.mean()),
                    "left_bias_turn0": float(bias[t == 0].mean()),
                    "tv_by_turn": [float(0.5 * np.abs(o[t == i] - m[t == i]).sum(1).mean()) for i in range(0, 40, 5)
                                   if (t == i).any()]}
        r = res[key]
        lines.append(f"{key:15s} {r['turns']} turns: noise TV {r['noise']['tv']:.3f} (argmax differs "
                     f"{r['noise']['argmax_differs']:.3f}); mirror TV {r['asymmetry']['tv']:.3f} (argmax differs "
                     f"{r['asymmetry']['argmax_differs']:.3f}; turn 0: {r['asymmetry_turn0']['tv']:.3f}, "
                     f"{r['asymmetry_turn0']['argmax_differs']:.3f}); left bias {r['left_bias']:+.3f} "
                     f"(turn 0 {r['left_bias_turn0']:+.3f})")
    print("\n".join(lines))
    stem = OUT / args.difficulty / "mirror" / args.model.split("/")[-1]
    stem.with_suffix(".txt").write_text("\n".join(lines) + "\n")
    stem.with_suffix(".json").write_text(json.dumps(res, indent=1))
    figstyle.apply()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(figstyle.DOUBLE, 2.4))
    s, o, m, t = pairs(saved, mirror, "mirror")
    ax.scatter(o[:, L], m[:, L], s=3, alpha=0.3, color=figstyle.BLUE, lw=0, label="P(left) / P(right) on the mirror")
    ax.scatter(o[:, R], m[:, R], s=3, alpha=0.3, color=figstyle.VERMILLION, lw=0, label="P(right) / P(left) on the mirror")
    ax.plot([0, 1], [0, 1], color=figstyle.GREY, lw=0.8, ls=":")
    ax.set(xlabel="original level", ylabel="mirror image, flipped back", title="Same state, reflected")
    ax.legend(fontsize=6)
    for key, colour in (("mirror", figstyle.BLUE), ("mirror_listing", figstyle.ORANGE)):
        bx.plot(range(0, 5 * len(res[key]["tv_by_turn"]), 5), res[key]["tv_by_turn"], "o-", ms=3, color=colour,
                label={"mirror": "reflected text", "mirror_listing": "usual listing order"}[key])
    bx.axhline(res["mirror"]["noise"]["tv"], color=figstyle.GREY, ls="--", lw=0.8, label="replay noise")
    bx.set(xlabel="turn", ylabel="total variation", title="Distance between the two")
    bx.legend(fontsize=6)
    figstyle.panel_labels([ax, bx])
    figstyle.save(fig, stem)


if __name__ == "__main__":
    main()
