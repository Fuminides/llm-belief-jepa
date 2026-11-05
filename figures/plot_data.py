"""Overview figure of the collected beliefs: how the LLM's belief compares with the Bayesian one."""
import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figures import figstyle
from games import game
from analysis.probe_layers import logit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    shards = [np.load(f) for f in sorted(glob.glob(f"{args.data}/shard_*.npz"))]
    llm, bayes, actions = (np.concatenate([s[k] for s in shards]) for k in ("llm_belief", "bayes_belief", "actions"))
    E, T = actions.shape
    attr, value, negated, source = np.moveaxis(np.array(game.ACTIONS)[actions], -1, 0)
    e, t = np.indices((E, T))
    sign = np.where(value == 0, 1, -1)
    shift = sign * (logit(llm[e, t + 1, attr]) - logit(llm[e, t, attr]))  # log-odds shift towards the asserted value
    other = np.ones((E, T, len(game.ATTR_NAMES)), bool)
    other[e, t, attr] = False
    spill = np.abs(logit(llm[:, 1:]) - logit(llm[:, :-1]))[other]

    print(f"{E} episodes of {T} turns")
    print("belief before any clue (Bayes 0.5):", dict(zip(game.ATTR_NAMES, llm[0, 0].round(3))))
    print("corr LLM vs Bayes after the last clue:", np.corrcoef(llm[:, -1].ravel(), bayes[:, -1].ravel())[0, 1].round(3))
    print("mean |log-odds shift| on attributes the clue is not about:", spill.mean().round(3))

    figstyle.apply()
    fig, axes = plt.subplots(1, 3, figsize=(figstyle.DOUBLE, 2.4))

    llm_style = {"color": figstyle.BLUE, "marker": "o", "ls": "-"}
    bayes_style = {"color": figstyle.GREY, "marker": "s", "ls": ":"}

    # A. LLM belief against the Bayesian posterior, after at least one clue.
    ax = axes[0]
    b, l = bayes[:, 1:].ravel(), llm[:, 1:].ravel()
    levels, inverse, counts = np.unique(b.round(4), return_inverse=True, return_counts=True)
    keep = counts >= 30
    q = np.array([np.percentile(l[inverse == i], (25, 50, 75)) for i in np.flatnonzero(keep)])
    ax.plot([0, 1], [0, 1], color=figstyle.GREY, lw=1, ls=":")
    ax.fill_between(levels[keep], q[:, 0], q[:, 2], color=figstyle.BLUE, alpha=0.2, lw=0)
    ax.plot(levels[keep], q[:, 1], **llm_style)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Bayesian P(value 0)", ylabel="LLM P(value 0)")
    ax.set_title("Belief (median, IQR)", loc="left")

    # B. Update on the attribute the clue is about, by clue type.
    ax = axes[1]
    groups = [(s, n) for s in range(len(game.SOURCE_NAMES)) for n in (0, 1)]
    names = [f"{game.SOURCE_NAMES[s]}\n{'neg.' if n else 'aff.'}" for s, n in groups]
    x = np.arange(len(groups))
    llm_mean, llm_ci, bayes_shift = [], [], []
    for s, n in groups:
        v = shift[(source == s) & (negated == n)]
        llm_mean.append(v.mean())
        llm_ci.append(1.96 * v.std() / np.sqrt(len(v)))
        r = game.SOURCES[game.SOURCE_NAMES[s]]
        bayes_shift.append(np.log(r / (1 - r)))
    ax.axhline(0, color="black", lw=0.5)
    ax.plot(x + 0.12, bayes_shift, **{**bayes_style, "ls": "none"})
    ax.errorbar(x - 0.12, llm_mean, yerr=llm_ci, capsize=2, lw=0.8, **{**llm_style, "ls": "none"})
    ax.set_xticks(x, names)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylabel("Log-odds shift to asserted value")
    ax.set_title("Update per clue (mean, 95% CI)", loc="left")

    # C. Confidence as clues accumulate.
    ax = axes[2]
    turns = np.arange(T + 1)
    ax.plot(turns, np.abs(llm - 0.5).mean(axis=(0, 2)), label="LLM", **llm_style)
    ax.plot(turns, np.abs(bayes - 0.5).mean(axis=(0, 2)), label="Bayes", **bayes_style)
    ax.set(xlabel="Turn (clues so far)", ylabel="Mean |P − 0.5|", ylim=(0, 0.5))
    ax.set_title("Confidence by turn", loc="left")

    fig.legend(*axes[2].get_legend_handles_labels(), loc="outside lower center", ncol=2)
    figstyle.panel_labels(axes)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    figstyle.save(fig, out / "data_overview")


if __name__ == "__main__":
    main()
