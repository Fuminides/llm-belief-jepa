"""The forecasting test (H1) as the paper shows it: plain and paraphrased clues side by side, from jepa.py runs with
--seeds (mean over seeds, the band one standard deviation across seeds).

  A  plain clues, R² per horizon at the largest training set
  B  plain clues, mean R² over horizons against the training episodes
  C  paraphrased clues, the same, with the GRU that also sees each clue's template
  D  the margin of the latent model over the GRU (mean over horizons, 95% bootstrap interval over the test
     episodes) on the trust game for the 2B, 9B and 27B, with the plain clue game of the 2B and the 27B as references

  python -m figures.forecast_figure   # manuscript/working_doc_0/figures/h1-forecasting.{pdf,png}
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figures import figstyle
LABELS = {"latent model": "anchored JEPA", "latent model, no action": "anchored JEPA, no action"}  # JSON key -> legend
STYLES = {"latent model": (figstyle.BLUE, "o", "-"), "GRU on clue ids": (figstyle.VERMILLION, "s", "-"),
          "GRU on clue ids + phrasing": (figstyle.ORANGE, "D", "--"),
          "latent model, no action": (figstyle.SKY, "^", "--")}


def band(ax, x, res_list, name, key=None):
    colour, marker, ls = STYLES[name]
    if key is None:  # one run, per horizon
        y, sd = np.array(res_list[name]), np.array(res_list.get("sd over seeds", {}).get(name, 0))
    else:  # over runs, the mean over horizons
        y = np.array([np.mean(r[name]) for r in res_list])
        sd = np.array([np.mean(r.get("sd over seeds", {}).get(name, 0)) for r in res_list])
    ax.plot(x, y, color=colour, marker=marker, ls=ls, label=LABELS.get(name, name))
    ax.fill_between(x, y - sd, y + sd, color=colour, alpha=0.2, lw=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plain", default="results/qwen3.5-2b-memory-mid/seeds/jepa_memory_layer5.json")
    p.add_argument("--paraphrase", default="results/qwen3.5-2b-paraphrase-memory-mid/seeds/jepa_memory_layer5.json")
    p.add_argument("--trust", nargs="+", default=["results/qwen3.5-2b-trust/seeds/jepa_memory_layer5.json",
                                                   "results/qwen3.5-9b-trust/seeds/jepa_memory_layer5.json",
                                                   "results/qwen3.5-27b-trust/seeds/jepa_memory_layer5.json"])
    p.add_argument("--trust-names", nargs="+", default=["2B", "9B", "27B"])
    p.add_argument("--plain-27b", default="results/qwen3.5-27b-memory-mid/seeds/jepa_memory_layer5.json")
    p.add_argument("--out", default="manuscript/working_doc_0/figures/h1-forecasting")
    args = p.parse_args()
    plain = json.loads(Path(args.plain).read_text())["results"]
    para = json.loads(Path(args.paraphrase).read_text())["results"]
    figstyle.apply()
    trust = [json.loads(Path(f).read_text())["results"] for f in args.trust]
    plain27 = json.loads(Path(args.plain_27b).read_text())["results"]
    fig, axes = plt.subplots(1, 4, figsize=(figstyle.DOUBLE, 2.4))
    largest = plain[-1]
    ks = np.arange(1, len(largest["GRU on clue ids"]) + 1)
    for name in ("latent model", "GRU on clue ids", "latent model, no action"):
        band(axes[0], ks, largest, name)
    axes[0].axhline(largest["linear probe of true state"], color=figstyle.GREY, ls="-.", lw=1,
                    label="linear probe of the true state")
    axes[0].set(xlabel="Horizon (clues ahead)", ylabel="Test $R^2$ of the belief logits", ylim=(-0.1, 1))
    axes[0].set_title(f"Plain clues, {largest['train_episodes']} episodes", loc="left")
    for ax, res, title, names in ((axes[1], plain, "Plain clues", ("latent model", "GRU on clue ids")),
                                  (axes[2], para, "Paraphrased clues",
                                   ("latent model", "GRU on clue ids", "GRU on clue ids + phrasing"))):
        n = [r["train_episodes"] for r in res]
        for name in names:
            band(ax, n, res, name, key="mean")
        ax.set(xscale="log", xlabel="Training episodes", ylabel="Mean test $R^2$, horizons 1 to 8", ylim=(0, 1))
        ax.set_xticks(n, n)
        ax.minorticks_off()
        ax.set_title(title, loc="left")
    ax = axes[3]
    ax.axhline(0, color=figstyle.GREY, lw=0.8)
    colours = dict(zip(args.trust_names, (figstyle.SKY, figstyle.BLUE, figstyle.PURPLE)))
    for res, size, ls in [(plain, args.trust_names[0], "--"), (plain27, args.trust_names[-1], "--")] + [
            (r, n, "-") for r, n in zip(trust, args.trust_names)]:
        n = [r["train_episodes"] for r in res]
        y = [np.mean(r["latent model"]) - np.mean(r["GRU on clue ids"]) for r in res]
        lo, hi = zip(*[r["latent minus GRU"]["mean_over_horizons_ci95"] for r in res])
        ax.plot(n, y, color=colours[size], marker="o", ls=ls)
        ax.fill_between(n, lo, hi, color=colours[size], alpha=0.2, lw=0)
    ax.set(xscale="log", xlabel="Training episodes", ylabel="Latent model minus GRU, mean $R^2$", ylim=(-0.14, 0.2))
    ax.set_xticks(n, n)
    ax.minorticks_off()
    ax.set_title("Margin by size and game", loc="left")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, marker="o", label=s) for s, c in colours.items()]
    handles += [Line2D([], [], color="k", ls="-", label="trust game"), Line2D([], [], color="k", ls="--", label="clue game")]
    ax.legend(handles=handles, fontsize=6, loc="upper left", ncol=2, columnspacing=0.8, handlelength=1.6)
    handles, labels = {}, []
    for ax in axes[:3]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in handles:
                handles[l] = h
    fig.legend(handles.values(), handles.keys(), loc="outside lower center", ncol=5)
    figstyle.panel_labels(axes)
    figstyle.save(fig, args.out)
    for label, res in [("plain", plain), ("paraphrase", para), ("plain 27B", plain27)] + list(zip(args.trust_names, trust)):
        for r in res:
            ci = r["latent minus GRU"]
            print(f"{label} {r['train_episodes']:5d}: latent {np.mean(r['latent model']):.3f} GRU "
                  f"{np.mean(r['GRU on clue ids']):.3f} margin 95% {np.round(ci['mean_over_horizons_ci95'], 3).tolist()}"
                  + (f" GRU+phrasing {np.mean(r['GRU on clue ids + phrasing']):.3f}" if "GRU on clue ids + phrasing" in r else ""))


if __name__ == "__main__":
    main()
