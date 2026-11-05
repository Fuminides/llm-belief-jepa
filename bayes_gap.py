"""The Bayes gap and its anatomy (the working doc's test list, H4): one panel per suspected departure, per model.

The weights come from fit_bayes.py's implied observer (least squares in log-odds; a stated prior that varies per
episode enters as a regressor), with bootstrap intervals over episodes. A Bayesian observer gives every clue of a
source the same weight, log(r / (1 - r)): 2.20 for Alice (90%) and 0.41 for Bob (60%), whatever the phrasing, the
age of the clue or the belief it arrives at.
  A. Negation: weight of a negated clue over that of an affirmative one, per source (Bayes: 1).
  B. Sources: weight of Bob over Alice, per phrasing (Bayes: 0.41 / 2.20 = 0.18).
  C. Recency: weight of an affirmative Alice clue when it is the last on its attribute, the second last, or older.
  D. Shrinking updates: the LLM's shift towards the asserted value against its belief in that value before the
     clue (affirmative Alice clues; Bayes: flat at 2.20); the slope per clue type is in the JSON.
  E. Wording: weight per designed convolutedness level (paraphrased clues, from results/qwen3.5-2b-paraphrase/
     paraphrase.json, whose data is not on this machine).

  python bayes_gap.py   # results/bayes_gap.{json,png,pdf}
"""
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import figstyle
import game
from fit_bayes import CLUE_TYPES, design, design_recency, fit
from probe_layers import logit

MODELS = {  # label: (data, colour); the clue theme with four attributes
    "Qwen3.5-2B": ("data/qwen3.5-2b-memory-mid", figstyle.BLUE),
    "Qwen3.5-9B": ("data/qwen3.5-9b-randprior", figstyle.VERMILLION),
    "Qwen3.5-27B": ("data/qwen3.5-27b-memory-mid", figstyle.GREEN),
}
PARAPHRASE = Path("results/qwen3.5-2b-paraphrase/paraphrase.json")
OUT = Path("results/bayes_gap")
BAYES = {s: float(np.log(r / (1 - r))) for s, r in game.SOURCES.items()}
TYPES = [t.replace("\n", " ") for t in CLUE_TYPES]  # Alice aff., Alice neg., Bob aff., Bob neg.


def load(data):
    shards = [np.load(f) for f in sorted(glob.glob(f"{data}/shard_*.npz"))]
    llm, actions = (np.concatenate([s[k] for s in shards]) for k in ("llm_belief", "actions"))
    prior = np.concatenate([s["prior"] for s in shards]) if "prior" in shards[0].files else None
    return llm, actions, prior if prior is not None and np.ptp(prior) > 0 else None


def boot(x, y, n, rng):
    """Point estimate and 95% interval of the least-squares weights, resampling episodes."""
    coef = fit(x, y)
    reps = np.array([fit(x[i], y[i]) for i in (rng.integers(len(x), size=len(x)) for _ in range(n))])
    return coef, reps


def ratio(reps, coef, i, j):
    """Weight i over weight j, with a bootstrap interval."""
    r = reps[:, i] / reps[:, j]
    return [float(coef[i] / coef[j]), *np.percentile(r, (2.5, 97.5)).tolist()]


def shrinkage(llm, actions, prior):
    """Per real clue: the belief in the asserted value before it and the shift towards it (log-odds)."""
    lb = logit(llm)
    E, T = actions.shape
    attr, value, negated, source = np.moveaxis(np.array(game.ACTIONS)[actions], -1, 0)
    sign = np.where(value == 0, 1, -1)
    e, t = np.indices((E, T))
    before = sign * lb[e, t, attr]
    shift = sign * (lb[e, t + 1, attr] - lb[e, t, attr])
    return before, shift, source * 2 + negated


def main():
    rng = np.random.default_rng(0)
    res, curves = {"bayes_weight": BAYES, "models": {}}, {}
    for name, (data, _) in MODELS.items():
        if not glob.glob(f"{data}/shard_*.npz"):
            continue
        llm, actions, prior = load(data)
        A = len(game.ATTR_NAMES)
        y = logit(llm[:, 1:])
        coef, reps = boot(design(actions, prior), y, 200, rng)
        w = {t: [float(coef[A + i]), *np.percentile(reps[:, A + i], (2.5, 97.5)).tolist()] for i, t in enumerate(TYPES)}
        coef_r, reps_r = boot(design_recency(actions, prior), y, 200, rng)
        recency = {t: {"weight": coef_r[A + np.arange(3) * len(TYPES) + i].tolist(),
                       "low": np.percentile(reps_r[:, A + np.arange(3) * len(TYPES) + i], 2.5, axis=0).tolist(),
                       "high": np.percentile(reps_r[:, A + np.arange(3) * len(TYPES) + i], 97.5, axis=0).tolist()}
                   for i, t in enumerate(TYPES)}
        before, shift, ctype = shrinkage(llm, actions, prior)
        slopes = {t: float(np.polyfit(before[ctype == i], shift[ctype == i], 1)[0]) for i, t in enumerate(TYPES)}
        res["models"][name] = {
            "data": data, "episodes": len(actions), "stated_prior_regressor": prior is not None,
            "weights": w,
            "negated_over_affirmative": {"Alice": ratio(reps, coef, A + 1, A), "Bob": ratio(reps, coef, A + 3, A + 2)},
            "bob_over_alice": {"affirmative": ratio(reps, coef, A + 2, A), "negated": ratio(reps, coef, A + 3, A + 1)},
            "recency": recency,
            "shrinkage_slope": slopes,
        }
        edges = np.quantile(before[ctype == 0], np.linspace(0, 1, 11))
        idx = np.digitize(before[ctype == 0], edges[1:-1])
        curves[name] = (np.array([before[ctype == 0][idx == k].mean() for k in range(10)]),
                        np.array([shift[ctype == 0][idx == k].mean() for k in range(10)]))
        print(name, json.dumps({k: res["models"][name][k] for k in ("negated_over_affirmative", "bob_over_alice",
                                                                    "shrinkage_slope")}), flush=True)
    wording = json.loads(PARAPHRASE.read_text())["weight_by_level"] if PARAPHRASE.exists() else None
    res["wording_weight_by_level_2b"] = wording
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bayes_gap.json").write_text(json.dumps(res, indent=1))
    plot(res, curves, wording)


def plot(res, curves, wording):
    figstyle.apply()
    fig, axes = plt.subplots(1, 5, figsize=(figstyle.DOUBLE, 2.5))
    names = list(res["models"])
    width = 0.8 / len(names)

    def bars(ax, key, groups, ref, ylabel, title):
        for m, name in enumerate(names):
            vals = np.array([res["models"][name][key][g] for g in groups])
            x = np.arange(len(groups)) + (m - (len(names) - 1) / 2) * width
            ax.bar(x, vals[:, 0], width * 0.9, color=MODELS[name][1], label=name,
                   yerr=[vals[:, 0] - vals[:, 1], vals[:, 2] - vals[:, 0]], error_kw={"lw": 0.6, "capsize": 1.5})
        ax.axhline(ref, color=figstyle.GREY, ls=":", lw=1, label="Bayes")
        ax.set_xticks(range(len(groups)), groups)
        ax.set(ylabel=ylabel)
        ax.set_title(title, loc="left")

    bars(axes[0], "negated_over_affirmative", ["Alice", "Bob"], 1, "Negated / affirmative weight", "Negation")
    bars(axes[1], "bob_over_alice", ["affirmative", "negated"], BAYES["Bob"] / BAYES["Alice"],
         "Bob / Alice weight", "Sources")
    axes[1].set_xticks(range(2), ["aff.", "neg."])
    ax = axes[2]
    for name in names:
        r = res["models"][name]["recency"]["Alice aff."]
        ax.errorbar(range(3), r["weight"], yerr=[np.subtract(r["weight"], r["low"]), np.subtract(r["high"], r["weight"])],
                    color=MODELS[name][1], marker="o", capsize=2, lw=1.2, label=name)
    ax.set_xticks(range(3), ["last", "2nd last", "older"])
    ax.set(ylabel="Weight, Alice aff. (log-odds)", ylim=(0, None))
    ax.set_title("Recency", loc="left")
    ax = axes[3]
    for name in names:
        x, y = curves[name]
        ax.plot(x, y, color=MODELS[name][1], marker="o", ms=3, label=name)
    ax.axhline(BAYES["Alice"], color=figstyle.GREY, ls=":", lw=1)
    ax.axhline(0, color="black", lw=0.5)
    ax.set(xlabel="Belief in asserted value\nbefore (logit)", ylabel="Shift, Alice aff. (log-odds)")
    ax.set_title("Shrinking updates", loc="left")
    ax = axes[4]
    if wording:
        for phrasing, colour, mfc in (("affirmative", figstyle.BLUE, figstyle.BLUE), ("negated", figstyle.BLUE, "white")):
            ax.plot(range(len(wording[phrasing])), wording[phrasing], color=colour, mfc=mfc, marker="o",
                    ls="-" if phrasing == "affirmative" else "--", label=f"2B, {phrasing}")
        ax.set_xticks(range(len(wording["affirmative"])))
        ax.legend(loc="lower left", fontsize=6)
    ax.set(xlabel="Convolutedness level", ylabel="Weight (log-odds)", ylim=(0, None))
    ax.set_title("Wording", loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    figstyle.panel_labels(axes)
    figstyle.save(fig, OUT / "bayes_gap")


if __name__ == "__main__":
    main()
