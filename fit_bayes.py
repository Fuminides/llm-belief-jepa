"""Fit a Bayesian observer with free priors and free clue reliabilities to the LLM's beliefs.

In log-odds a Bayesian update is additive: logit P_t = logit prior + sum over clues of ±w(source, phrasing).
Least squares on the LLM's logit beliefs after at least one clue gives the prior the LLM acts as if it had
(one intercept per attribute) and the reliability it acts as if each clue type had (r = sigmoid(w)). The
implied prior is compared with the belief the LLM states before any clue. Intervals: bootstrap over episodes.
A second fit lets each weight depend on recency (most recent clue on that attribute, second most recent,
older), which a Bayesian observer would not do. When the intro stated a different prior in each episode
(collect.py --prior random), both fits also take that prior's log-odds as a regressor: its weight is 1 for a
Bayesian observer, 0 for one that ignores it, and the intercepts are then the prior acted on when 50% is stated.
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import figstyle
import game
from probe_layers import history, logit

CLUE_TYPES = [f"{s}\n{'neg.' if n else 'aff.'}" for s in game.SOURCE_NAMES for n in (0, 1)]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def with_prior(x, prior):
    """x with the log-odds of the stated prior ([E, A]) appended as a last column, if one is given."""
    if prior is None:
        return x
    return np.concatenate([x, np.broadcast_to(logit(prior)[:, None, :, None], (*x.shape[:3], 1))], -1)


def design(actions, prior=None):
    """[E, T, A, A + 4 (+ 1)]: per (episode, turn, attribute), a one-hot attribute and that attribute's 4 signed
    counts (and the stated prior's log-odds)."""
    E, T = actions.shape
    A = len(game.ATTR_NAMES)
    counts = history(actions).reshape(E, T, A, -1)  # last axis: (source, phrasing)
    return with_prior(np.concatenate([np.broadcast_to(np.eye(A), (E, T, A, A)), counts], -1), prior)


def design_recency(actions, prior=None):
    """[E, T, A, A + 12 (+ 1)]: one-hot attribute and signed counts per (recency, clue type), recency in (0, 1, 2+)
    (and the stated prior's log-odds)."""
    E, T = actions.shape
    A = len(game.ATTR_NAMES)
    attr, value, negated, source = np.moveaxis(np.array(game.ACTIONS)[actions], -1, 0)
    sign, ctype = np.where(value == 0, 1, -1), source * 2 + negated
    x = np.zeros((E, T, A, A + 3 * len(CLUE_TYPES)))
    x[..., :A] = np.eye(A)
    seen = np.zeros((E, A), int)  # clues so far per attribute, used to age the earlier ones
    order = np.full((E, T), -1)
    for t in range(T):
        e = np.arange(E)
        order[:, t] = seen[e, attr[:, t]]
        seen[e, attr[:, t]] += 1
        for s in range(t + 1):  # recency of clue s at turn t = clues on its attribute after it
            recency = np.minimum(seen[e, attr[:, s]] - 1 - order[:, s], 2)
            x[e, t, attr[:, s], A + recency * len(CLUE_TYPES) + ctype[:, s]] += sign[:, s]
    return with_prior(x, prior)


def fit(x, y):
    return np.linalg.lstsq(x.reshape(-1, x.shape[-1]), y.ravel(), rcond=None)[0]


def r2(y, pred):
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--boot", type=int, default=200)
    args = p.parse_args()

    shards = [np.load(f) for f in sorted(glob.glob(f"{args.data}/shard_*.npz"))]
    llm, bayes, actions = (np.concatenate([s[k] for s in shards]) for k in ("llm_belief", "bayes_belief", "actions"))
    game.use_attributes(llm.shape[-1])
    prior = np.concatenate([s["prior"] for s in shards]) if "prior" in shards[0] else None
    prior = prior if prior is not None and np.ptp(prior) > 0 else None  # a constant prior is in the intercepts
    E, T = actions.shape
    A = len(game.ATTR_NAMES)
    x, y = design(actions, prior), logit(llm[:, 1:])  # turn 0 left out: the prior is extrapolated from the updates
    coef = fit(x, y)
    rng = np.random.default_rng(0)
    boot = np.array([fit(x[idx], y[idx]) for idx in (rng.integers(E, size=E) for _ in range(args.boot))])
    lo, hi = np.percentile(boot, (2.5, 97.5), axis=0)
    fitted = x @ coef
    xr = design_recency(actions, prior)
    coef_r = fit(xr, y)
    boot_r = np.array([fit(xr[idx], y[idx]) for idx in (rng.integers(E, size=E) for _ in range(args.boot))])
    lo_r, hi_r = np.percentile(boot_r, (2.5, 97.5), axis=0)
    w_r = [c[A:A + 3 * len(CLUE_TYPES)].reshape(3, len(CLUE_TYPES)) for c in (coef_r, lo_r, hi_r)]

    res = {
        "episodes": E,
        "stated_prior": dict(zip(game.ATTR_NAMES, llm[:, 0].mean(0).tolist())),
        "implied_prior": {a: [sigmoid(coef[i]), sigmoid(lo[i]), sigmoid(hi[i])] for i, a in enumerate(game.ATTR_NAMES)},
        "implied_reliability": {t.replace("\n", " "): [sigmoid(coef[A + i]), sigmoid(lo[A + i]), sigmoid(hi[A + i])]
                                for i, t in enumerate(CLUE_TYPES)},
        "r2_fitted_observer": float(r2(y, fitted)),
        "r2_true_bayes": float(r2(y, logit(bayes[:, 1:]))),
        "r2_recency_observer": float(r2(y, xr @ coef_r)),
        "recency_weights": {t.replace("\n", " "): w_r[0][:, i].tolist() for i, t in enumerate(CLUE_TYPES)},
    }
    if prior is not None:
        # Weight of the stated prior after clues (the fits above) and before any (attribute intercepts and that slope).
        x0 = with_prior(np.broadcast_to(np.eye(A), (E, 1, A, A)), prior)
        boot0 = [fit(x0[idx], logit(llm[idx, :1])) for idx in (rng.integers(E, size=E) for _ in range(args.boot))]
        res["prior_weight"] = [float(coef[-1]), float(lo[-1]), float(hi[-1])]
        res["prior_weight_before_clues"] = [float(fit(x0, logit(llm[:, :1]))[-1]),
                                            *np.percentile([b[-1] for b in boot0], (2.5, 97.5)).tolist()]
    print(json.dumps(res, indent=1))

    figstyle.apply()
    fig, axes = plt.subplots(2, 2, figsize=(figstyle.DOUBLE, 4.6))
    axes = axes.ravel()
    implied_style = {"fmt": "o", "color": figstyle.BLUE, "capsize": 2, "lw": 0.8, "label": "implied (95% CI)"}
    stated_style = {"marker": "s", "ls": "none", "color": figstyle.VERMILLION, "mfc": "white", "label": "stated"}
    ref_style = {"color": figstyle.GREY, "lw": 1, "ls": ":", "label": "0.5 (uniform prior; clue ignored)"}

    # A. Prior: stated before any clue vs implied by the updates.
    ax = axes[0]
    xs = np.arange(A)
    implied = np.array(list(res["implied_prior"].values()))
    ax.axhline(0.5, **ref_style)
    ax.errorbar(xs + 0.12, implied[:, 0], yerr=[implied[:, 0] - implied[:, 1], implied[:, 2] - implied[:, 0]],
                **implied_style)
    ax.plot(xs - 0.12, list(res["stated_prior"].values()), **stated_style)
    ax.set_xticks(xs, [f"{a}\n({v[0]})" for a, v in game.ATTRIBUTES.items()])
    ax.set(ylim=(0, 1), ylabel="Prior P(value 0)")
    ax.set_title("Prior per attribute (stated = before any clue)" + (
        f"\nintro's prior weighs {res['prior_weight'][0]:.2f} ({res['prior_weight_before_clues'][0]:.2f} before clues)"
        if "prior_weight" in res else ""), loc="left")

    # B. Reliability the LLM acts as if each clue type had, against the stated one.
    ax = axes[1]
    xs = np.arange(len(CLUE_TYPES))
    rel = np.array(list(res["implied_reliability"].values()))
    ax.axhline(0.5, **ref_style)
    ax.errorbar(xs + 0.12, rel[:, 0], yerr=[rel[:, 0] - rel[:, 1], rel[:, 2] - rel[:, 0]], **implied_style)
    ax.plot(xs - 0.12, [game.SOURCES[s] for s in game.SOURCE_NAMES for _ in (0, 1)], **stated_style)
    ax.set_xticks(xs, CLUE_TYPES)
    ax.set(ylim=(0, 1), ylabel="Reliability")
    ax.set_title("Clue reliability (stated = in the prompt)", loc="left")

    # C. How well a Bayesian observer with these parameters reproduces the LLM.
    ax = axes[2]
    f, o = fitted.ravel(), y.ravel()
    bins = np.digitize(f, np.linspace(f.min(), f.max(), 16)[1:-1])
    full = [b for b in np.unique(bins) if (bins == b).sum() >= 30]
    centre = np.array([f[bins == b].mean() for b in full])
    q = np.array([np.percentile(o[bins == b], (25, 50, 75)) for b in full])
    lim = (min(centre.min(), q.min()) - 0.3, max(centre.max(), q.max()) + 0.3)
    ax.plot(lim, lim, color=figstyle.GREY, lw=1, ls=":")
    ax.fill_between(centre, q[:, 0], q[:, 2], color=figstyle.BLUE, alpha=0.2, lw=0)
    ax.plot(centre, q[:, 1], color=figstyle.BLUE, marker="o")
    ax.set(xlim=lim, ylim=lim, xlabel="Fitted observer, logit", ylabel="LLM logit (median, IQR)")
    ax.set_title(f"Fitted observer: R² {res['r2_fitted_observer']:.2f} (true Bayes {res['r2_true_bayes']:.2f})",
                 loc="left")

    # D. Weight of a clue by recency: a Bayesian observer would give flat lines.
    ax = axes[3]
    styles = [(figstyle.GREEN, "o", "-"), (figstyle.GREEN, "o", "--"), (figstyle.PURPLE, "s", "-"),
              (figstyle.PURPLE, "s", "--")]
    for i, (label, (colour, marker, ls)) in enumerate(zip(CLUE_TYPES, styles)):
        ax.errorbar(np.arange(3), w_r[0][:, i], yerr=[w_r[0][:, i] - w_r[1][:, i], w_r[2][:, i] - w_r[0][:, i]],
                    color=colour, marker=marker, ls=ls, capsize=2, lw=1.2, mfc="white" if ls == "--" else colour,
                    label=label.replace("\n", " "))
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(range(3), ["last", "2nd last", "older"])
    ax.set(xlabel="Clue on that attribute", ylabel="Weight (log-odds)")
    ax.set_title(f"Weight by recency: R² {res['r2_recency_observer']:.2f}", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
    figstyle.panel_labels(axes)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "implied_bayes.json").write_text(json.dumps(res, indent=1))
    figstyle.save(fig, out / "implied_bayes")


if __name__ == "__main__":
    main()
