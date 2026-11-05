"""Linear probes per layer: how much of the LLM's belief, and of the clue history, each layer's state holds.

States are the last-token vector (`acts`) or the mean over the turn's tokens (`pool`). Turn 0 is left out
(identical in every episode). Probes are ridge regressions on standardised features, scored by R² on
held-out episodes.
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

import figstyle
import game

ALPHAS = np.logspace(-1, 5, 13)


def history(actions):
    """[E, T, 16] signed clue counts per (attribute, source, phrasing) up to each turn, +1 towards value 0."""
    E, T = actions.shape
    attr, value, negated, source = np.moveaxis(np.array(game.ACTIONS)[actions], -1, 0)
    counts = np.zeros((E, T, len(game.ATTR_NAMES), len(game.SOURCE_NAMES), 2))
    e, t = np.indices((E, T))
    counts[e, t, attr, source, negated] = np.where(value == 0, 1, -1)
    return counts.reshape(E, T, -1).cumsum(1)


def logit(p, eps=1e-3):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def probe(x_tr, y_tr, x_te, y_te):
    """Test R² per output column."""
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
    model = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit((x_tr - mu) / sd, y_tr)
    return r2_score(y_te, model.predict((x_te - mu) / sd), multioutput="raw_values")


def run(data_dir, test_frac):
    files = sorted(glob.glob(f"{data_dir}/shard_*.npz"))
    shards = [np.load(f) for f in files]
    actions = np.concatenate([s["actions"] for s in shards])
    E, T = actions.shape
    belief = logit(np.concatenate([s["llm_belief"] for s in shards])[:, 1:])  # [E, T, 4]
    hist = history(actions)                                                   # [E, T, 16]
    y = np.concatenate([belief, hist], -1).reshape(E * T, -1)
    n_tr = round(E * (1 - test_frac)) * T  # rows are episode-major, so this splits by episode
    fit = lambda x, cols=slice(None): probe(x[:n_tr], y[:n_tr, cols], x[n_tr:], y[n_tr:, cols])
    print(f"{len(files)} shards, {E} episodes, {E * T} states, test fraction {test_frac}")

    res = {"controls": {
        "belief from turn index": float(fit(np.tile(np.eye(T), (E, 1)), slice(4)).mean()),
        "belief from clue counts": float(fit(hist.reshape(E * T, -1), slice(4)).mean()),
    }}
    print(res["controls"])

    for key in ("acts", "pool"):
        states = [s[key] for s in shards]  # kept per shard; one layer at a time is concatenated below
        res[key] = {"belief": [], "belief_per_attr": [], "history": []}
        for layer in range(states[0].shape[2]):
            r2 = fit(np.concatenate([a[:, 1:, layer] for a in states]).reshape(E * T, -1).astype(np.float32))
            res[key]["belief"].append(float(r2[:4].mean()))
            res[key]["belief_per_attr"].append(r2[:4].tolist())
            res[key]["history"].append(float(r2[4:].mean()))
            print(f"{key} layer {layer:2d}  belief R² {r2[:4].mean():.3f}  history R² {r2[4:].mean():.3f}", flush=True)
        del states
    return res


def plot(res, stem):
    figstyle.apply()
    fig, axes = plt.subplots(1, 2, figsize=(figstyle.DOUBLE, 2.6), sharey=True)
    layers = np.arange(len(res["acts"]["belief"]))
    series = (("acts", figstyle.BLUE, "o", "-", "last token"),
              ("pool", figstyle.VERMILLION, "s", "--", "mean over turn tokens"))
    for ax, target, title in zip(axes, ("belief", "history"),
                                 ("LLM belief (logit, 4 attributes)", "Clue history (16 signed counts)")):
        for key, colour, marker, ls, label in series:
            ax.plot(layers, res[key][target], color=colour, marker=marker, ls=ls, label=label)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Layer (0 = embeddings, last = after final norm)")
    for (name, value), ls in zip(res["controls"].items(), (":", "-.")):
        axes[0].axhline(value, color=figstyle.GREY, lw=1, ls=ls, label=name.replace("belief from ", "") + ", no activations")
    axes[0].set_ylabel("Test R²")
    axes[0].set_ylim(min(-0.05, min(min(res[k][t]) for k in ("acts", "pool") for t in ("belief", "history"))), 1)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4)
    figstyle.panel_labels(axes)
    figstyle.save(fig, stem)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--test-frac", type=float, default=0.2)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = run(args.data, args.test_frac)
    (out / "probes.json").write_text(json.dumps(res, indent=1))
    plot(res, out / "probes")
