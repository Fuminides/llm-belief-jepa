"""One belief or several (H2), first half: candidate observers, per-layer probes and the mixture fit.

Every observer turns the clue history into log-odds per attribute, from the same signed clue counts (towards value
0) with different weights:
  Bayes            log(r / (1 - r)) per clue of a source with reliability r, whatever the phrasing or the age
  recency          Bayes weights discounted by 0.7 per rank from the most recent clue on the attribute
  Alice only       Bayes weights, Bob's clues ignored
  Bob only         Bayes weights, Alice's clues ignored
  negation-blind   Bayes weights, negated clues ignored
  majority         +2 if more clues assert value 0 than value 1, -2 if fewer, 0 if tied
  implied          the LLM's own habits: least squares of its stated log-odds on the 16 counts (fit_bayes.design),
                   fitted on the training episodes, the ceiling for anything linear in the counts
Per layer, a ridge probe from the recurrent memory S to each observer's log-odds, on held-out episodes: the matrix
layer x observer, with the counts' own readability as the reference, since every observer is a function of them.
Then the stated belief as a mixture of the observers, weights constant and weights that change where the sources
contradict each other on the attribute (both values asserted so far), against the implied observer alone. H2 wants
different layers best explained by different observers, and gate weights that move with contradiction; the
numbers here are the per-layer and the behavioural halves, heads.py gives the latent's heads.

  python observers.py --data data/qwen3.5-2b-memory-all --out results/qwen3.5-2b-memory-all
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeCV

import figstyle
import game
from fit_bayes import design, fit
from probe_layers import history, logit

ALPHAS = np.logspace(0, 5, 11)
DISCOUNT = 0.7


def clue_table(actions):
    """attr, sign (+1 towards value 0), negated, source: each [E, T]."""
    attr, value, negated, source = np.moveaxis(np.array(game.ACTIONS)[actions], -1, 0)
    return attr, np.where(value == 0, 1.0, -1.0), negated, source


def observers(actions):
    """name -> log-odds [E, T+1, A] after each turn (turn 0: before any clue)."""
    E, T = actions.shape
    A = len(game.ATTR_NAMES)
    attr, sign, negated, source = clue_table(actions)
    w = np.array([np.log(game.SOURCES[s] / (1 - game.SOURCES[s])) for s in game.SOURCE_NAMES])[source]  # [E, T]
    keep = {"Bayes": np.ones_like(w), "Alice only": (source == 0).astype(float), "Bob only": (source == 1).astype(float),
            "negation-blind": (negated == 0).astype(float)}
    out = {}
    for name, k in keep.items():
        lo = np.zeros((E, T + 1, A))
        for t in range(T):
            lo[:, t + 1] = lo[:, t]
            lo[np.arange(E), t + 1, attr[:, t]] += k[:, t] * w[:, t] * sign[:, t]
        out[name] = lo
    rec = np.zeros((E, T + 1, A))
    for t in range(1, T + 1):  # rank 0 is the most recent clue on the attribute, by the time of turn t
        for e in range(E):
            for a in range(A):
                idx = np.flatnonzero(attr[e, :t] == a)[::-1]
                rec[e, t, a] = sum(DISCOUNT ** r * w[e, i] * sign[e, i] for r, i in enumerate(idx))
    out["recency"] = rec
    counts = np.zeros((E, T + 1, A))
    for t in range(T):
        counts[:, t + 1] = counts[:, t]
        counts[np.arange(E), t + 1, attr[:, t]] += sign[:, t]
    out["majority"] = 2.0 * np.sign(counts)
    return out


def contradicted(actions):
    """[E, T+1, A]: both values of the attribute asserted so far."""
    E, T = actions.shape
    A = len(game.ATTR_NAMES)
    attr, sign, _, _ = clue_table(actions)
    seen = np.zeros((E, T + 1, A, 2))
    for t in range(T):
        seen[:, t + 1] = seen[:, t]
        seen[np.arange(E), t + 1, attr[:, t], (sign[:, t] < 0).astype(int)] = 1
    return seen.prod(-1)


def implied(actions, stated, n_train):
    """The implied observer: stated log-odds after t >= 1 clues as a linear function of the attribute and its counts."""
    x = design(actions)  # [E, T, A, A + 4]
    coef = fit(x[:n_train], stated[:n_train, 1:])
    lo = np.zeros_like(stated)
    lo[:, 1:] = x @ coef
    return lo


def r2(pred, target):
    return float(1 - ((target - pred) ** 2).sum() / ((target - target.mean()) ** 2).sum())


def probe(x, y, n_train):
    """Ridge from standardised x [E, T+1, d] to y [E, T+1, m]; R² per column on the test episodes."""
    d, m = x.shape[-1], y.shape[-1]
    mu, sd = x[:n_train].mean((0, 1)), x[:n_train].std((0, 1)) + 1e-6
    xs = ((x - mu) / sd).reshape(-1, d)
    ys = y.reshape(-1, m)
    n = n_train * x.shape[1]
    model = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(xs[:n], ys[:n])
    pred = model.predict(xs[n:])
    return [r2(pred[:, j], ys[n:, j]) for j in range(m)]


def mixture(obs, stated, contra, n_train):
    """Stated log-odds (turns >= 1) as a weighted sum of the observers, with constant weights and with weights that
    change with contradiction on the attribute. Returns weights and test R²."""
    names = list(obs)
    X = np.stack([obs[k][:, 1:] for k in names], -1)  # [E, T, A, J]
    c = contra[:, 1:, :, None]
    Xc = np.concatenate([X, X * c, c, np.ones_like(c)], -1)
    X1 = np.concatenate([X, np.ones_like(c)], -1)
    y = stated[:, 1:]
    out = {}
    for label, Z in (("constant weights", X1), ("weights by contradiction", Xc)):
        coef = fit(Z[:n_train], y[:n_train])
        pred = Z @ coef
        w = dict(zip(names, coef[:len(names)].round(3).tolist()))
        entry = {"weights": w, "r2_test": r2(pred[n_train:], y[n_train:]),
                 "r2_test_contradicted": r2(pred[n_train:][contra[n_train:, 1:] > 0], y[n_train:][contra[n_train:, 1:] > 0]),
                 "r2_test_not_contradicted": r2(pred[n_train:][contra[n_train:, 1:] == 0],
                                                y[n_train:][contra[n_train:, 1:] == 0])}
        if label == "weights by contradiction":
            entry["weights when contradicted"] = {k: round(float(coef[j] + coef[len(names) + j]), 3)
                                                  for j, k in enumerate(names)}
        out[label] = entry
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/qwen3.5-2b-memory-all")
    p.add_argument("--out", default="results/qwen3.5-2b-memory-all")
    p.add_argument("--test-frac", type=float, default=0.2)
    args = p.parse_args()
    shards = [np.load(f) for f in sorted(glob.glob(f"{args.data}/shard_*.npz"))]
    actions = np.concatenate([s["actions"] for s in shards])
    stated = logit(np.concatenate([s["llm_belief"] for s in shards]))
    layers = [int(l) for l in shards[0]["memory_layers"]]
    E, T = actions.shape
    n_train = round(E * (1 - args.test_frac))
    obs = observers(actions)
    obs["implied"] = implied(actions, stated, n_train)
    contra = contradicted(actions)
    names = list(obs)
    res = {"layers": layers, "train_episodes": n_train, "test_episodes": E - n_train, "observers": names}

    # How much of the stated belief each observer explains on its own (test episodes, turns >= 1).
    res["observer_vs_stated_r2"] = {k: r2(v[n_train:, 1:], stated[n_train:, 1:]) for k, v in obs.items()}
    res["observer_vs_bayes_r2"] = {k: r2(v[n_train:, 1:], obs["Bayes"][n_train:, 1:]) for k, v in obs.items()}
    print("observer vs stated:", {k: round(v, 3) for k, v in res["observer_vs_stated_r2"].items()}, flush=True)

    # Per layer: probes of every observer (and of the stated belief and the counts) from S.
    targets = np.concatenate([obs[k] for k in names] + [stated], -1)  # [E, T+1, 4J + 4]
    counts = np.concatenate([np.zeros((E, 1, 16)), history(actions)], 1)  # the 16 signed counts, turn 0 empty
    res["per_layer"] = {}
    for j, layer in enumerate(layers):
        x = np.concatenate([s["memory"][:, :, j] for s in shards]).astype(np.float32)
        cols = probe(x, np.concatenate([targets, counts], -1), n_train)
        A = len(game.ATTR_NAMES)
        entry = {k: float(np.mean(cols[i * A:(i + 1) * A])) for i, k in enumerate(names)}
        entry["stated belief"] = float(np.mean(cols[len(names) * A:(len(names) + 1) * A]))
        entry["clue counts"] = float(np.mean(cols[(len(names) + 1) * A:]))
        res["per_layer"][layer] = entry
        print(f"layer {layer:2d}", {k: round(v, 3) for k, v in entry.items()}, flush=True)
        del x

    res["mixture"] = mixture({k: obs[k] for k in names if k != "implied"}, stated, contra, n_train)
    res["mixture"]["implied alone"] = {"r2_test": r2(obs["implied"][n_train:, 1:], stated[n_train:, 1:])}
    print(json.dumps(res["mixture"], indent=1))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "observers.json").write_text(json.dumps(res, indent=1))
    plot(res, out / "observers")


def plot(res, stem):
    figstyle.apply()
    fig, axes = plt.subplots(1, 2, figsize=(figstyle.DOUBLE, 2.6), gridspec_kw={"width_ratios": [1.9, 1]})
    layers = list(res["per_layer"])
    ax = axes[0]
    colours = {"Bayes": figstyle.GREEN, "recency": figstyle.ORANGE, "Alice only": figstyle.SKY, "Bob only": figstyle.PURPLE,
               "negation-blind": figstyle.VERMILLION, "majority": figstyle.GREY, "implied": figstyle.BLUE,
               "stated belief": "black", "clue counts": "#bbbbbb"}
    for name, colour in colours.items():
        ax.plot([int(l) for l in layers], [res["per_layer"][l][name] for l in layers], color=colour, marker="o", ms=2.5,
                lw=1, ls="--" if name in ("stated belief", "clue counts") else "-", label=name)
    ax.set(xlabel="Layer (recurrent memory $S$)", ylabel="Probe $R^2$, held-out episodes", ylim=(0, 1))
    ax.set_title("Which observer each layer's memory supports", loc="left")
    ax.legend(fontsize=6, ncol=3, loc="lower left")
    ax = axes[1]
    m = res["mixture"]
    labels = ["implied alone", "constant weights"]
    ax.bar(range(2), [m[k]["r2_test"] for k in labels], color=[figstyle.BLUE, figstyle.GREY], width=0.6)
    for i, k in enumerate(labels):
        ax.text(i, m[k]["r2_test"] + 0.01, f"{m[k]['r2_test']:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(2), ["implied\nobserver", "mixture of the\nsix candidates"], fontsize=6.5)
    ax.set(ylabel="$R^2$ of the stated belief", ylim=(0, 1))
    ax.set_title("The stated belief as a mixture", loc="left")
    figstyle.panel_labels(axes)
    figstyle.save(fig, stem)


if __name__ == "__main__":
    main()
