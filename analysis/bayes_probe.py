"""Does the LLM's state hold the Bayesian posterior that its answer misses? (Notes against reading.)

The LLM departs from Bayes: it gives Bob almost Alice's weight and negations about half. Either its state already
loses that information (the notes are wrong), or the state keeps it and the step from state to answer combines it
wrongly (the reading is wrong). Per state, linear probes on held-out episodes of:
  - the exact Bayesian log-odds (from the clue ids and the stated reliabilities);
  - the LLM's own belief (logit);
  - the evidence each source gave (signed counts of Alice's and of Bob's clues per attribute), and the evidence of
    affirmative and of negated clues, which Bayes weighs alike.
The same targets are predicted from the LLM's answer alone (its four belief logits), with a linear map and with
gradient boosting, the best recalibration of what it says. If a linear probe of the state predicts Bayes better than
any function of the answer, the information was in the notes and the answer throws it away. The evidence components
show whether the state keeps them apart, which reweighting them needs; the answer is one number per attribute and
cannot (even a Bayesian answer would hardly predict Bob's evidence), so it is no bar for them. States: the recurrent
memory S of the saved Gated DeltaNet layers, and the running mean over turns of the residual stream (turn means).
The same probes are then scored only on the divergent turns, where the answer is off the posterior by more than
DIVERGENT in probability: the R² there, and the mean error in log-odds of the answer and of the probe of each layer's
memory on the same turns. If the probe stays close to the posterior where the answer is not, the state had it.

  python -m analysis.bayes_probe --data data/qwen3.5-2b-memory --out results/qwen3.5-2b-memory
"""
import argparse
import glob
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from figures import figstyle
from games import game
from analysis.probe_layers import ALPHAS, history, logit, probe

RESIDUAL = (3, 6, 9, 12, 15, 18, 21, 23, 24)  # pool index L + 1 = after decoder layer L; 24 is after the final norm
TARGETS = ("Bayes", "LLM belief", "Alice's evidence", "Bob's evidence", "affirmative evidence", "negated evidence")
DIVERGENT = 0.2  # |answer - posterior| in probability above which a (turn, attribute) counts as divergent


def targets(actions, llm):
    """[E, T, 6 * 4]: each target for the four attributes, after each clue."""
    counts = history(actions).reshape(*actions.shape, len(game.ATTR_NAMES), len(game.SOURCE_NAMES), 2)
    w = np.array([math.log(r / (1 - r)) for r in game.SOURCES.values()])
    parts = [np.einsum("etasp,s->eta", counts, w), logit(llm[:, 1:]),
             counts[..., 0, :].sum(-1), counts[..., 1, :].sum(-1), counts[..., 0].sum(-1), counts[..., 1].sum(-1)]
    return np.concatenate(parts, -1)


def from_answer(a_tr, y_tr, a_te, y_te):
    """Test R² per target column from the LLM's four belief logits: linear, and gradient boosting."""
    x_tr, x_te = np.c_[a_tr, np.ones(len(a_tr))], np.c_[a_te, np.ones(len(a_te))]
    linear = r2_score(y_te, x_te @ np.linalg.lstsq(x_tr, y_tr, rcond=None)[0], multioutput="raw_values")
    boosted = np.array([r2_score(y_te[:, j], HistGradientBoostingRegressor(random_state=0).fit(a_tr, y_tr[:, j])
                                 .predict(a_te)) for j in range(y_tr.shape[1])])
    return linear, boosted


def predictions(x_tr, y_tr, x_te):
    """Ridge predictions on the test rows, as probe() but returning them (standardised features, alpha per target)."""
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
    return RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit((x_tr - mu) / sd, y_tr).predict((x_te - mu) / sd)


def on_divergent(pred, y, mask):
    """R² and mean absolute error of the Bayes columns on the divergent (row, attribute) pairs, pooled over
    attributes; pred and y are [rows, 4] log-odds, mask [rows, 4]."""
    p, t = pred[mask], y[mask]
    return {"r2": float(1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()), "mae": float(np.abs(t - p).mean())}


def grouped(r2):
    return {name: float(r2[4 * i:4 * i + 4].mean()) for i, name in enumerate(TARGETS)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--test-frac", type=float, default=0.2)
    args = p.parse_args()

    shards = [np.load(f) for f in sorted(glob.glob(f"{args.data}/shard_*.npz"))]
    layers = [int(x) for x in shards[0]["memory_layers"]]
    actions, llm, bayes = (np.concatenate([s[k] for s in shards]) for k in ("actions", "llm_belief", "bayes_belief"))
    E, T = actions.shape
    y = targets(actions, llm)
    ok = (bayes[:, 1:] > 1e-3) & (bayes[:, 1:] < 1 - 1e-3)  # the stored posterior is clipped by logit() beyond this
    assert np.allclose(y[..., :4][ok], logit(bayes[:, 1:])[ok], atol=1e-4), "Bayes log-odds do not match the data"
    y = y.reshape(E * T, -1)
    answer = logit(llm[:, 1:]).reshape(E * T, -1)
    n_tr = round(E * (1 - args.test_frac)) * T  # rows are episode-major, so this splits by episode
    print(f"{E} episodes, {E * T} states, {E * T - n_tr} test states")

    # divergent turns: the answer off the posterior by more than DIVERGENT, on the test episodes
    y_te, a_te = y[n_tr:, :4], answer[n_tr:]
    mask = np.abs(llm[:, 1:].reshape(E * T, -1)[n_tr:] - bayes[:, 1:].reshape(E * T, -1)[n_tr:]) > DIVERGENT
    res = {"targets": TARGETS, "memory_layers": layers, "residual_layers": [i - 1 for i in RESIDUAL],
           "divergent": {"threshold": DIVERGENT, "fraction": float(mask.mean()),
                         "answer as is": on_divergent(a_te, y_te, mask),
                         "answer, linear": on_divergent(predictions(answer[:n_tr], y[:n_tr, :4], a_te), y_te, mask),
                         "memory S": [], "residual, mean of turns so far": []}}
    print(f"divergent (turn, attribute) pairs: {mask.mean():.1%}; answer as is: {res['divergent']['answer as is']}; "
          f"answer recalibrated: {res['divergent']['answer, linear']}")
    counts = history(actions).reshape(E * T, -1)
    res["clue counts, linear"] = grouped(probe(counts[:n_tr], y[:n_tr], counts[n_tr:], y[n_tr:]))
    linear, boosted = from_answer(answer[:n_tr], y[:n_tr], answer[n_tr:], y[n_tr:])
    res["answer, linear"], res["answer, boosted"] = grouped(linear), grouped(boosted)
    for key in ("clue counts, linear", "answer, linear", "answer, boosted"):
        print(f"{key:28s}", "  ".join(f"{k} {v:.3f}" for k, v in res[key].items()), flush=True)

    res["memory S"] = []
    for i, layer in enumerate(layers):
        x = np.concatenate([s["memory"][:, 1:, i] for s in shards]).astype(np.float32).reshape(E * T, -1)
        res["memory S"].append(grouped(probe(x[:n_tr], y[:n_tr], x[n_tr:], y[n_tr:])))
        res["divergent"]["memory S"].append(on_divergent(predictions(x[:n_tr], y[:n_tr, :4], x[n_tr:]), y_te, mask))
        print(f"   divergent turns: {res['divergent']['memory S'][-1]}")
        print(f"memory S, layer {layer:2d}        ", "  ".join(f"{k} {v:.3f}" for k, v in res["memory S"][-1].items()),
              flush=True)
    res["residual, mean of turns so far"] = []
    for index in RESIDUAL:
        pool = np.concatenate([s["pool"][:, :, index] for s in shards]).astype(np.float32)
        x = (np.cumsum(pool, 1) / np.arange(1, T + 2)[None, :, None])[:, 1:].reshape(E * T, -1)
        res["residual, mean of turns so far"].append(grouped(probe(x[:n_tr], y[:n_tr], x[n_tr:], y[n_tr:])))
        res["divergent"]["residual, mean of turns so far"].append(
            on_divergent(predictions(x[:n_tr], y[:n_tr, :4], x[n_tr:]), y_te, mask))
        print(f"residual running, after {index - 1:2d}  ",
              "  ".join(f"{k} {v:.3f}" for k, v in res["residual, mean of turns so far"][-1].items()), flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "bayes_probe.json").write_text(json.dumps(res, indent=1))
    plot(res, out / "bayes_probe")
    plot_divergent(res, out / "bayes_probe_divergent")


def plot_divergent(res, stem):
    """The posterior on the divergent turns: R² per layer and the error in log-odds, against the answer."""
    figstyle.apply()
    d = res["divergent"]
    fig, axes = plt.subplots(1, 2, figsize=(figstyle.DOUBLE * 0.7, 2.5))
    for ax, key, label in ((axes[0], "r2", "Test R² of the posterior"), (axes[1], "mae", "Mean |error| in log-odds")):
        ax.plot(res["memory_layers"], [r[key] for r in d["memory S"]], color=figstyle.BLUE, marker="o",
                label="memory S, linear probe")
        ax.plot(res["residual_layers"], [r[key] for r in d["residual, mean of turns so far"]],
                color=figstyle.VERMILLION, marker="^", ls="--", label="residual, mean of turns so far")
        ax.axhline(d["answer, linear"][key], color=figstyle.GREY, ls=":", label="the LLM's answer, recalibrated")
        if key == "mae":
            ax.axhline(d["answer as is"][key], color="#333333", ls="-", lw=0.8, label="the LLM's answer as is")
        ax.set(xlabel="Decoder layer", ylabel=label)
    axes[0].set_ylim(0, 1)
    fig.suptitle(f"Turns where the answer is off the posterior by more than {d['threshold']:.0%} "
                 f"({d['fraction']:.0%} of all turns and attributes)", x=0.02, ha="left", fontsize=8)
    fig.legend(*axes[1].get_legend_handles_labels(), loc="outside lower center", ncol=2)
    figstyle.panel_labels(axes)
    figstyle.save(fig, stem)


def plot(res, stem):
    figstyle.apply()
    shown = (("Bayes", "Bayesian posterior"), ("Alice's evidence", "Alice's evidence"),
             ("Bob's evidence", "Bob's evidence"), ("negated evidence", "Negated clues' evidence"))
    fig, axes = plt.subplots(1, len(shown), figsize=(figstyle.DOUBLE, 2.6), sharey=True)
    for ax, (target, title) in zip(axes, shown):
        ax.plot(res["memory_layers"], [r[target] for r in res["memory S"]], color=figstyle.BLUE, marker="o",
                label="memory S, linear probe")
        ax.plot(res["residual_layers"], [r[target] for r in res["residual, mean of turns so far"]],
                color=figstyle.VERMILLION, marker="^", ls="--", label="residual, mean of turns so far, linear probe")
        if target == "Bayes":  # one number per attribute cannot keep two components apart, so no bar for those
            ax.axhline(res["answer, boosted"][target], color=figstyle.GREY, ls=":",
                       label="the LLM's answer, best recalibration")
        ax.set(xlabel="Decoder layer", ylim=(0, 1))
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("Test R²")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
    figstyle.panel_labels(axes)
    figstyle.save(fig, stem)


if __name__ == "__main__":
    main()
