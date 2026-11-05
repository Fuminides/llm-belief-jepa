"""How the phrasing of a clue changes the LLM's update, by designed convolutedness and by text measures.

Needs a run with --paraphrase and a plain run with the same seed and shard size (identical clue sequences).
  A. Weight of a clue (log-odds) by convolutedness level, affirmative vs negated, from a fitted observer.
  B. Weight of each template against the number of negation cues in its text.
  C. Paired divergence: mean |logit(belief) difference| between the two runs per turn, for attributes that have or
     have not received a level-3 clue so far.
"""
import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figures import figstyle
from games import game
from analysis.fit_bayes import r2
from analysis.probe_layers import logit

NEGATION = {"not", "false", "denies"}
SUBORDINATE = {"that", "who", "if", "when", "whether"}


def text_measures(action, template):
    words = re.findall(r"[a-z']+", game.clue_text(action, template).lower())
    return {"words": len(words), "negations": sum(w in NEGATION for w in words),
            "subordinates": sum(w in SUBORDINATE for w in words)}


def load(data_dir):
    shards = [np.load(f) for f in sorted(glob.glob(f"{data_dir}/shard_*.npz"))]
    out = {k: np.concatenate([s[k] for s in shards]) for k in ("llm_belief", "actions")}
    if "templates" in shards[0].files:
        out["templates"] = np.concatenate([s["templates"] for s in shards])
    return out


def design(actions, column):
    """[E, T, A, A + n]: one-hot attribute and, per attribute, signed clue counts in the column chosen per clue."""
    E, T = actions.shape
    A = len(game.ATTR_NAMES)
    attr, value, _, _ = np.moveaxis(np.array(game.ACTIONS)[actions], -1, 0)
    sign = np.where(value == 0, 1, -1)
    n = column.max() + 1
    x = np.zeros((E, T, A, A + n))
    x[..., :A] = np.eye(A)
    e = np.arange(E)
    for t in range(T):
        x[e, t:, attr[:, t], A + column[:, t]] += sign[:, t, None]
    return x


def fit(x, y):
    return np.linalg.lstsq(x.reshape(-1, x.shape[-1]), y.ravel(), rcond=None)[0]


def boot(x, y, n_boot, rng):
    coef = fit(x, y)
    reps = np.array([fit(x[i], y[i]) for i in (rng.integers(len(x), size=len(x)) for _ in range(n_boot))])
    return coef, *np.percentile(reps, (2.5, 97.5), axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--plain", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--boot", type=int, default=200)
    args = p.parse_args()

    d, plain = load(args.data), load(args.plain)
    assert (d["actions"] == plain["actions"]).all(), "the two runs must share their clue sequences"
    actions, templates = d["actions"], d["templates"]
    E, T = actions.shape
    A = len(game.ATTR_NAMES)
    negated = np.array(game.ACTIONS)[actions][..., 2].astype(int)
    level = np.array([lvl for lvl, _ in game.TEMPLATES])[templates]
    y = logit(d["llm_belief"][:, 1:])
    rng = np.random.default_rng(0)

    # Observers of increasing detail: phrasing only, phrasing x level, phrasing x template.
    x_phr = design(actions, negated)
    x_lvl = design(actions, negated * len(game.LEVELS) + level)
    x_tpl = design(actions, negated * len(game.TEMPLATES) + templates)
    coef_lvl, lo_lvl, hi_lvl = boot(x_lvl, y, args.boot, rng)
    coef_tpl = fit(x_tpl, y)
    res = {"episodes": E,
           "r2_observer_phrasing": float(r2(y, x_phr @ fit(x_phr, y))),
           "r2_observer_phrasing_level": float(r2(y, x_lvl @ coef_lvl)),
           "r2_observer_phrasing_template": float(r2(y, x_tpl @ coef_tpl))}
    w_lvl = [c[A:].reshape(2, len(game.LEVELS)) for c in (coef_lvl, lo_lvl, hi_lvl)]
    res["weight_by_level"] = {ph: w_lvl[0][n].tolist() for n, ph in enumerate(("affirmative", "negated"))}
    w_tpl = coef_tpl[A:].reshape(2, len(game.TEMPLATES))
    example = {0: game.ACTIONS.index((0, 0, False, 0)), 1: game.ACTIONS.index((0, 0, True, 0))}
    res["templates"] = [{"template": i, "level": lvl, "phrasing": ph, "weight": float(w_tpl[n, i]),
                         **text_measures(example[n], i)}
                        for i, (lvl, _) in enumerate(game.TEMPLATES) for n, ph in enumerate(("affirmative", "negated"))]

    # Paired divergence per turn, split by whether the attribute has had a level-3 clue so far.
    attr = np.array(game.ACTIONS)[actions][..., 0]
    had3 = np.zeros((E, T + 1, A), bool)
    for t in range(T):
        had3[:, t + 1] = had3[:, t]
        had3[np.arange(E), t + 1, attr[:, t]] |= level[:, t] == 3
    gap = np.abs(logit(d["llm_belief"]) - logit(plain["llm_belief"]))
    res["divergence"] = {name: [float(gap[:, t][mask[:, t]].mean()) if mask[:, t].any() else None
                                for t in range(T + 1)]
                         for name, mask in (("no level-3 clue", ~had3), ("had a level-3 clue", had3))}
    print(json.dumps({k: v for k, v in res.items() if k != "templates"}, indent=1))

    figstyle.apply()
    fig, axes = plt.subplots(1, 3, figsize=(figstyle.DOUBLE, 2.7))
    styles = {"affirmative": (figstyle.BLUE, "o", "-", None), "negated": (figstyle.VERMILLION, "s", "--", "white")}

    ax = axes[0]
    for n, (ph, (colour, marker, ls, mfc)) in enumerate(styles.items()):
        yerr = np.clip([w_lvl[0][n] - w_lvl[1][n], w_lvl[2][n] - w_lvl[0][n]], 0, None)
        ax.errorbar(game.LEVELS, w_lvl[0][n], yerr=yerr,
                    color=colour, marker=marker, ls=ls, mfc=mfc or colour, capsize=2, lw=1.2, label=ph)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(game.LEVELS)
    ax.set(xlabel="Convolutedness level", ylabel="Weight of a clue (log-odds)")
    ax.set_title("Update by level (95% CI)", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)

    ax = axes[1]
    level_colours = plt.cm.viridis(np.linspace(0.1, 0.85, len(game.LEVELS)))
    for row in res["templates"]:
        colour, marker = level_colours[row["level"]], "o" if row["phrasing"] == "affirmative" else "s"
        ax.plot(row["negations"] + (0.08 if row["phrasing"] == "negated" else -0.08), row["weight"], marker,
                color=colour, mfc=colour if row["phrasing"] == "affirmative" else "white", mew=1.2)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(range(max(r["negations"] for r in res["templates"]) + 1))
    ax.set(xlabel="Negation cues in the clue", ylabel="Weight of a clue (log-odds)")
    ax.set_title("Per template", loc="left")
    handles = [plt.Line2D([], [], color=c, marker="o", ls="none", label=f"level {i}") for i, c in enumerate(level_colours)]
    handles += [plt.Line2D([], [], color="black", marker="o", ls="none", label="affirmative"),
                plt.Line2D([], [], color="black", marker="s", mfc="white", ls="none", label="negated")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)

    ax = axes[2]
    for (name, values), (colour, marker, ls) in zip(res["divergence"].items(),
                                                    ((figstyle.BLUE, "o", "-"), (figstyle.VERMILLION, "s", "--"))):
        turns = [t for t, v in enumerate(values) if v is not None]
        ax.plot(turns, [values[t] for t in turns], color=colour, marker=marker, ls=ls, label=name)
    ax.set(xlabel="Turn", ylabel="|Δ logit| vs plain phrasing")
    ax.set_title("Same clues, other words", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1)

    figstyle.panel_labels(axes)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "paraphrase.json").write_text(json.dumps(res, indent=1))
    figstyle.save(fig, out / "paraphrase")


if __name__ == "__main__":
    main()
