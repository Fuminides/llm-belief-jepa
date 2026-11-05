"""Learning the dynamics: what the model expects a move to do, against the engine and a Bayesian learner.

The consequence question (dungeon_states.py --consequences) asks before every move what each of the four moves would
do: safe, damage, die, slide or wall. In the stated condition the rules say what every kind of cell does; in the
hidden condition (effects "hidden") they list the effects without saying which kind has which, and that any kind
can have any of them, so the belief about a kind is a distribution over four effects (nothing, damage, death, a
slide) and the Bayesian reference is one posterior per kind, from a uniform prior, updated by each consequence
observed: a slide shows a slippery kind; ending a turn on a kind without damage rules damage and death out for it;
entering a kind without sliding is (1 - p) times as likely under the slippery effect as under the others. Empty
cells, doors and the exit are not hidden kinds. Answers about moves into a hidden kind are scored by the
probability given to the engine's answer, by kind and by how many times that kind's effect has been observed so
far, against: the same kinds in the stated condition, the Bayesian learner, and a player that never updates (the
model's own mean answer about that kind before observing it).

With the cue (level.cue), the cracked cells inside the doors of dangerous rooms: the answer about stepping onto a
cracked cell (it does nothing), and the probability of moving through a door with a cracked cell behind it against
a door without, before and after the first time the player is hurt in a room it entered past a cracked cell.

  python -m analysis.dungeon_learning --model Qwen/Qwen3.5-27B    # results/dungeon/learning/<model>.{json,png,pdf}
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figures import figstyle
from games.dungeon import CONSEQUENCES, MOVES, Dungeon, State, step
from collect.dungeon_states import DATA

EFFECTS = ("safe", "damage", "die", "slide")  # the hidden kinds' possible effects, in the consequence words
CONDITIONS = ("medium", "medium_hidden", "medium_cue", "medium_cue_hidden")
OUT = Path("results/dungeon/learning")


def load(difficulty, name):
    """The games of a consequence run: per game its record and the [T, 4, 5] answers."""
    root = DATA / difficulty / "consequences" / name.replace("/", "__")
    records = json.loads((root / "games.json").read_text())
    games = []
    for path in sorted(glob.glob(f"{root}/shard_*.npz")):
        shard = np.load(path)
        for g, seed in enumerate(shard["seeds"]):
            games.append((records[str(seed)], shard["consequence"][g, :shard["turns"][g]]))
    return games


class Learner:
    """The Bayesian learner of the hidden condition: per hidden kind, a posterior over EFFECTS, from a uniform prior."""

    def __init__(self, dungeon, kinds):
        self.d = dungeon
        self.post = {k: np.ones(len(EFFECTS)) / len(EFFECTS) for k in kinds}
        slip = [t["slip_probability"] for t in dungeon.terrain.values() if t.get("slip_probability")]
        p = slip[0] if slip else 0.5
        self.slid, self.not_slid = np.array([0, 0, 0, 1.0]), np.array([1, 1, 1, 1 - p])
        self.seen = {k: 0 for k in kinds}  # turns on which each kind's effect was observed so far

    def update(self, level, entered, end, events):
        """One turn: the move entered a cell (None after a bump or the word) and the turn ended on end. Entering shows
        whether the kind slides; ending the turn on a kind shows whether it hurts or kills."""
        likelihood = {}
        if entered is not None and level[entered] in self.post:
            likelihood[level[entered]] = self.slid if "slide" in events else self.not_slid
        if level[end] in self.post:
            here = (np.array([0, 0, 1.0, 0]) if "lethal" in events else np.array([0, 1.0, 0, 0]) if "damage" in events
                    else np.array([1.0, 0, 0, 1]))
            likelihood[level[end]] = likelihood.get(level[end], 1) * here
        for kind, lik in likelihood.items():
            self.post[kind] = self.post[kind] * lik / (self.post[kind] * lik).sum()
            self.seen[kind] += 1

    def answer(self, level, cell, move):
        """P(safe, damage, die, slide, wall) for the move."""
        nxt = step(cell, move)
        q = np.zeros(len(CONSEQUENCES))
        if not self.d.blocks(level, nxt) and level[nxt] in self.post:
            q[:len(EFFECTS)] = self.post[level[nxt]]
        else:
            q[CONSEQUENCES.index(self.d.consequence(level, State(cell, 0, None), move))] = 1
        return q


def rows_of(difficulty, name):
    """One row per (turn, move into a hidden kind): kind, observations of it so far, P(true answer) for the model
    and for the Bayesian learner, and the turn."""
    d = Dungeon(difficulty=difficulty)
    level_cfg = d.config["level"]
    kinds = list(level_cfg["density"]) + ([level_cfg["cue"]["terrain"]] if level_cfg.get("cue") else [])
    hidden = d.config.get("effects") == "hidden"
    rows = []
    for record, answers in load(difficulty, name):
        level = d.generate(record["seed"])
        learner = Learner(d, kinds)
        log = record["log"]
        for t, turn in enumerate(log):
            cell = tuple(turn["cell"])
            for m, move in enumerate(MOVES):
                nxt = step(cell, move)
                if d.blocks(level, nxt) or level[nxt] not in kinds:
                    continue
                truth = CONSEQUENCES.index(turn["truth"][m])
                rows.append({"kind": level[nxt], "seen": learner.seen[level[nxt]], "turn": t,
                             "model": float(answers[t, m, truth]),
                             "bayes": float(learner.answer(level, cell, move)[truth]) if hidden else 1.0,
                             "answer": answers[t, m].tolist()})
            end = tuple(log[t + 1]["cell"]) if t + 1 < len(log) else tuple(record["final"])
            nxt = step(cell, turn["move"]) if turn["move"] in MOVES else None
            entered = None if nxt is None or d.blocks(level, nxt) else nxt
            learner.update(level, entered, end, turn["events"])
    return rows


def summarise(rows, kinds):
    out = {}
    for kind in kinds + ["all"]:
        sel = [r for r in rows if kind in ("all", r["kind"])]
        if not sel:
            continue
        prior = [r for r in sel if r["seen"] == 0]
        never = np.mean([r["model"] for r in prior]) if prior else np.nan  # its own answer before observing the kind
        out[kind] = {"n": len(sel),
                     "by_seen": {k: {"n": len(s), "model": float(np.mean([r["model"] for r in s])),
                                     "bayes": float(np.mean([r["bayes"] for r in s]))}
                                 for k in (0, 1, 2, 3) if (s := [r for r in sel if min(r["seen"], 3) == k])},
                     "never_updates": float(never)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-27B")
    args = p.parse_args()
    res = {}
    for difficulty in CONDITIONS:
        if not (DATA / difficulty / "consequences" / args.model.replace("/", "__") / "games.json").exists():
            continue
        d = Dungeon(difficulty=difficulty)
        kinds = list(d.config["level"]["density"]) + ([d.config["level"]["cue"]["terrain"]]
                                                       if d.config["level"].get("cue") else [])
        rows = rows_of(difficulty, args.model)
        res[difficulty] = summarise(rows, kinds)
        for kind, s in res[difficulty].items():
            curve = "  ".join(f"{k}: {v['model']:.2f} (Bayes {v['bayes']:.2f}, n {v['n']})" for k, v in s["by_seen"].items())
            print(f"{difficulty:18s} {kind:8s} P(true answer) by observations so far  {curve}  | never updates "
                  f"{s['never_updates']:.2f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / args.model.split("/")[-1]
    Path(f"{stem}.json").write_text(json.dumps(res, indent=1))
    plot(res, stem)


def effect_words(d, kind):
    """A kind's effect in a few words, from the config, for the figure's reminder line."""
    t = d.terrain[kind]
    if t.get("lethal"):
        return "death on entry"
    if t.get("damage"):
        return f"{t['damage']} {'life' if t['damage'] == 1 else 'lives'} lost per turn ended on it"
    if t.get("slip_probability"):
        return f"{t['slip_probability']:.0%} chance to slide {t['slip_cells']} more cell{'s' if t['slip_cells'] > 1 else ''}"
    return "nothing"


def cell_title(ax, d, kind, y=1.04):
    """The panel's title: the cell's sprite, as the paper's dungeon figure draws it, and its name in the rules."""
    from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, OffsetImage, TextArea
    from games.pixelart import tile
    title = HPacker(children=[OffsetImage(np.asarray(tile(kind)), zoom=0.9, interpolation="nearest"),
                              TextArea(d.terrain[kind]["name"], textprops=dict(fontsize=8))], pad=0, sep=3, align="center")
    ax.add_artist(AnchoredOffsetbox(loc="lower center", child=title, pad=0, borderpad=0.3, frameon=False,
                                    bbox_to_anchor=(0.5, y), bbox_transform=ax.transAxes))


def plot(res, stem):
    figstyle.apply()
    d = Dungeon(difficulty="medium")
    kinds = [k for k in ("fire", "pit", "ice", "cracked") if any(k in r for r in res.values())]
    fig, axes = plt.subplots(1, len(kinds), figsize=(figstyle.DOUBLE, 2.5), sharey=True, squeeze=False)
    fig.get_layout_engine().set(rect=(0, 0.13, 1, 1))  # room under the panels for the reminder line
    styles = {"medium": (figstyle.GREY, "-"), "medium_hidden": (figstyle.BLUE, "-"),
              "medium_cue": (figstyle.GREY, "--"), "medium_cue_hidden": (figstyle.BLUE, "--")}
    for ax, kind in zip(axes[0], kinds):
        for difficulty, r in res.items():
            if kind not in r:
                continue
            xs = sorted(int(k) for k in r[kind]["by_seen"])
            colour, ls = styles[difficulty]
            ax.plot(xs, [r[kind]["by_seen"][str(k)]["model"] for k in xs], "o", ls=ls, color=colour, ms=3,
                    label=f"model, {difficulty.replace('medium_', '').replace('medium', 'stated')}")
            if "hidden" in difficulty:
                ax.plot(xs, [r[kind]["by_seen"][str(k)]["bayes"] for k in xs], "^", ls=ls, color=figstyle.GREEN, lw=1, ms=3,
                        label=f"Bayesian learner, {difficulty.replace('medium_', '')}")
        ax.set(xticks=[0, 1, 2, 3], xticklabels=["0", "1", "2", "3+"], ylim=(0, 1), xlim=(-0.4, 3.4))
        cell_title(ax, d, kind)
    fig.text(0.5, 0.075, "Times the kind's effect was observed so far", ha="center", va="bottom", fontsize=8)
    fig.text(0.5, 0.01, "   ".join(f"{d.terrain[k]['name']}: {effect_words(d, k)}" for k in kinds), ha="center",
             va="bottom", fontsize=6.5, color=figstyle.GREY)
    axes[0][0].set_ylabel("P(true consequence)")
    handles = {}
    for ax in axes[0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    axes[0][-1].legend(handles.values(), handles.keys(), fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    figstyle.panel_labels(axes[0])
    figstyle.save(fig, stem)


if __name__ == "__main__":
    main()
