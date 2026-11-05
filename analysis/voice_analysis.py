"""Does the model heed the voice in the dark? From the saved medium_voice games (play_dungeon.py).

On every turn the voice speaks, the move probabilities are split into the mass on moves that bring the player closer
to the claimed cell and the rest. Since the voice picks a hidden cell regardless of what is there, the difference
between "pit" and "no pit" claims is the claim's effect, to compare with the exact posterior (dungeon.pit_posterior:
with 5% pits and a 70% voice, a pit claim leaves 0.11, a no-pit claim 0.02, so a Bayesian player hardly moves).
The level regenerates from the seed, so the turn a claimed cell comes into view, and whether the claim was true, is
known; claims made after an earlier claim of the game proved true or false are compared.

  python -m analysis.voice_analysis            # every model in results/dungeon/medium_voice, text to voice_analysis.txt there
"""
import glob
import json
from pathlib import Path

import numpy as np

from games.dungeon import MOVES, Dungeon, pit_posterior, step

NAMES = list(MOVES)

DIR = Path("results/dungeon/medium_voice")


def towards(cell, target):
    """Indices of the moves that reduce the Manhattan distance from cell to target."""
    d0 = abs(cell[0] - target[0]) + abs(cell[1] - target[1])
    return [i for i, m in enumerate(NAMES) if abs(step(cell, m)[0] - target[0]) + abs(step(cell, m)[1] - target[1]) < d0]


def claims(dungeon, game):
    """One row per spoken turn: mass towards the claimed cell, the claim, its truth, and what earlier claims of the
    game had been revealed by then (None, True = a revealed claim was right, False = one was wrong)."""
    level = dungeon.generate(game["seed"])
    cells = [tuple(t["cell"]) for t in game["log"]] + [tuple(game["final"])]
    rows, pending, revealed = [], [], []
    for i, t in enumerate(game["log"]):
        seen = dungeon.visible(level, cells[i])
        for a in pending[:]:
            if tuple(a["cell"]) in seen:
                revealed.append(a["claim"] == a["truth"])
                pending.remove(a)
        a = t.get("advice")
        if not a:
            continue
        p = np.array(t["p"])
        idx = towards(cells[i], tuple(a["cell"]))
        rows.append({"towards": p[idx].sum() / p.sum(), "n_towards": len(idx), "claim": a["claim"], "truth": a["truth"],
                     "after": None if not revealed else revealed[-1], "moved_towards": NAMES.index(t["move"]) in idx})
        pending.append(a)
    return rows


def report(path, out):
    r = json.loads(Path(path).read_text())
    d = Dungeon(r["config"])
    prior, rel = d.config["level"]["density"][d.lethal], d.advisor["reliability"]
    out(f"{r['model']}  ({d.advisor['name']}, right {rel:.0%}, speaks {d.advisor['chance']:.0%} of the turns; pits {prior:.0%}: "
        f"posterior after 'pit' {pit_posterior(prior, rel, True):.3f}, after 'no pit' {pit_posterior(prior, rel, False):.3f})")
    for policy, res in r["results"].items():
        rows = [row for g in res["games"] for row in claims(d, g)]
        # Only turns where some but not all moves lead towards the cell carry information.
        rows = [x for x in rows if 0 < x["n_towards"] < len(MOVES)]
        m = lambda sel, key="towards": (np.mean([x[key] for x in rows if sel(x)]), sum(sel(x) for x in rows))
        pit, no = m(lambda x: x["claim"]), m(lambda x: not x["claim"])
        out(f"  {policy:7s} {len(rows)} spoken turns. Mass on moves towards the named cell: 'pit' {pit[0]:.3f} (n={pit[1]}), "
            f"'no pit' {no[0]:.3f} (n={no[1]}), difference {no[0] - pit[0]:+.3f}; "
            f"a move towards it was taken {m(lambda x: x['claim'], 'moved_towards')[0]:.2f} vs "
            f"{m(lambda x: not x['claim'], 'moved_towards')[0]:.2f} of the times")
        for after, label in ((True, "an earlier claim proved right"), (False, "an earlier claim proved wrong"), (None, "nothing revealed yet")):
            pit, no = m(lambda x: x["claim"] and x["after"] is after), m(lambda x: not x["claim"] and x["after"] is after)
            if pit[1] and no[1]:
                out(f"           after {label}: 'pit' {pit[0]:.3f} (n={pit[1]}), 'no pit' {no[0]:.3f} (n={no[1]}), "
                    f"difference {no[0] - pit[0]:+.3f}")
        true, false = m(lambda x: x["claim"] and x["truth"]), m(lambda x: x["claim"] and not x["truth"])
        out(f"           'pit' claims that were true {true[0]:.3f} (n={true[1]}) vs false {false[0]:.3f} (n={false[1]}): "
            f"the model cannot see the cell, so these should agree")


if __name__ == "__main__":
    lines = []
    out = lambda s: (print(s), lines.append(s))
    for path in sorted(glob.glob(str(DIR / "*.json"))):
        report(path, out)
    (DIR / "voice_analysis.txt").write_text("\n".join(lines) + "\n")
