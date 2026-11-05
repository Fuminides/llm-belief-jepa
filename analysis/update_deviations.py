"""The update rule the latent model has learned, against Bayes and the LLM: the list of systematic deviations (the
working doc's test list, H1 and H4). explain.py's update analysis gives the mean shift and its slope on the belief
per clue type; this adds what else the shift depends on, which a Bayesian update (a constant log(r / (1 - r)) per
source) does not: how many clues the attribute has already had, and whether the clue agrees with the current lean.

At every test state before a clue, each of the 32 clues is applied through the predictor and the belief decoded
(counterfactual shifts towards the asserted value); the LLM's shifts are those of the clues that really arrived.
Each deviation is a least-squares coefficient of the shift on clue type, belief before, earlier clues on the
attribute and agreement, fitted separately to the model's counterfactual shifts and to the LLM's real ones.

  python -m analysis.update_deviations   # results/qwen3.5-2b-memory-mid/update_deviations.{json,png,pdf}
"""
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from figures import figstyle
from games import game
from latent import jepa
from analysis.probe_layers import logit

DATA, LAYER = "data/qwen3.5-2b-memory-mid", 14
OUT = Path("results/qwen3.5-2b-memory-mid")
TYPES = [f"{s} {'neg.' if n else 'aff.'}" for s in game.SOURCE_NAMES for n in (0, 1)]


def earlier_counts(a):
    """[E, T, A]: clues on each attribute before clue t."""
    E, T = a.shape
    attr = np.array(game.ACTIONS)[a][..., 0]
    counts = np.zeros((E, T, len(game.ATTR_NAMES)))
    for t in range(1, T):
        counts[:, t] = counts[:, t - 1]
        counts[np.arange(E), t, attr[:, t - 1]] += 1
    return counts


def regress(shift, ctype, before, earlier, agrees):
    """Shift on clue-type intercepts, belief before, earlier clues on the attribute (capped at 3) and agreement."""
    x = np.column_stack([np.eye(len(TYPES))[ctype], before, np.minimum(earlier, 3), agrees.astype(float)])
    coef = np.linalg.lstsq(x, shift, rcond=None)[0]
    names = [f"intercept, {t}" for t in TYPES] + ["per log-odds of belief before", "per earlier clue on the attribute",
                                                  "agrees with the current lean"]
    return dict(zip(names, coef.tolist()))


def main():
    shards = [np.load(f) for f in sorted(glob.glob(f"{DATA}/shard_*.npz"))]
    li = list(shards[0]["memory_layers"]).index(LAYER)
    h = np.concatenate([s["memory"][:, :, li] for s in shards]).astype(np.float32)
    a = np.concatenate([s["actions"] for s in shards])
    lo = logit(np.concatenate([s["llm_belief"] for s in shards]))
    E, T = a.shape
    n = round(E * 0.8)
    mu, sd = h[:n].mean((0, 1)), h[:n].std((0, 1)) + 1e-6
    hs = torch.tensor((h[n:] - mu) / sd)
    model = jepa.LatentModel(h.shape[-1])
    model.load_state_dict(torch.load(f"{DATA}/latent_memory_layer{li}_mlp.pt", map_location="cpu"))
    model.eval()
    A_te = a[n:]
    with torch.no_grad():
        z = model.enc(hs)[:, :-1]
        base = model.read(z).numpy()
        after = np.stack([model.read(model.step(z, torch.full(z.shape[:2], k))).numpy() for k in range(game.N_ACTIONS)])
    attr_k, value_k, neg_k, src_k = np.array(game.ACTIONS).T
    sign_k, ctype_k = np.where(value_k == 0, 1, -1), src_k * 2 + neg_k
    earlier = earlier_counts(A_te)
    Et, Tt = A_te.shape
    # counterfactual: every clue at every state
    cf = {"shift": [], "ctype": [], "before": [], "earlier": [], "agrees": []}
    for k in range(game.N_ACTIONS):
        bef = sign_k[k] * base[..., attr_k[k]]
        cf["shift"].append((sign_k[k] * (after[k][..., attr_k[k]] - base[..., attr_k[k]])).ravel())
        cf["ctype"].append(np.full(Et * Tt, ctype_k[k]))
        cf["before"].append(bef.ravel())
        cf["earlier"].append(earlier[..., attr_k[k]].ravel())
        cf["agrees"].append((bef > 0).ravel())
    cf = {key: np.concatenate(v) for key, v in cf.items()}
    # the LLM's real clues
    e, t = np.indices(A_te.shape)
    at, sg, ct = attr_k[A_te], sign_k[A_te], ctype_k[A_te]
    bef = sg * lo[n:, :-1][e, t, at]
    real = {"shift": (sg * (lo[n:, 1:][e, t, at] - lo[n:, :-1][e, t, at])).ravel(), "ctype": ct.ravel(),
            "before": bef.ravel(), "earlier": earlier[e, t, at].ravel(), "agrees": (bef > 0).ravel()}
    with torch.no_grad():  # the model's prediction for the real clues, to compare like with like
        pred = model.read(model.step(z, torch.tensor(A_te))).numpy()
    real_pred = dict(real, shift=(sg * (pred[e, t, at] - base[e, t, at])).ravel())
    r = np.array([game.SOURCES[s] for s in game.SOURCE_NAMES])
    bayes = {f"intercept, {t}": float(np.log(r[i // 2] / (1 - r[i // 2]))) for i, t in enumerate(TYPES)}
    bayes.update({"per log-odds of belief before": 0.0, "per earlier clue on the attribute": 0.0,
                  "agrees with the current lean": 0.0})
    res = {"layer": LAYER, "test_episodes": E - n, "bayes": bayes,
           "model, every clue at every state": regress(**cf),
           "model, the real clues": regress(**real_pred),
           "LLM, the real clues": regress(**real)}
    by_earlier = lambda d: [float(d["shift"][(d["ctype"] == 0) & (np.minimum(d["earlier"], 3) == j)].mean())
                            for j in range(4)]
    res["mean shift of an affirmative Alice clue by earlier clues on the attribute (0, 1, 2, 3+)"] = {
        "model, every clue at every state": by_earlier(cf), "LLM, the real clues": by_earlier(real)}
    print(json.dumps(res, indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "update_deviations.json").write_text(json.dumps(res, indent=1))
    plot(res)


def plot(res):
    figstyle.apply()
    fig, axes = plt.subplots(1, 2, figsize=(figstyle.DOUBLE, 2.5), gridspec_kw={"width_ratios": [1.6, 1]})
    who = [("bayes", figstyle.GREY), ("model, every clue at every state", figstyle.BLUE),
           ("model, the real clues", figstyle.SKY), ("LLM, the real clues", figstyle.VERMILLION)]
    labels = {"bayes": "Bayes", "model, every clue at every state": "anchored JEPA, all 32 clues at every state, read or not",
              "model, the real clues": "anchored JEPA, the clues the LLM read", "LLM, the real clues": "LLM, the clues it read"}
    keys = list(res["bayes"])
    w = 0.8 / len(who)
    for i, (name, colour) in enumerate(who):
        axes[0].bar(np.arange(len(keys)) + (i - (len(who) - 1) / 2) * w, [res[name][k] for k in keys], w * 0.9,
                    color=colour, label=labels[name])
    axes[0].axhline(0, color="black", lw=0.5)
    axes[0].axvline(3.5, color=figstyle.GREY, lw=0.5, ls="--")  # the four clue types | what else the shift depends on
    short = {"per log-odds of belief before": "belief before\nseeing the clue", "per earlier clue on the attribute": "earlier\nclues",
             "agrees with the current lean": "agrees with\nthe LLM belief"}
    axes[0].set_xticks(range(len(keys)), [short.get(k, k.replace("intercept, ", "").replace(" ", "\n")) for k in keys],
                       fontsize=6)
    axes[0].set(ylabel="Coefficient (log-odds)")
    axes[0].set_title("What the shift towards the asserted value depends on", loc="left")
    d = res["mean shift of an affirmative Alice clue by earlier clues on the attribute (0, 1, 2, 3+)"]
    for name, colour in who[1:2] + who[3:]:
        axes[1].plot(range(4), d[name], color=colour, marker="o", label=labels[name])
    axes[1].axhline(res["bayes"]["intercept, Alice aff."], color=figstyle.GREY, ls=":", lw=1, label="Bayes")
    axes[1].set_xticks(range(4), ["0", "1", "2", "3+"])
    axes[1].set(xlabel="Earlier clues on the attribute", ylabel="Shift, Alice aff. (log-odds)", ylim=(0, None))
    axes[1].set_title("Alice affirmative", loc="left")
    handles = {}  # one shared legend under the panels; Bayes is the grey bar in A and the grey dotted line in B
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    fig.legend(handles.values(), handles.keys(), loc="outside lower center", ncol=4, fontsize=6.5)
    figstyle.panel_labels(axes)
    figstyle.save(fig, OUT / "update_deviations")


if __name__ == "__main__":
    main()
