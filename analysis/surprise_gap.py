"""Surprise as a diagnostic, and the gap between the belief the state carries and the belief the LLM states (H5).

Same split as explain.py (first 80% of episodes to train, the rest to test), the latent model explain.py trained
on the memory of layer 14 (data/qwen3.5-2b-memory-mid, latent_memory_layer5_mlp.pt), plain clues.

Internal against stated belief. The belief the state carries is read two ways: a ridge probe from S to the stated
belief logits (per layer, on the 18 layers of data/qwen3.5-2b-memory-all, which has the same clues and the same
answers), and the latent model's belief decoder R(E(S)) at layer 14. The gap is the probe's or the decoder's error
on test episodes, in log-odds: per layer and turn; on the attribute the last clue was about, per clue type, and
whether the clue contradicts an earlier clue on that attribute; signed, whether the stated belief is more extreme
than the internal one; and whether the internal belief is closer to the Bayesian posterior than the stated one.

Surprise. Per real transition, the latent model's one-step prediction error ||ẑ_{t+1} − E(h_{t+1})||², z-scored
within the turn (it grows with the turn). Three targets: a misread clue (the LLM's shift on the clue's attribute is
in the top tenth of distances from the usual shift for its clue type and its belief before, a per-type linear fit),
an answer wrong after the clue (on the clue's attribute, the stated belief and Bayes fall on different sides of 0.5
while Bayes is at least 73% sure), and a final answer wrong (per episode and attribute, after the last clue, the same
rule; detectors averaged over the clues on that attribute). Detectors: surprise, the entropy of the stated answer,
the probe's lack of confidence (minus its |logit|), and the internal-stated gap itself. Scored by AUROC.

  python -m analysis.surprise_gap   # results/qwen3.5-2b-memory-mid/surprise_gap.{json,png,pdf}
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

from figures import figstyle
from games import game
from latent import jepa
from analysis.probe_layers import logit

MID, ALL = "data/qwen3.5-2b-memory-mid", "data/qwen3.5-2b-memory-all"
LAYER = 14
OUT = Path("results/qwen3.5-2b-memory-mid")
TYPES = [f"{s} {'neg.' if n else 'aff.'}" for s in game.SOURCE_NAMES for n in (0, 1)]


def ridge_probe(x, y, n_train, alpha=1e3):
    """Ridge from standardised x [E, T+1, d] to y [E, T+1, 4], trained on the first n_train episodes; predictions for
    every episode."""
    E, T1, d = x.shape
    mu, sd = x[:n_train].mean((0, 1)), x[:n_train].std((0, 1)) + 1e-6
    xs = ((x - mu) / sd).reshape(-1, d)
    model = Ridge(alpha=alpha).fit(xs[:n_train * T1], y[:n_train].reshape(-1, y.shape[-1]))
    return model.predict(xs).reshape(E, T1, -1)


def pick_alpha(x, y, n_train):
    """Ridge strength by a validation split inside the training episodes."""
    n_fit = round(n_train * 0.8)
    best = None
    for alpha in (1e1, 1e2, 1e3, 1e4, 1e5):
        pred = ridge_probe(x[:n_train], y[:n_train], n_fit, alpha)
        err = ((pred[n_fit:] - y[n_fit:n_train]) ** 2).mean()
        best = (err, alpha) if best is None or err < best[0] else best
    return best[1]


def clue_frame(a):
    attr, value, negated, source = np.moveaxis(np.array(game.ACTIONS)[a], -1, 0)
    sign = np.where(value == 0, 1, -1)
    E, T = a.shape
    contradicts = np.zeros((E, T), bool)
    for t in range(1, T):
        contradicts[:, t] = ((attr[:, :t] == attr[:, t:t + 1]) & (value[:, :t] != value[:, t:t + 1])).any(1)
    return attr, sign, source * 2 + negated, contradicts


def auroc(target, score):
    target, score = np.ravel(target), np.ravel(score)
    return float(roc_auc_score(target, score)) if 0 < target.mean() < 1 else None


def main():
    global OUT
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=MID, help="the collection whose layer-14 memory the latent model reads")
    p.add_argument("--all", default=ALL, help="the 18-layer collection with the same clues, or none: then the "
                   "probes run on the collection's own memory layers")
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()
    OUT = Path(args.out)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mid = [np.load(f) for f in sorted(glob.glob(f"{args.data}/shard_*.npz"))]
    templates = np.concatenate([s["templates"] for s in mid]) if "templates" in mid[0].files else None
    li = list(mid[0]["memory_layers"]).index(LAYER)
    h = np.concatenate([s["memory"][:, :, li] for s in mid]).astype(np.float32)
    a = np.concatenate([s["actions"] for s in mid])
    b = np.concatenate([s["llm_belief"] for s in mid]).clip(1e-3, 1 - 1e-3)
    bayes = logit(np.concatenate([s["bayes_belief"] for s in mid]))
    E, T = a.shape
    n_train = round(E * 0.8)
    te = slice(n_train, None)
    stated = logit(b)
    mu, sd = h[:n_train].mean((0, 1)), h[:n_train].std((0, 1)) + 1e-6
    hs = torch.tensor((h - mu) / sd)
    model = jepa.LatentModel(h.shape[-1]).to(device)
    path = Path(f"{args.data}/latent_memory_layer{li}_mlp.pt")
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
    else:  # explain.py's model, trained here when the collection has none
        model = jepa.train_latent(hs[:n_train], torch.tensor(a[:n_train]), torch.tensor(b[:n_train]), True, 3000,
                                  device, 0)
        torch.save(model.state_dict(), path)
    with torch.no_grad():
        z = model.enc(hs.to(device))
        decoded = model.read(z).cpu().numpy()
        zhat = model.step(z[:, :-1], torch.tensor(a).to(device))
        raw = ((zhat - z[:, 1:]) ** 2).mean(-1).cpu().numpy()  # [E, T]: transition caused by clue t

    # --- internal against stated belief ------------------------------------------------------------------------
    if args.all != "none":
        alls = [np.load(f) for f in sorted(glob.glob(f"{args.all}/shard_*.npz"))]
        assert (np.concatenate([s["actions"] for s in alls]) == a).all(), "memory-all must have the same clues"
        layers = list(alls[0]["memory_layers"])
        memory = np.concatenate([s["memory"] for s in alls])  # float16, read once
    else:
        layers = list(mid[0]["memory_layers"])
        memory = np.concatenate([s["memory"] for s in mid])
    per_layer, internal = {}, None
    for j, layer in enumerate(layers):
        x = memory[:, :, j].astype(np.float32)
        alpha = pick_alpha(x, stated, n_train)
        pred = ridge_probe(x, stated, n_train, alpha)
        gap = pred[te] - stated[te]
        per_layer[int(layer)] = {"alpha": alpha, "rms_gap": float(np.sqrt((gap ** 2).mean())),
                                 "rms_gap_by_turn": np.sqrt((gap ** 2).mean((0, 2))).tolist(),
                                 "r2": jepa.r2(pred[te], stated[te]),
                                 "mse_to_bayes_probe": float(((pred[te] - bayes[te]) ** 2).mean())}
        if layer == LAYER:
            internal = pred
        print(f"layer {layer}: alpha {alpha:g}, R² {per_layer[int(layer)]['r2']:.3f}, rms gap "
              f"{per_layer[int(layer)]['rms_gap']:.3f}", flush=True)
        del x
    del memory
    attr, sign, ctype, contradicts = clue_frame(a[te])
    e, t = np.indices(a[te].shape)
    level = np.array([lvl for lvl, _ in game.TEMPLATES])[templates[te]] if templates is not None else None
    res = {"layer": LAYER, "train_episodes": n_train, "test_episodes": E - n_train, "per_layer": per_layer,
           "mse_stated_to_bayes": float(((stated[te] - bayes[te]) ** 2).mean()), "readers": {}}
    for name, inner in (("probe of S", internal), ("latent decoder", decoded)):
        gap = inner[te] - stated[te]
        on = gap[:, 1:][e, t, attr]  # on the attribute of clue t, after it
        st, inn = stated[te][:, 1:][e, t, attr], inner[te][:, 1:][e, t, attr]
        excess = np.abs(st) - np.abs(inn)  # > 0: the stated belief is more extreme than the internal one
        r = {"rms_gap": float(np.sqrt((gap ** 2).mean())), "r2": jepa.r2(inner[te], stated[te]),
             "rms_gap_by_turn": np.sqrt((gap ** 2).mean((0, 2))).tolist(),
             "mse_internal_to_bayes": float(((inner[te] - bayes[te]) ** 2).mean()),
             "clue_attribute": {"rms_gap": float(np.sqrt((on ** 2).mean())),
                                "by_type": {k: float(np.sqrt((on[ctype == i] ** 2).mean())) for i, k in enumerate(TYPES)},
                                "contradicting": float(np.sqrt((on[contradicts] ** 2).mean())),
                                "not_contradicting": float(np.sqrt((on[~contradicts & (t > 0)] ** 2).mean())),
                                "stated_more_extreme_by": float(excess.mean()),
                                "stated_more_extreme_by_type": {k: float(excess[ctype == i].mean())
                                                                for i, k in enumerate(TYPES)},
                                "stated_more_extreme_contradicting": float(excess[contradicts].mean()),
                                "stated_more_extreme_not_contradicting": float(excess[~contradicts & (t > 0)].mean())}}
        # Where they disagree, which is nearer Bayes?
        bt = bayes[te][:, 1:][e, t, attr]
        big = np.abs(on) > np.quantile(np.abs(on), 0.9)
        r["clue_attribute"]["largest_tenth_gap"] = {
            "stated_abs_error_to_bayes": float(np.abs(st - bt)[big].mean()),
            "internal_abs_error_to_bayes": float(np.abs(inn - bt)[big].mean())}
        if level is not None:  # the wording: the gap by the designed convolutedness level of the clue just read
            r["clue_attribute"]["by_level"] = {str(l): float(np.sqrt((on[level == l] ** 2).mean()))
                                               for l in sorted(set(level.ravel()))}
            r["clue_attribute"]["stated_more_extreme_by_level"] = {str(l): float(excess[level == l].mean())
                                                                   for l in sorted(set(level.ravel()))}
        res["readers"][name] = r

    # --- surprise as a diagnostic ------------------------------------------------------------------------------
    surprise = (raw[te] - raw[te].mean(0)) / raw[te].std(0)
    st_before, st_after = stated[te][:, :-1][e, t, attr], stated[te][:, 1:][e, t, attr]
    shift = sign * (st_after - st_before)
    before = sign * st_before
    usual = np.zeros_like(shift)
    at, sg, ct, _ = clue_frame(a[:n_train])
    ee, tt = np.indices(a[:n_train].shape)
    b0 = sg * stated[:n_train][:, :-1][ee, tt, at]
    s0 = sg * (stated[:n_train][:, 1:][ee, tt, at] - stated[:n_train][:, :-1][ee, tt, at])
    for i in range(len(TYPES)):  # the usual shift for the clue type at that belief, fitted on the training episodes
        coef = np.polyfit(b0[ct == i], s0[ct == i], 1)
        usual[ctype == i] = np.polyval(coef, before[ctype == i])
    deviation = np.abs(shift - usual)
    r_src = np.array([game.SOURCES[s] for s in game.SOURCE_NAMES])[ctype // 2]
    from_bayes = np.abs(shift - np.log(r_src / (1 - r_src)))
    bt_after = bayes[te][:, 1:][e, t, attr]
    misread = deviation > np.quantile(deviation, 0.9)
    wrong = (np.sign(st_after) != np.sign(bt_after)) & (np.abs(bt_after) >= 1)
    p = 1 / (1 + np.exp(-st_after))
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    probe_after = internal[te][:, 1:][e, t, attr]
    detectors = {"surprise": surprise, "answer entropy": entropy, "probe unsureness": -np.abs(probe_after),
                 "internal-stated gap": np.abs(probe_after - st_after)}
    # Final answers, per episode and attribute.
    fin_st, fin_bayes = stated[te][:, -1], bayes[te][:, -1]
    final_wrong = (np.sign(fin_st) != np.sign(fin_bayes)) & (np.abs(fin_bayes) >= 1)
    per_attr = lambda x: np.array([[x[k][attr[k] == j].mean() if (attr[k] == j).any() else np.nan for j in range(4)]
                                   for k in range(len(x))])
    pf = 1 / (1 + np.exp(-fin_st))
    final_detectors = {"surprise": per_attr(surprise), "answer entropy": -(pf * np.log(pf) + (1 - pf) * np.log(1 - pf)),
                       "probe unsureness": -np.abs(internal[te][:, -1]),
                       "internal-stated gap": np.abs(internal[te][:, -1] - fin_st)}
    seen = ~np.isnan(final_detectors["surprise"])
    res["surprise"] = {
        "rates": {"misread": float(misread.mean()), "wrong_after_clue": float(wrong.mean()),
                  "final_wrong": float(final_wrong[seen].mean())},
        "spearman_vs_deviation_from_usual": float(spearman(surprise, deviation)),
        "spearman_vs_deviation_from_bayes": float(spearman(surprise, from_bayes)),
        "auroc": {"misread clue": {k: auroc(misread, v) for k, v in detectors.items()},
                  "answer wrong after the clue": {k: auroc(wrong, v) for k, v in detectors.items()},
                  "final answer wrong": {k: auroc(final_wrong[seen], v[seen]) for k, v in final_detectors.items()}},
        "contradicting_clue": {k: auroc(contradicts[t > 0], v[t > 0]) for k, v in detectors.items()}}
    if level is not None:
        res["surprise"]["convoluted_clue"] = {k: auroc(level >= 2, v) for k, v in detectors.items()}
        res["surprise"]["level3_clue"] = {k: auroc(level == 3, v) for k, v in detectors.items()}
    print(json.dumps(res["surprise"], indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rms_gap_by_turn"} for k, v in res["readers"].items()},
                     indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "surprise_gap.json").write_text(json.dumps(res, indent=1))
    plot(res)


def spearman(x, y):
    from scipy.stats import spearmanr
    return spearmanr(np.ravel(x), np.ravel(y))[0]


def plot(res):
    """The gap by clue and the detectors. The per-layer gap (res["per_layer"]) and the surprise against the distance
    of the LLM's update from its usual one (res["surprise"]["spearman_*"]) are stated in the text."""
    figstyle.apply()
    fig, axes = plt.subplots(1, 2, figsize=(figstyle.DOUBLE * 0.8, 2.8), gridspec_kw={"width_ratios": [1.1, 1.4]})
    ax = axes[0]
    r = res["readers"]["probe of S"]["clue_attribute"]
    groups = TYPES + ["contradicting", "not contradicting"]
    vals = [r["by_type"][k] for k in TYPES] + [r["contradicting"], r["not_contradicting"]]
    colours = [figstyle.GREEN, figstyle.GREEN, figstyle.PURPLE, figstyle.PURPLE, figstyle.GREY, figstyle.GREY]
    hatches = [None, "////", None, "////", None, "////"]
    for i, (v, c, hch) in enumerate(zip(vals, colours, hatches)):
        ax.bar(i, v, 0.7, color=c if hch is None else "white", edgecolor=c, hatch=hch, lw=0.8)
    ax.set_xticks(range(len(groups)), TYPES + ["contradicts", "does not"], fontsize=6, rotation=45, ha="right",
                  rotation_mode="anchor")
    ax.set(ylabel="RMS gap on the clue's attribute")
    ax.set_title("Gap by clue (layer 14)", loc="left")
    ax = axes[1]
    targets = list(res["surprise"]["auroc"])
    # the probe's unsureness is scored in the json but not drawn: it is the entropy of the probed belief, which
    # tracks the entropy of the stated one
    dets = [d for d in res["surprise"]["auroc"][targets[0]] if d != "probe unsureness"]
    colours = [figstyle.BLUE, figstyle.GREY, figstyle.VERMILLION]
    w = 0.8 / len(dets)
    for i, (d, c) in enumerate(zip(dets, colours)):
        ax.bar(np.arange(len(targets)) + (i - (len(dets) - 1) / 2) * w,
               [res["surprise"]["auroc"][tg][d] or np.nan for tg in targets], w * 0.9, color=c, label=d)
    ax.axhline(0.5, color=figstyle.GREY, ls=":", lw=1)
    short = {"misread clue": "misread\nclue", "answer wrong after the clue": "wrong after\nthe clue",
             "final answer wrong": "final\nanswer wrong"}
    ax.set_xticks(range(len(targets)), [short[tg] for tg in targets], fontsize=6)
    ax.set(ylabel="AUROC", ylim=(0.2, 1))
    ax.set_title("Detectors", loc="left")
    figstyle.panel_labels(axes)
    gloss = {"surprise": "latent prediction error", "answer entropy": "entropy of the stated belief",
             "internal-stated gap": "|probed − stated belief|"}
    fig.legend(handles=[Patch(color=c, label=f"{d}: {gloss[d]}") for d, c in zip(dets, colours)], loc="upper left",
               bbox_to_anchor=(0.0, -0.01), ncol=3, fontsize=6, frameon=False, handlelength=1.2, columnspacing=1.5)
    figstyle.save(fig, OUT / "surprise_gap")


if __name__ == "__main__":
    if "--plot" in sys.argv:  # redraw from the saved results
        plot(json.loads((OUT / "surprise_gap.json").read_text()))
    else:
        main()
