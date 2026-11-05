"""What the free numbers u of the anchored JEPA hold, drawn (candidates for the working doc, next to the probes of
anchored.py). z = [b, u]: b the LLM's four belief log-odds, u the k free numbers. On held-out episodes:

  1. map:          a t-SNE map of u, the same points coloured by the known variables of the state (turn,
                   contradiction, the last clue's source and negation, recency on one attribute) and by the Bayesian
                   log-odds, which the probes say u does not hold.
  2. pca:          the same on the first two principal components of u, with the standard deviation of each free
                   number (how many of the k are alive under the L2 penalty).
  3. clue_shift:   which clues move u, and along what: the mean change of u after each of the 32 clues, projected on
                   the probe direction of each known variable (left) and on the raw free numbers (right).
  4. wording:      on paraphrased clues, the change of u after each of the ten templates, projected on the
                   directions of the wording variables, and its size by convolutedness level.

The free numbers have no fixed rotation (the L2 penalty is rotation-invariant), so the raw numbers are shown once,
for how many live, and the attributions use probe directions, which are the same up to seed.

  python -m latent.free_numbers [--data data/qwen3.5-2b-memory-mid] [--paraphrase data/qwen3.5-2b-paraphrase-memory-mid]
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.manifold import TSNE

from figures import figstyle
from games import game
from latent import jepa
from latent.anchored import AnchoredModel, history_variables
from latent.anchored_paraphrase import wording_variables

ATTRS = game.ATTR_NAMES


def encode(data_dir, layer, k, seed):
    """z = [b, u] of every state [E, T+1, 4+k], the actions, the beliefs, the templates (or None) and the train/test
    split."""
    memory_layers = list(np.load(f"{data_dir}/shard_000000.npz")["memory_layers"])
    h, a, b, tpl = jepa.load(data_dir, "memory", memory_layers.index(layer))
    n = round(len(h) * 0.8)
    mu, sd = h[:n].mean((0, 1)), h[:n].std((0, 1)) + 1e-6
    model = AnchoredModel(h.shape[-1], k)
    model.load_state_dict(torch.load(f"{data_dir}/anchored_layer{layer}_k{k}_seed{seed}.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        z = np.concatenate([model.enc(torch.tensor((h[i:i + 200] - mu) / sd)).numpy() for i in range(0, len(h), 200)])
    return z, a, b, tpl, n


def flat(x):
    return x.reshape(-1, x.shape[-1])


def directions(u, known, n):
    """A unit direction in u per scalar variable (ridge on the training states) with its held-out R²."""
    out = {}
    for name, y in known.items():
        for j in range(y.shape[-1]):
            label = name if y.shape[-1] == 1 else f"{name}: {ATTRS[j] if y.shape[-1] == 4 else j}"
            ridge = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(flat(u[:n]), flat(y[:n])[:, j])
            r2 = jepa.r2(ridge.predict(flat(u[n:])), flat(y[n:])[:, j])
            w = ridge.coef_ / (np.linalg.norm(ridge.coef_) + 1e-12)
            out[label] = (w, float(r2))
    return out


def clue_label(action_id):
    attr, value, negated, source = game.ACTIONS[action_id]
    values = game.ATTRIBUTES[ATTRS[attr]]
    claim = f"is not {values[1 - value]}" if negated else f"is {values[value]}"
    return f"{game.SOURCE_NAMES[source]}: {claim}"


def scatter_maps(xy, colourings, title, stem):
    """xy [N, 2]; colourings: (label, values, kind) with kind 'cat' (few values, Okabe-Ito) or 'seq' / 'div'."""
    figstyle.apply()
    fig, axes = plt.subplots(2, 3, figsize=(figstyle.DOUBLE, 4.6))
    palette = [figstyle.BLUE, figstyle.ORANGE, figstyle.GREEN, figstyle.PURPLE, figstyle.VERMILLION, figstyle.SKY]
    for ax, (label, v, kind) in zip(axes.ravel(), colourings):
        order = np.random.default_rng(0).permutation(len(xy))  # no colour drawn systematically on top
        if kind == "cat":
            values = np.unique(v)
            for value, colour in zip(values, palette):
                m = v[order] == value
                ax.scatter(xy[order][m, 0], xy[order][m, 1], s=2, c=colour, lw=0, alpha=0.6,
                           label=str(int(value)) if float(value).is_integer() else str(value))
            ax.legend(fontsize=6, markerscale=3, handletextpad=0.2, loc="best")
        else:
            cmap, lim = ("coolwarm", np.abs(v).max()) if kind == "div" else ("viridis", None)
            sc = ax.scatter(xy[order, 0], xy[order, 1], s=2, c=v[order], lw=0, alpha=0.7, cmap=cmap,
                            vmin=-lim if lim else None, vmax=lim)
            fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        ax.set_title(label, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ("left", "bottom"):
            ax.spines[s].set_visible(False)
    figstyle.panel_labels(axes.ravel())
    fig.suptitle(title, fontsize=8, y=1.0)
    figstyle.save(fig, stem)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/qwen3.5-2b-memory-mid")
    p.add_argument("--paraphrase", default="data/qwen3.5-2b-paraphrase-memory-mid")
    p.add_argument("--layer", type=int, default=14)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/qwen3.5-2b-memory/free_numbers_candidates")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"data": args.data, "layer": args.layer, "k": args.k, "seed": args.seed}

    z, a, _, _, n = encode(args.data, args.layer, args.k, args.seed)
    E, T1, _ = z.shape
    u = z[..., 4:]
    known = history_variables(a)
    u_te, known_te = u[n:], {k: v[n:] for k, v in known.items()}
    sd = flat(u_te).std(0)
    live = np.argsort(-sd)
    res["u_std"] = sd.round(4).tolist()
    print(f"{E} episodes, {E - n} held out; u std (sorted): {np.round(sd[live], 3).tolist()}", flush=True)

    # 1. the t-SNE map (identical states, all of turn 0, embedded once)
    pts = flat(u_te)
    unique, inverse = np.unique(pts.round(5), axis=0, return_inverse=True)
    tsne = TSNE(n_components=2, perplexity=50, init="pca", random_state=0).fit_transform(unique)[inverse.ravel()]
    col = ATTRS.index("colour")
    colourings = [
        ("turn", flat(known_te["turn"])[:, 0], "seq"),
        ("attributes with both values claimed", flat(known_te["contradicted attributes"])[:, 0], "seq"),
        ("last clue from Bob (1) or Alice (0)", flat(known_te["clue just read is from Bob"])[:, 0], "cat"),
        ("last clue negated", flat(known_te["clue just read is negated"])[:, 0], "cat"),
        ("clues since the last one on colour", flat(known_te["clues since the last on each attribute"])[:, col], "seq"),
        ("Bayesian log-odds of colour", flat(known_te["Bayesian log-odds"])[:, col], "div"),
    ]
    scatter_maps(tsne, colourings, f"t-SNE of the {args.k} free numbers, layer {args.layer}, held-out states",
                 out / "map_tsne")
    print("[1/4] map", flush=True)

    # 2. the principal components, and the life of each free number
    mean = pts.mean(0)
    _, s, vt = np.linalg.svd(pts - mean, full_matrices=False)
    pca = (pts - mean) @ vt[:2].T
    res["pca_variance"] = (s[:2] ** 2 / (s ** 2).sum()).round(3).tolist()
    scatter_maps(pca, colourings, f"First two principal components of the free numbers "
                                  f"({100 * sum(res['pca_variance']):.0f}% of their variance)", out / "map_pca")
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    ax.bar(np.arange(args.k), sd[live], color=figstyle.BLUE)
    ax.set_xlabel("free number, by standard deviation")
    ax.set_ylabel("std on held-out states")
    ax.set_xticks([])
    figstyle.save(fig, out / "u_std")
    print("[2/4] pca", flush=True)

    # 3. which clues move u, along which directions
    scalar = {k: v for k, v in known.items() if k != "clue just read (signed, per attribute)"}
    dirs = directions(u, scalar, n)
    res["directions_r2"] = {k: v[1] for k, v in dirs.items()}
    du = (u_te[:, 1:] - u_te[:, :-1]).reshape(-1, args.k)  # the change of u after each clue
    clue = a[n:].reshape(-1)
    order = sorted(range(game.N_ACTIONS), key=lambda i: (game.ACTIONS[i][3], game.ACTIONS[i][2], game.ACTIONS[i][0],
                                                          game.ACTIONS[i][1]))
    labels = [clue_label(i) for i in order]
    names = list(dirs)
    proj = du @ np.stack([dirs[name][0] for name in names]).T  # [N, directions]
    proj = proj / (proj.std(0) + 1e-12)
    by_dir = np.stack([proj[clue == i].mean(0) for i in order])  # [32, directions]
    raw = du[:, live] / (du[:, live].std(0) + 1e-12)
    by_dim = np.stack([raw[clue == i].mean(0) for i in order])
    res["clue_shift"] = {"clues": labels, "directions": names, "by_direction": by_dir.round(3).tolist(),
                         "by_free_number": by_dim.round(3).tolist()}
    fig, axes = plt.subplots(1, 2, figsize=(figstyle.DOUBLE, 6.2), gridspec_kw={"width_ratios": [len(names), args.k]})
    for ax, m, cols in ((axes[0], by_dir, [f"{nm}\n(R² {dirs[nm][1]:.2f})" for nm in names]),
                        (axes[1], by_dim, [f"u{j}\n(std {sd[j]:.2f})" for j in live])):
        lim = np.abs(m).max()
        im = ax.imshow(m, cmap="coolwarm", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=90, fontsize=5.5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels if ax is axes[0] else [], fontsize=6)
        for s_ in ("top", "right", "left", "bottom"):
            ax.spines[s_].set_visible(True)
            ax.spines[s_].set_linewidth(0.5)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="mean change after the clue (in std)")
    axes[0].set_title("along the probe direction of each variable", fontsize=8)
    axes[1].set_title("along each free number", fontsize=8)
    figstyle.panel_labels(axes)
    figstyle.save(fig, out / "clue_shift")
    print("[3/4] clue_shift", flush=True)

    # 4. wording: the paraphrased collection and its own anchored model
    zp, ap, _, tpl, npar = encode(args.paraphrase, args.layer, args.k, args.seed)
    up = zp[..., 4:]
    kp = {**wording_variables(ap, tpl), **history_variables(ap)}
    wording = {k: v for k, v in kp.items() if k in ("its convolutedness level", "its negation cues", "turn",
                                                     "contradicted attributes", "clue just read is negated",
                                                     "clue just read is from Bob")}
    dirs_p = directions(up, wording, npar)
    res["wording_directions_r2"] = {k: v[1] for k, v in dirs_p.items()}
    dup = (up[npar:, 1:] - up[npar:, :-1]).reshape(-1, args.k)
    t_te = tpl[npar:].reshape(-1)
    names_p = list(dirs_p)
    proj = dup @ np.stack([dirs_p[nm][0] for nm in names_p]).T
    proj = proj / (proj.std(0) + 1e-12)
    by_tpl = np.stack([proj[t_te == j].mean(0) for j in range(len(game.TEMPLATES))])
    size = np.linalg.norm(dup, axis=1)
    levels = np.array([lvl for lvl, _ in game.TEMPLATES])[t_te]
    res["wording"] = {"templates": [t for _, t in game.TEMPLATES], "directions": names_p,
                      "by_template": by_tpl.round(3).tolist(),
                      "size_by_level": {int(l): float(size[levels == l].mean()) for l in np.unique(levels)}}
    fig, axes = plt.subplots(1, 2, figsize=(figstyle.DOUBLE, 3.4), gridspec_kw={"width_ratios": [3, 1.2]})
    lim = np.abs(by_tpl).max()
    im = axes[0].imshow(by_tpl, cmap="coolwarm", vmin=-lim, vmax=lim, aspect="auto")
    axes[0].set_xticks(range(len(names_p)))
    axes[0].set_xticklabels([f"{nm}\n(R² {dirs_p[nm][1]:.2f})" for nm in names_p], rotation=90, fontsize=5.5)
    axes[0].set_yticks(range(len(game.TEMPLATES)))
    axes[0].set_yticklabels([f"L{lvl}: {t.format(source='Alice', subject='the object', claim='is red', anticlaim='is not red')}"
                             for lvl, t in game.TEMPLATES], fontsize=6)
    fig.colorbar(im, ax=axes[0], fraction=0.04, pad=0.02, label="mean change after the clue (in std)")
    axes[0].set_title("the change of u after each template, along the probe directions", fontsize=8)
    for lvl in np.unique(levels):
        axes[1].bar(lvl, size[levels == lvl].mean(), color=figstyle.BLUE,
                    yerr=size[levels == lvl].std() / np.sqrt((levels == lvl).sum()), capsize=2)
    axes[1].set_xlabel("convolutedness level of the clue")
    axes[1].set_ylabel("|Δu| after the clue")
    axes[1].set_xticks(range(4))
    figstyle.panel_labels(axes)
    figstyle.save(fig, out / "wording")
    print("[4/4] wording", flush=True)
    (out / "free_numbers.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
