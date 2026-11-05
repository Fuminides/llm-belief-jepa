"""What the latent model has learned: clusters of latent states and what their members share.

The latent is trained to predict the belief, so clusters of the raw latent mostly split by belief. The latent with
the belief regressed out (the residual) shows what else it keeps. Each clustering is compared with the known
variables of every state (belief pattern, turn, and the last clue's attribute, source, phrasing, template, level,
negations, and whether it contradicts an earlier clue) by normalised mutual information, and a report lists each
cluster's dominant values and example clue texts.

The model (attention encoder over the turn means, by default) is trained once on the first 80% of episodes and
saved next to the data; analyses use the remaining episodes.
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score

import figstyle
import game
import jepa
from probe_layers import logit

NEGATION = {"not", "false", "denies"}
CLUSTER_COLOURS = [figstyle.ORANGE, figstyle.SKY, figstyle.GREEN, "#F0E442", figstyle.BLUE, figstyle.VERMILLION,
                   figstyle.PURPLE, "#000000"]


def get_model(h, a, b, n_train, path, states, encoder, steps, device):
    model = jepa.LatentModel(h.shape[-1], encoder=encoder).to(device)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device))
        return model.eval()
    model = jepa.train_latent(h[:n_train], a[:n_train], b[:n_train], True, steps, device, 0, encoder)
    torch.save(model.state_dict(), path)
    return model


def metadata(actions, templates, beliefs):
    """Per state after clue t (t = 1..T): known variables, each an integer category per state [E, T]."""
    E, T = actions.shape
    attr, value, negated, source = np.moveaxis(np.array(game.ACTIONS)[actions], -1, 0)
    tpl = np.zeros_like(actions) if templates is None else templates
    level = np.array([lvl for lvl, _ in game.TEMPLATES])[tpl]
    negations = np.array([[sum(w in NEGATION for w in re.findall(r"[a-z]+", game.clue_text(x, t).lower()))
                           for x, t in zip(row_a, row_t)] for row_a, row_t in zip(actions, tpl)])
    contradicts = np.zeros((E, T), int)  # the clue asserts the opposite of an earlier clue on the same attribute
    for t in range(1, T):
        earlier = (attr[:, :t] == attr[:, t:t + 1]) & (value[:, :t] != value[:, t:t + 1])
        contradicts[:, t] = earlier.any(1)
    bins = np.digitize(logit(beliefs[:, 1:]), [-1, 1])  # per attribute: leans value 1, unsure, leans value 0
    pattern = (bins * 3 ** np.arange(4)).sum(-1)
    return {"belief pattern": pattern, "turn": np.tile(np.arange(1, T + 1), (E, 1)), "clue attribute": attr,
            "clue source": source, "clue phrasing": negated.astype(int), "clue template": tpl, "clue level": level,
            "clue negations": negations, "clue contradicts": contradicts}


def readable(name, value):
    if name == "clue attribute":
        return game.ATTR_NAMES[value]
    if name == "clue source":
        return game.SOURCE_NAMES[value]
    if name == "clue phrasing":
        return ("affirmative", "negated")[value]
    if name == "clue template":
        return f"{value}: \"{game.TEMPLATES[value][1][:40]}...\""
    if name == "clue contradicts":
        return ("no", "yes")[value]
    return value


def describe(labels, meta, texts, beliefs, rng, n_examples=3):
    """Per cluster: size, mean belief, the most over-represented value of each variable, and example texts."""
    out = []
    for c in np.unique(labels):
        m = labels == c
        row = {"cluster": int(c), "states": int(m.sum()),
               "mean P(value 0)": dict(zip(game.ATTR_NAMES, beliefs[m].mean(0).astype(float).round(2).tolist()))}
        enriched = {}
        for name, values in meta.items():
            if name == "belief pattern":
                continue
            cats, counts = np.unique(values[m], return_counts=True)
            overall = np.array([(values == k).mean() for k in cats])
            k = np.argmax(counts / m.sum() / overall)
            enriched[name] = {"value": readable(name, int(cats[k])), "share in cluster": round(float(counts[k] / m.sum()), 2),
                              "share overall": round(float(overall[k]), 2)}
        row["most over-represented"] = enriched
        row["examples"] = [texts[i] for i in rng.choice(np.flatnonzero(m), min(n_examples, m.sum()), replace=False)]
        out.append(row)
    return out


def run_clusters(model, h, a, b, tpl, te, args, device):
    with torch.no_grad():
        z = model.enc(h[te].to(device))[:, 1:].cpu().numpy()  # states after clues 1..T
    E, T, D = z.shape
    beliefs = b[te]
    meta = {k: v.ravel() for k, v in metadata(a[te], None if tpl is None else tpl[te], beliefs).items()}
    flat_z = z.reshape(-1, D)
    belief_logits = logit(beliefs[:, 1:]).reshape(-1, 4)
    design = np.column_stack([belief_logits, np.ones(len(belief_logits))])
    residual = flat_z - design @ np.linalg.lstsq(design, flat_z, rcond=None)[0]
    explained = 1 - residual.var(0).sum() / flat_z.var(0).sum()
    print(f"share of latent variance explained linearly by the belief: {explained:.2f}")

    texts = []
    for e in range(E):
        for t in range(T):
            clues = [game.clue_text(a[te][e, s], 0 if tpl is None else tpl[te][e, s]).strip() for s in range(t + 1)]
            texts.append(" | ".join(clues[-2:]))
    rng = np.random.default_rng(0)
    res = {"belief variance share": float(explained), "clusterings": {}}
    labels = {}
    for name, x in (("raw latent", flat_z), ("latent minus belief", residual)):
        x = (x - x.mean(0)) / (x.std(0) + 1e-6)
        labels[name] = KMeans(args.clusters, n_init=10, random_state=0).fit_predict(x)
        nmi = {var: float(normalized_mutual_info_score(values, labels[name])) for var, values in meta.items()}
        res["clusterings"][name] = {"nmi": nmi,
                                    "clusters": describe(labels[name], meta, texts, beliefs[:, 1:].reshape(-1, 4), rng)}
        print(name, "NMI:", {k: round(v, 2) for k, v in nmi.items()})

    figstyle.apply()
    fig, axes = plt.subplots(1, 3, figsize=(figstyle.DOUBLE, 2.7), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    for ax, (name, x) in zip(axes[:2], (("raw latent", flat_z), ("latent minus belief", residual))):
        xy = PCA(2, random_state=0).fit_transform((x - x.mean(0)) / (x.std(0) + 1e-6))
        for c in range(args.clusters):
            m = labels[name] == c
            ax.scatter(xy[m, 0], xy[m, 1], s=3, color=CLUSTER_COLOURS[c % len(CLUSTER_COLOURS)], alpha=0.6, lw=0)
        ax.set(xlabel="PC 1", ylabel="PC 2", xticks=[], yticks=[])
        ax.set_title(name.capitalize(), loc="left")
    ax = axes[2]
    variables = list(meta)
    grid = np.array([[res["clusterings"][n]["nmi"][v] for v in variables] for n in labels])
    im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=max(0.5, grid.max()), aspect="auto")
    ax.set_xticks(range(len(variables)), [v.replace("clue ", "") for v in variables], rotation=90)
    ax.set_yticks(range(len(labels)), ["raw", "minus belief"])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if grid[i, j] < 0.6 * im.get_clim()[1] else "black")
    ax.set_title("What clusters share (NMI)", loc="left")
    figstyle.panel_labels(axes)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"clusters_{args.states}_layer{args.layer}_{args.encoder}"
    Path(f"{stem}.json").write_text(json.dumps(res, indent=1))
    lines = [f"# Clusters of the latent model ({args.states}, layer {args.layer}, {args.encoder} encoder)", "",
             f"Belief explains {explained:.0%} of the latent variance linearly. {args.clusters} k-means clusters on "
             f"{E} test episodes x {T} turns.", ""]
    for name, block in res["clusterings"].items():
        lines += [f"## {name}", "", "NMI with each variable: " +
                  ", ".join(f"{k} {v:.2f}" for k, v in sorted(block["nmi"].items(), key=lambda kv: -kv[1])), ""]
        for c in block["clusters"]:
            top = sorted(c["most over-represented"].items(),
                         key=lambda kv: -kv[1]["share in cluster"] / max(kv[1]["share overall"], 1e-9))[:3]
            lines.append(f"### Cluster {c['cluster']} ({c['states']} states)")
            lines.append(f"- mean P(value 0): {c['mean P(value 0)']}")
            lines.append("- over-represented: " + "; ".join(
                f"{k} = {v['value']} ({v['share in cluster']:.0%} vs {v['share overall']:.0%})" for k, v in top))
            lines += [f"- example: {x}" for x in c["examples"]] + [""]
    Path(f"{stem}.md").write_text("\n".join(lines))
    figstyle.save(fig, stem)


CLUE_TYPES = [f"{s} {'neg.' if n else 'aff.'}" for s in game.SOURCE_NAMES for n in (0, 1)]
TYPE_STYLES = [(figstyle.GREEN, "o", "-", None), (figstyle.GREEN, "o", "--", "white"),
               (figstyle.PURPLE, "s", "-", None), (figstyle.PURPLE, "s", "--", "white")]


def clue_parts(actions):
    attr, value, negated, source = np.moveaxis(np.array(game.ACTIONS)[actions], -1, 0)
    return attr, np.where(value == 0, 1, -1), source * 2 + negated


def binned(x, y, edges, min_count=20):
    idx = np.digitize(x, edges[1:-1])
    keep = [k for k in range(len(edges) - 1) if (idx == k).sum() >= min_count]
    return np.array([x[idx == k].mean() for k in keep]), np.array([y[idx == k].mean() for k in keep])


def run_update_rule(model, h, a, b, tpl, te, args, device):
    """Test 3: the update each clue type would cause at every real state, against Bayes and the LLM's own updates."""
    A = a[te]
    with torch.no_grad():
        z = model.enc(h[te].to(device))[:, :-1]  # state before each clue
        base = model.read(z)
        delta = np.stack([(model.read(model.step(z, torch.full(z.shape[:2], k, device=device))) - base).cpu().numpy()
                          for k in range(game.N_ACTIONS)])  # [action, E, T, attribute]
    base = base.cpu().numpy()
    attr, sign, ctype = clue_parts(np.arange(game.N_ACTIONS))
    # Every action at every state: belief in the asserted value before the clue, and the shift towards it.
    cur_cf = np.stack([sign[k] * base[..., attr[k]] for k in range(game.N_ACTIONS)])
    shift_cf = np.stack([sign[k] * delta[k][..., attr[k]] for k in range(game.N_ACTIONS)])
    # The real transitions.
    lb = logit(b[te])
    e, t = np.indices(A.shape)
    at_attr, at_sign, at_type = attr[A], sign[A], ctype[A]
    cur_llm = at_sign * lb[e, t, at_attr]
    shift_llm = at_sign * (lb[e, t + 1, at_attr] - lb[e, t, at_attr])
    shift_pred = at_sign * delta[A, e, t, at_attr]
    others = np.ones((game.N_ACTIONS, 4), bool)
    others[np.arange(game.N_ACTIONS), attr] = False
    spill_cf = np.mean([np.abs(delta[k][..., others[k]]).mean() for k in range(game.N_ACTIONS)])
    llm_other = np.ones(lb[:, 1:].shape, bool)
    llm_other[e, t, at_attr] = False
    spill_llm = np.abs(lb[:, 1:] - lb[:, :-1])[llm_other].mean()
    bayes = {s: float(np.log(r / (1 - r))) for s, r in game.SOURCES.items()}

    res = {"r2_predicted_vs_llm_shift": jepa.r2(shift_pred, shift_llm),
           "spillover_model": float(spill_cf), "spillover_llm": float(spill_llm), "bayes_shift": bayes, "types": {}}
    for k, name in enumerate(CLUE_TYPES):
        m_cf, m_llm = ctype == k, at_type == k
        slope = lambda x, y: float(np.polyfit(x, y, 1)[0])
        res["types"][name] = {
            "mean_shift_model": float(shift_cf[m_cf].mean()), "mean_shift_llm": float(shift_llm[m_llm].mean()),
            "slope_model": slope(cur_cf[m_cf].ravel(), shift_cf[m_cf].ravel()),
            "slope_llm": slope(cur_llm[m_llm], shift_llm[m_llm])}
    print(json.dumps(res, indent=1))

    figstyle.apply()
    fig, axes = plt.subplots(1, 3, figsize=(figstyle.DOUBLE, 2.7))
    edges = np.quantile(cur_llm, np.linspace(0, 1, 11))
    for ax, (cur, shift, types, title) in zip(axes[:2], (
            (cur_cf, shift_cf, np.broadcast_to(ctype[:, None, None], cur_cf.shape), "Latent model, every clue"),
            (cur_llm, shift_llm, at_type, "LLM, real clues"))):
        for k, (name, (colour, marker, ls, mfc)) in enumerate(zip(CLUE_TYPES, TYPE_STYLES)):
            x, y = binned(cur[types == k], shift[types == k], edges)
            ax.plot(x, y, color=colour, marker=marker, ls=ls, mfc=mfc or colour, label=name)
        for (src, value), ls in zip(bayes.items(), (":", "-.")):
            ax.axhline(value, color=figstyle.GREY, lw=1, ls=ls, label=f"Bayes, {src}")
        ax.axhline(0, color="black", lw=0.5)
        ax.set(xlabel="Belief in the asserted value before (logit)", ylabel="Shift towards it (log-odds)")
        ax.set_title(title, loc="left")
    lo = min(ax.get_ylim()[0] for ax in axes[:2])
    hi = max(ax.get_ylim()[1] for ax in axes[:2])
    for ax in axes[:2]:
        ax.set_ylim(lo, hi)
    ax = axes[2]
    x, y = binned(shift_pred, shift_llm, np.quantile(shift_pred, np.linspace(0, 1, 13)))
    lim = (min(x.min(), y.min()) - 0.1, max(x.max(), y.max()) + 0.1)
    ax.plot(lim, lim, color=figstyle.GREY, lw=1, ls=":")
    ax.plot(x, y, color=figstyle.BLUE, marker="o")
    ax.set(xlim=lim, ylim=lim, xlabel="Shift predicted by the latent model", ylabel="LLM shift (bin mean)")
    ax.set_title(f"Real clues: R\u00b2 {res['r2_predicted_vs_llm_shift']:.2f}", loc="left")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
    figstyle.panel_labels(axes)
    return res, fig


def run_surprise(model, h, a, b, tpl, te, args, device):
    """Test 6: latent prediction error per real transition, by clue phrasing and against unusual updates.

    The raw error grows with the turn, and contradictions can only happen late, so surprise is standardised within
    each turn before comparing groups. "Unusual" is measured against the mean shift of the clue's type (what a
    model driven by clue ids expects), and against Bayes for reference.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    A = a[te]
    with torch.no_grad():
        z = model.enc(h[te].to(device))
        zhat = model.step(z[:, :-1], torch.tensor(A).to(device))
        raw = ((zhat - z[:, 1:]) ** 2).mean(-1).cpu().numpy()  # [E, T], transition caused by clue t
    surprise = (raw - raw.mean(0)) / raw.std(0)
    meta = metadata(A, None if tpl is None else tpl[te], b[te])
    attr, sign, ctype = clue_parts(A)
    lb = logit(b[te])
    e, t = np.indices(A.shape)
    shift_llm = sign * (lb[e, t + 1, attr] - lb[e, t, attr])
    r = np.array([game.SOURCES[s] for s in game.SOURCE_NAMES])[np.array(game.ACTIONS)[A][..., 3]]
    from_bayes = np.abs(shift_llm - np.log(r / (1 - r)))
    deviation = np.abs(shift_llm - np.array([shift_llm[ctype == k].mean() for k in range(len(CLUE_TYPES))])[ctype])
    rho = float(spearmanr(surprise.ravel(), deviation.ravel())[0])
    flags = {"two negation cues": meta["clue negations"] >= 2, "contradicts an earlier clue": meta["clue contradicts"] == 1}
    if tpl is not None:
        flags = {"misread template (9)": meta["clue template"] == 9, **flags}
    flags = {k: v for k, v in flags.items() if 0 < v.mean() < 1}  # plain clues never have two negations
    res = {"spearman_surprise_vs_deviation_from_type_mean": rho,
           "spearman_surprise_vs_deviation_from_bayes": float(spearmanr(surprise.ravel(), from_bayes.ravel())[0]),
           "auc": {k: float(roc_auc_score(v.ravel(), surprise.ravel())) for k, v in flags.items()},
           "raw_error_by_turn": [float(raw[:, k].mean()) for k in range(A.shape[1])]}
    groups = (meta["clue template"] * 2 + meta["clue phrasing"]) if tpl is not None else ctype
    res["by_group"] = {int(g): float(surprise[groups == g].mean()) for g in np.unique(groups)}
    print(json.dumps(res, indent=1))

    figstyle.apply()
    fig, axes = plt.subplots(1, 3, figsize=(figstyle.DOUBLE, 2.7), gridspec_kw={"width_ratios": [1.4, 1, 0.9]})
    ax = axes[0]
    if tpl is not None:
        level_colours = plt.cm.viridis(np.linspace(0.1, 0.85, len(game.LEVELS)))
        for i, (level, _) in enumerate(game.TEMPLATES):
            for n in (0, 1):
                vals = surprise[(meta["clue template"] == i) & (meta["clue phrasing"] == n)]
                ax.bar(i + (n - 0.5) * 0.4, vals.mean(), 0.38, yerr=1.96 * vals.std() / np.sqrt(len(vals)),
                       color=level_colours[level] if n == 0 else "white", edgecolor=level_colours[level], lw=1,
                       error_kw={"lw": 0.6, "capsize": 1.5})
        ax.set_xticks(range(len(game.TEMPLATES)))
        ax.axhline(0, color="black", lw=0.5)
        ax.set(xlabel="Template (filled: affirmative, hollow: negated)", ylabel="Surprise (z within turn)")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=f"level {i}") for i, c in enumerate(level_colours)]
        ax.legend(handles=handles, loc="upper left", ncol=2, fontsize=6)
        ax.set_title("Surprise by template (95% CI)", loc="left")
    else:
        for k, (name, (colour, *_)) in enumerate(zip(CLUE_TYPES, TYPE_STYLES)):
            ax.bar(k, surprise[ctype == k].mean(), 0.7, color=colour)
        ax.set_xticks(range(len(CLUE_TYPES)), [c.replace(" ", "\n") for c in CLUE_TYPES])
        ax.set(ylabel="Surprise (z within turn)")
        ax.set_title("Surprise by clue type", loc="left")
    ax = axes[1]
    x, y = binned(deviation.ravel(), surprise.ravel(), np.quantile(deviation, np.linspace(0, 1, 11)))
    ax.plot(x, y, color=figstyle.BLUE, marker="o")
    ax.set(xlabel="|LLM shift − usual shift of its type|", ylabel="Surprise (z within turn)")
    ax.set_title(f"Unusual updates (\u03c1 {rho:.2f})", loc="left")
    ax = axes[2]
    names = list(res["auc"])
    ax.barh(range(len(names)), [res["auc"][n] for n in names], color=figstyle.BLUE, height=0.6)
    ax.axvline(0.5, color=figstyle.GREY, lw=1, ls=":")
    ax.set_yticks(range(len(names)), [n.replace(" (9)", "\n(9)").replace(" an earlier", "\nan earlier") for n in names])
    ax.set(xlim=(0.4, 1), xlabel="AUC of surprise")
    ax.set_title("Flags", loc="left")
    figstyle.panel_labels(axes)
    return res, fig


def dci(train_x, train_y, test_x, test_y):
    """Eastwood & Williams (2018): Lasso importance of each dimension for each attribute, then disentanglement
    (each dimension serves one attribute), completeness (each attribute uses few dimensions) and
    informativeness (test R² per attribute)."""
    from sklearn.linear_model import LassoCV
    mu, sd = train_x.mean(0), train_x.std(0) + 1e-6
    tx, vx = (train_x - mu) / sd, (test_x - mu) / sd
    importance, info = [], []
    for k in range(train_y.shape[1]):
        lasso = LassoCV(cv=5, random_state=0, max_iter=5000).fit(tx, train_y[:, k])
        importance.append(np.abs(lasso.coef_))
        info.append(jepa.r2(lasso.predict(vx), test_y[:, k]))
    R = np.array(importance).T + 1e-12  # [dims, attributes]
    K, D = R.shape[1], R.shape[0]
    p_dim = R / R.sum(1, keepdims=True)
    disent = 1 + (p_dim * np.log(p_dim)).sum(1) / np.log(K)
    p_attr = R / R.sum(0, keepdims=True)
    complete = 1 + (p_attr * np.log(p_attr)).sum(0) / np.log(D)
    weight = R.sum(1) / R.sum()
    return {"disentanglement": float((weight * disent).sum()), "completeness": float(complete.mean()),
            "informativeness": float(np.mean(info)), "importance": R}


def run_factors(model, h, a, b, tpl, te, args, device):
    """Test 5: does the latent give each attribute its own directions? DCI on the latent's axes and, since the
    latent need not align factors with its axes, on 16 ICA components; for reference, the same on as many principal
    components of the input activations. Also ICA components against each attribute and t-SNE coloured by belief."""
    from sklearn.decomposition import FastICA
    from sklearn.manifold import TSNE
    tr = slice(0, te.start)
    with torch.no_grad():
        z_tr = model.enc(h[tr].to(device))[:, 1:].reshape(-1, model.read.in_features).cpu().numpy()
        z_te = model.enc(h[te].to(device))[:, 1:].reshape(-1, model.read.in_features).cpu().numpy()
    y_tr, y_te = logit(b[tr, 1:]).reshape(-1, 4), logit(b[te, 1:]).reshape(-1, 4)
    # Reference: principal components of the activations the encoder reads (turns pooled so far, as the encoder
    # can), with as many components as the latent has dimensions.
    x = h.numpy() if args.states != "pool" else (np.cumsum(h.numpy(), 1) / np.arange(1, h.shape[1] + 1)[None, :, None])
    x_tr, x_te = x[tr, 1:].reshape(-1, x.shape[-1]), x[te, 1:].reshape(-1, x.shape[-1])
    pca = PCA(z_tr.shape[1], random_state=0).fit(x_tr)
    p_tr, p_te = pca.transform(x_tr), pca.transform(x_te)
    res = {"latent": dci(z_tr, y_tr, z_te, y_te), "PCA of activations": dci(p_tr, y_tr, p_te, y_te)}
    for name, (a_tr, a_te) in (("latent", (z_tr, z_te)), ("PCA of activations", (p_tr, p_te))):
        rot = FastICA(16, random_state=0, max_iter=2000).fit(a_tr)
        res[f"{name}, ICA"] = dci(rot.transform(a_tr), y_tr, rot.transform(a_te), y_te)
    ica = FastICA(8, random_state=0, max_iter=2000).fit(z_tr)
    s_te = ica.transform(z_te)
    corr = np.abs(np.corrcoef(np.column_stack([s_te, y_te]).T)[:8, 8:])  # [component, attribute]
    order = np.argsort(corr.argmax(1) * 10 - corr.max(1))  # group components by their best attribute
    tsne = TSNE(2, random_state=0, init="pca", perplexity=40).fit_transform(z_te)
    out = {k: {m: v[m] for m in ("disentanglement", "completeness", "informativeness")} for k, v in res.items()}
    out["ica_abs_correlation"] = corr.tolist()
    print(json.dumps(out, indent=1))

    figstyle.apply()
    fig = plt.figure(figsize=(figstyle.DOUBLE, 5.2))
    top_axes = fig.subfigures(2, 1, height_ratios=[1, 0.9])
    ax_imp, ax_dci, ax_ica = top_axes[0].subplots(1, 3, gridspec_kw={"width_ratios": [1, 1, 0.9]})
    R = res["latent"]["importance"]
    keep = np.argsort(-R.sum(1))[:24]
    keep = keep[np.argsort(R[keep].argmax(1), kind="stable")]
    shown = R[keep] / R[keep].max(1, keepdims=True)
    ax_imp.imshow(shown.T, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax_imp.set_yticks(range(4), game.ATTR_NAMES)
    ax_imp.set_xticks([])
    ax_imp.set(xlabel="24 most used latent dimensions")
    ax_imp.set_title("Lasso importance (row-scaled)", loc="left")
    metrics = ("disentanglement", "completeness", "informativeness")
    xs = np.arange(len(metrics))
    variants = (("latent", figstyle.BLUE, figstyle.BLUE, None), ("latent, ICA", figstyle.SKY, figstyle.SKY, None),
                ("PCA of activations", "white", figstyle.BLUE, "////"),
                ("PCA of activations, ICA", "white", figstyle.SKY, "\\\\\\\\"))
    for i, (name, colour, edge, hatch) in enumerate(variants):
        ax_dci.bar(xs + (i - 1.5) * 0.2, [res[name][m] for m in metrics], 0.19, color=colour, edgecolor=edge,
                   hatch=hatch, lw=0.8, label=name)
    ax_dci.set_xticks(xs, ["disent.", "complet.", "inform."])
    ax_dci.set(ylim=(0, 1))
    ax_dci.set_title("DCI", loc="left")
    ax_dci.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=6)
    ax_ica.imshow(corr[order].T, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    for i, c in enumerate(order):
        for k in range(4):
            ax_ica.text(i, k, f"{corr[c, k]:.1f}", ha="center", va="center", fontsize=6,
                        color="white" if corr[c, k] < 0.6 else "black")
    ax_ica.set_yticks(range(4), game.ATTR_NAMES)
    ax_ica.set_xticks(range(8), [f"IC{i + 1}" for i in range(8)], rotation=90)
    ax_ica.set_title("ICA |correlation|", loc="left")
    tsne_axes = top_axes[1].subplots(1, 4)
    p_te = b[te, 1:].reshape(-1, 4)
    for k, ax in enumerate(tsne_axes):
        sc = ax.scatter(tsne[:, 0], tsne[:, 1], c=p_te[:, k], cmap="RdBu_r", vmin=0, vmax=1, s=2, lw=0)
        ax.set(xticks=[], yticks=[])
        ax.set_title(f"t-SNE: {game.ATTR_NAMES[k]}", loc="left")
    top_axes[1].colorbar(sc, ax=tsne_axes, shrink=0.8, label="P(value 0)")
    figstyle.panel_labels([ax_imp, ax_dci, ax_ica, *tsne_axes])
    return out, fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--states", choices=["acts", "pool", "running", "memory"], default="pool")
    p.add_argument("--encoder", choices=["mlp", "attention"], default="attention")
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--clusters", type=int, default=8)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--out", required=True)
    p.add_argument("--analyses", nargs="+", choices=["clusters", "update", "surprise", "factors"],
                   default=["clusters", "update", "surprise", "factors"])
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    h, a, b, tpl = jepa.load(args.data, args.states, args.layer)
    n_train = round(len(h) * 0.8)
    mu, sd = h[:n_train].mean((0, 1)), h[:n_train].std((0, 1)) + 1e-6
    h = torch.tensor((h - mu) / sd)
    path = Path(args.data) / f"latent_{args.states}_layer{args.layer}_{args.encoder}.pt"
    model = get_model(h, torch.tensor(a), torch.tensor(b), n_train, path, args.states, args.encoder, args.steps,
                      device)

    te = slice(n_train, None)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.states}_layer{args.layer}_{args.encoder}"
    if "clusters" in args.analyses:
        run_clusters(model, h, a, b, tpl, te, args, device)
    for name, fn in (("update", run_update_rule), ("surprise", run_surprise), ("factors", run_factors)):
        if name in args.analyses:
            res, fig = fn(model, h, a, b, tpl, te, args, device)
            stem = out / f"{name}_{suffix}"
            Path(f"{stem}.json").write_text(json.dumps(res, indent=1))
            figstyle.save(fig, stem)


if __name__ == "__main__":
    main()
