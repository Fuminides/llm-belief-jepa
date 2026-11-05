"""Clusters of the free numbers u of the anchored JEPA, and what their states have in common: the known variables
first (explain.py's recipe: mutual information of the clustering with each variable, the over-represented value of
each per cluster), then, for what the variables do not settle, another LLM reads the clue histories.

Stage `cluster` (any python with sklearn): k-means on u of the held-out states after clues 1..T, a t-SNE map
coloured by cluster, the NMI table, and per cluster a report with example histories and the two samples for the
reader: histories inside the cluster and histories from other clusters with the same number of clues (so the turn,
the obvious variable, cannot be the answer). Stage `describe` (the vllm environment): a chat model reads each
cluster's two samples and says in a few sentences what the first group shares that the second lacks; the answers
go into the report.

  python free_clusters.py cluster  [--clusters 8]                  # results/.../free_numbers_candidates/clusters.*
  python free_clusters.py describe [--model Qwen/Qwen3.5-9B]       # adds the descriptions to clusters.{json,md}
"""
import argparse
import json
from pathlib import Path

import numpy as np

import game

OUT = Path("results/qwen3.5-2b-memory/free_numbers_candidates")
SAMPLE = 15  # histories per group shown to the reader


def history(actions, tpl, e, t):
    """The clues of episode e up to and including clue t (1-based), as one line."""
    return " | ".join(game.clue_text(actions[e, s], 0 if tpl is None else tpl[e, s]).strip() for s in range(t))


def cluster(args):
    import matplotlib.pyplot as plt
    import torch
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    from sklearn.metrics import normalized_mutual_info_score

    import figstyle
    from anchored import history_variables
    from explain import CLUSTER_COLOURS, metadata, readable
    from free_numbers import encode

    torch.manual_seed(0)
    z, a, b, tpl, n = encode(args.data, args.layer, args.k, args.seed)
    u = z[n:, 1:, 4:]  # held-out states after clues 1..T
    a_te, b_te, tpl_te = a[n:], b[n:], None if tpl is None else tpl[n:]
    E, T, _ = u.shape
    flat = u.reshape(-1, args.k)
    meta = {k: v.ravel() for k, v in metadata(a_te, tpl_te, b_te).items()}
    meta["contradicted attributes"] = history_variables(a_te)["contradicted attributes"][:, 1:].reshape(-1).astype(int)
    meta = {k: v for k, v in meta.items() if len(np.unique(v)) > 1}
    x = (flat - flat.mean(0)) / (flat.std(0) + 1e-6)
    labels = KMeans(args.clusters, n_init=10, random_state=0).fit_predict(x)
    nmi = {var: float(normalized_mutual_info_score(values, labels)) for var, values in meta.items()}
    print("NMI:", {k: round(v, 2) for k, v in sorted(nmi.items(), key=lambda kv: -kv[1])}, flush=True)

    turn = meta["turn"]
    rng = np.random.default_rng(0)
    texts = [history(a_te, tpl_te, i // T, i % T + 1) for i in range(E * T)]
    clusters = []
    for c in range(args.clusters):
        m = labels == c
        inside = np.flatnonzero(m)
        enriched = {}
        for name, values in meta.items():
            cats, counts = np.unique(values[m], return_counts=True)
            overall = np.array([(values == k).mean() for k in cats])
            k = np.argmax(counts / m.sum() / overall)
            enriched[name] = {"value": readable(name, int(cats[k])) if name != "contradicted attributes" else int(cats[k]),
                              "share in cluster": round(float(counts[k] / m.sum()), 2),
                              "share overall": round(float(overall[k]), 2)}
        picked = rng.choice(inside, min(SAMPLE, len(inside)), replace=False)
        # the contrast: states of other clusters with the same number of clues as each picked state
        contrast = []
        for i in picked:
            pool = np.flatnonzero(~m & (turn == turn[i]))
            contrast.append(int(rng.choice(pool)))
        clusters.append({"cluster": c, "states": int(m.sum()),
                         "turns": {int(t): int((turn[m] == t).sum()) for t in np.unique(turn[m])},
                         "mean P(value 0)": dict(zip(game.ATTR_NAMES, b_te[:, 1:].reshape(-1, 4)[m].mean(0).round(2).tolist())),
                         "most over-represented": enriched,
                         "inside": [texts[i] for i in picked], "contrast": [texts[i] for i in contrast]})

    # the map: t-SNE of u coloured by cluster (identical states embedded once)
    unique, inverse = np.unique(flat.round(5), axis=0, return_inverse=True)
    tsne = TSNE(n_components=2, perplexity=50, init="pca", random_state=0).fit_transform(unique)[inverse.ravel()]
    figstyle.apply()
    fig, axes = plt.subplots(1, 2, figsize=(figstyle.DOUBLE, 3.0), gridspec_kw={"width_ratios": [1.1, 1]})
    order = rng.permutation(len(tsne))
    for c in range(args.clusters):
        m = labels[order] == c
        axes[0].scatter(tsne[order][m, 0], tsne[order][m, 1], s=2.5, color=CLUSTER_COLOURS[c % len(CLUSTER_COLOURS)],
                        lw=0, alpha=0.7, label=f"{c} ({(labels == c).sum()})")
    axes[0].legend(fontsize=6, markerscale=3, ncol=2, title="cluster (states)", title_fontsize=6, loc="best")
    axes[0].set(xticks=[], yticks=[])
    for s in ("left", "bottom"):
        axes[0].spines[s].set_visible(False)
    axes[0].set_title(f"t-SNE of the {args.k} free numbers, {args.clusters} k-means clusters", fontsize=8, loc="left")
    names = sorted(nmi, key=lambda v: -nmi[v])
    axes[1].barh(range(len(names)), [nmi[v] for v in names], color=figstyle.BLUE)
    axes[1].set_yticks(range(len(names)), [v.replace("clue ", "last clue: ") for v in names], fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("normalised mutual information with the clustering")
    axes[1].set_title("Relevant cluster variables", fontsize=8, loc="left")
    figstyle.panel_labels(axes)
    OUT.mkdir(parents=True, exist_ok=True)
    figstyle.save(fig, OUT / "clusters")
    res = {"data": args.data, "layer": args.layer, "k": args.k, "seed": args.seed, "clusters_k": args.clusters,
           "held_out_states": int(E * T), "nmi": nmi, "clusters": clusters}
    if (OUT / "clusters.json").exists():  # the clustering is deterministic: keep an earlier stage's descriptions
        old = json.loads((OUT / "clusters.json").read_text())
        for c, o in zip(clusters, old.get("clusters", [])):
            if "description" in o and o["states"] == c["states"] and o["inside"] == c["inside"]:
                c["description"] = o["description"]
    (OUT / "clusters.json").write_text(json.dumps(res, indent=1))
    report(res)


def describe(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    res = json.loads((OUT / "clusters.json").read_text())
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(args.model, dtype="bfloat16", gpu_memory_utilization=args.memory, max_model_len=8192, seed=0)
    intro = ("In a game, a secret object has four properties (colour red or blue, shape round or square, size big or "
             "small, material wooden or metal) and two sources, Alice (right 90% of the time) and Bob (right 60%), "
             "give clues in turn. Below are two groups of clue histories, each history the clues read so far, in "
             "order. A model of a language model's memory grouped the histories of group A together; group B are "
             "histories from other groups with the same number of clues. In at most three sentences, say what the "
             "histories of group A have in common that those of group B do not: the kind of clues, their order, "
             "their sources, repetitions, contradictions, or anything else. If you find nothing clear, say so.")
    prompts = []
    for c in res["clusters"]:
        group_a = "\n".join(f"A{j + 1}. {h}" for j, h in enumerate(c["inside"]))
        group_b = "\n".join(f"B{j + 1}. {h}" for j, h in enumerate(c["contrast"]))
        messages = [{"role": "user", "content": f"{intro}\n\nGroup A:\n{group_a}\n\nGroup B:\n{group_b}"}]
        prompts.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False))
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=300), use_tqdm=False)
    for j, (c, o) in enumerate(zip(res["clusters"], outs), start=1):
        c["description"] = {"model": args.model, "text": o.outputs[0].text.strip()}
        print(f"[{j}/{len(outs)}] cluster {c['cluster']}: {c['description']['text'][:120]}", flush=True)
    (OUT / "clusters.json").write_text(json.dumps(res, indent=1))
    report(res)


def report(res):
    lines = [f"# Clusters of the free numbers (anchored JEPA, layer {res['layer']}, k = {res['k']}, seed {res['seed']})",
             "", f"{res['clusters_k']} k-means clusters on {res['held_out_states']} held-out states (after clues 1 to 8).",
             "", "NMI with each variable: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(res["nmi"].items(),
                                                                                             key=lambda kv: -kv[1])), ""]
    for c in res["clusters"]:
        top = sorted(c["most over-represented"].items(),
                     key=lambda kv: -kv[1]["share in cluster"] / max(kv[1]["share overall"], 1e-9))[:4]
        lines.append(f"## Cluster {c['cluster']} ({c['states']} states; turns {c['turns']})")
        lines.append(f"- mean P(value 0): {c['mean P(value 0)']}")
        lines.append("- over-represented: " + "; ".join(
            f"{k} = {v['value']} ({v['share in cluster']:.0%} vs {v['share overall']:.0%})" for k, v in top))
        if "description" in c:
            lines.append(f"- **{c['description']['model']} says:** {c['description']['text']}")
        lines += [f"- inside: {x}" for x in c["inside"][:5]]
        lines += [f"- contrast (same number of clues, other clusters): {x}" for x in c["contrast"][:3]] + [""]
    (OUT / "clusters.md").write_text("\n".join(lines))
    print("wrote", OUT / "clusters.md", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["cluster", "describe"])
    p.add_argument("--data", default="data/qwen3.5-2b-memory-mid")
    p.add_argument("--layer", type=int, default=14)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clusters", type=int, default=8)
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--memory", type=float, default=0.3, help="share of GPU memory vLLM may take")
    args = p.parse_args()
    (cluster if args.stage == "cluster" else describe)(args)


if __name__ == "__main__":
    main()
