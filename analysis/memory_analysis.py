"""Is the belief in the recurrent memory of Qwen3.5's linear-attention layers?

For each saved memory layer (decoder layer L, a Gated DeltaNet layer), linear probes of the LLM's belief and of
the clue history from: its recurrent state S (projected), the residual stream after the same layer at the current
turn (turn mean), and the running mean of those turn means. S summarises the whole context by construction; the
question is whether it holds the belief as well as, or better than, pooling the residual stream over turns.

  python -m analysis.memory_analysis --data data/qwen3.5-2b-memory --out results/qwen3.5-2b
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figures import figstyle
from analysis.probe_layers import history, logit, probe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--test-frac", type=float, default=0.2)
    args = p.parse_args()

    shards = [np.load(f) for f in sorted(glob.glob(f"{args.data}/shard_*.npz"))]
    layers = [int(x) for x in shards[0]["memory_layers"]]
    actions = np.concatenate([s["actions"] for s in shards])
    E, T = actions.shape
    y = np.concatenate([logit(np.concatenate([s["llm_belief"] for s in shards])[:, 1:]), history(actions)], -1)
    y = y.reshape(E * T, -1)
    n_tr = round(E * (1 - args.test_frac)) * T
    fit = lambda x: probe(x[:n_tr], y[:n_tr], x[n_tr:], y[n_tr:])

    res = {"memory_layers": layers, "states": {}}
    for i, layer in enumerate(layers):
        pool = np.concatenate([s["pool"][:, :, layer + 1] for s in shards]).astype(np.float32)  # after layer L
        variants = {
            "recurrent memory S": np.concatenate([s["memory"][:, 1:, i] for s in shards]).astype(np.float32),
            "residual, current turn": pool[:, 1:],
            "residual, mean of turns so far": (np.cumsum(pool, 1) / np.arange(1, T + 2)[None, :, None])[:, 1:],
        }
        for name, x in variants.items():
            r2 = fit(x.reshape(E * T, -1))
            res["states"].setdefault(name, {"belief": [], "history": []})
            res["states"][name]["belief"].append(float(r2[:4].mean()))
            res["states"][name]["history"].append(float(r2[4:].mean()))
            print(f"layer {layer:2d} {name:32s} belief R² {r2[:4].mean():.3f}  history R² {r2[4:].mean():.3f}",
                  flush=True)

    figstyle.apply()
    fig, axes = plt.subplots(1, 2, figsize=(figstyle.DOUBLE, 2.6), sharey=True)
    styles = {"recurrent memory S": (figstyle.BLUE, "o", "-"), "residual, current turn": (figstyle.GREY, "s", ":"),
              "residual, mean of turns so far": (figstyle.VERMILLION, "^", "--")}
    for ax, target, title in zip(axes, ("belief", "history"),
                                 ("LLM belief (logit, 4 attributes)", "Clue history (16 signed counts)")):
        for name, (colour, marker, ls) in styles.items():
            ax.plot(layers, res["states"][name][target], color=colour, marker=marker, ls=ls, label=name)
        ax.set_xticks(layers)
        ax.set(xlabel="Decoder layer (Gated DeltaNet)", ylim=(0, 1))
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("Test R²")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
    figstyle.panel_labels(axes)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "memory_probes.json").write_text(json.dumps(res, indent=1))
    figstyle.save(fig, out / "memory_probes")


if __name__ == "__main__":
    main()
