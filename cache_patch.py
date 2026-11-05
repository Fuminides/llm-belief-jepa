"""Where a clue's effect lives in Qwen3.5's cache: which pieces are sufficient and which are necessary.

latent_edit.py found that the recurrent memory S of one Gated DeltaNet layer, even the real one after reading a clue,
moves the LLM's belief by nothing, and that the clue's effect only comes back when S, the attention keys and values and
the convolution states are all patched together. This asks the converse question too. At a test state (episode, turn
t) and a random clue a', the cache after reading a' (real) and the one before (base) differ in three kinds of piece:
  S     the recurrent memory of each of the 18 Gated DeltaNet layers
  conv  the short causal convolution state of the same layers (the last tokens' projections)
  KV    the keys and values of the 6 full-attention layers (all six together: the clue's tokens are appended to each,
        and the layers must agree on the sequence length)
The conditions start from base and copy pieces from real:
  every subset of {S, conv, KV}                   sufficiency and necessity of each kind (all three = real, a check)
  all but the S of one layer                       is any single memory necessary?
  KV and the S and conv of the layers below k      depth: how much the lower (or upper) layers carry, with the
  (or from k up), and the same without KV          attention route open or closed

  python cache_patch.py collect   # runs the LLM; saves data/cache_patch.npz
  python cache_patch.py analyse   # figure and JSON in results/qwen3.5-2b-memory/
"""
import argparse
import copy
import glob
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import figstyle
import game
from collect import question_batch, read_beliefs, readout
from probe_layers import logit

MODEL = "Qwen/Qwen3.5-2B"
DATA_DIR = "data/qwen3.5-2b-memory"
DATA = Path("data/cache_patch.npz")
OUT = Path("results/qwen3.5-2b-memory")
KINDS = ("S", "conv", "KV")
CUTS = (4, 8, 12, 16, 20)  # layer boundaries for the depth conditions


def conditions(linear):
    """Name -> (layers whose S to copy, layers whose conv state to copy, copy KV)."""
    out = {"base": ((), (), False)}
    for r in range(1, 4):
        for kinds in combinations(KINDS, r):
            out[" + ".join(kinds)] = (linear if "S" in kinds else (), linear if "conv" in kinds else (), "KV" in kinds)
    for i in linear:
        out[f"all but S{i}"] = ([j for j in linear if j != i], linear, True)
    for kv in (True, False):
        for k in CUTS:
            below, above = [i for i in linear if i < k], [i for i in linear if i >= k]
            tag = ", KV" if kv else ""
            out[f"below {k}{tag}"] = (below, below, kv)
            out[f"from {k}{tag}"] = (above, above, kv)
    return out


def collect(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=device).eval()
    intro_text, questions = readout(tok, "bold")
    batch = question_batch(questions, device)
    ask = lambda past: read_beliefs(llm, past, questions, *batch)[0]
    linear = [i for i, kind in enumerate(llm.config.layer_types) if kind == "linear_attention"]
    attention = [i for i, kind in enumerate(llm.config.layer_types) if kind == "full_attention"]
    conds = conditions(linear)

    actions = np.concatenate([np.load(f)["actions"] for f in sorted(glob.glob(f"{DATA_DIR}/shard_*.npz"))])
    test = np.arange(round(len(actions) * 0.8), len(actions))
    rng = np.random.default_rng(args.seed)
    episodes = rng.choice(test, size=min(args.states, len(test)), replace=False)
    turns = rng.integers(actions.shape[1], size=len(episodes))
    clues = rng.integers(game.N_ACTIONS, size=len(episodes))

    def run(text, past):
        ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        return llm(ids, past_key_values=past, use_cache=True).past_key_values

    def patched(past, real, S, conv, kv):
        c = copy.deepcopy(past)
        for i in S:
            c.layers[i].recurrent_states[0].copy_(real.layers[i].recurrent_states[0])
        for i in conv:
            c.layers[i].conv_states[0].copy_(real.layers[i].conv_states[0])
        for i in attention if kv else []:
            c.layers[i].keys, c.layers[i].values = real.layers[i].keys.clone(), real.layers[i].values.clone()
        return ask(c)

    beliefs = []
    with torch.no_grad():
        for n, (e, t, a_new) in enumerate(zip(episodes, turns, clues)):
            bos = tok.bos_token
            ids = tok(intro_text, add_special_tokens=not (bos and intro_text.startswith(bos)),
                      return_tensors="pt").input_ids.to(device)
            past = llm(ids, use_cache=True).past_key_values
            for x in actions[e, :t]:
                past = run(game.clue_text(x), past)
            real = run(game.clue_text(a_new), copy.deepcopy(past))
            beliefs.append([ask(real)] + [patched(past, real, *spec) for spec in conds.values()])
            if (n + 1) % 25 == 0:
                print(f"{n + 1}/{len(episodes)} states", flush=True)
    DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez(DATA, beliefs=np.array(beliefs, dtype=np.float32), conditions=np.array(["real"] + list(conds)),
             episodes=episodes, turns=turns, clues=clues, linear=np.array(linear))


def analyse(args):
    d = np.load(DATA)
    names = list(d["conditions"])
    linear = [int(i) for i in d["linear"]]
    change = logit(d["beliefs"]) - logit(d["beliefs"])[:, names.index("base")][:, None]
    attr, value = np.array(game.ACTIONS)[d["clues"]].T[:2]
    sign = np.where(value == 0, 1.0, -1.0)  # the belief is P(value 0)
    toward = sign[:, None] * change[np.arange(len(attr)), :, attr]  # [N, C]
    real = toward[:, names.index("real")]
    rng = np.random.default_rng(0)
    boot = rng.integers(len(real), size=(1000, len(real)))
    frac = {}
    for c, name in enumerate(names):
        ratios = toward[boot, c].mean(1) / real[boot].mean(1)
        frac[name] = (float(toward[:, c].mean() / real.mean()), *np.percentile(ratios, [2.5, 97.5]).tolist())
    res = {"states": len(real), "real_toward": float(real.mean()),
           "fraction_of_real": {k: {"mean": v[0], "ci95": v[1:]} for k, v in frac.items() if k != "real"}}
    print(json.dumps(res, indent=1))

    figstyle.apply()
    fig, axes = plt.subplots(1, 3, figsize=(figstyle.DOUBLE, 2.6), gridspec_kw={"width_ratios": [1, 1.4, 1]})
    subsets = [" + ".join(k) for r in range(1, 4) for k in combinations(KINDS, r)]
    ax = axes[0]
    ax.barh(range(len(subsets)), [frac[s][0] for s in subsets], color=figstyle.BLUE,
            xerr=np.abs(np.array([[frac[s][1], frac[s][2]] for s in subsets]).T - [frac[s][0] for s in subsets]))
    ax.set_yticks(range(len(subsets)), subsets)
    ax.invert_yaxis()
    ax.axvline(1, color=figstyle.GREY, lw=0.5, ls="--")
    ax.set_xlabel("fraction of the clue's effect")
    ax.set_title("Copied from the real cache")
    ax = axes[1]
    loo = [frac[f"all but S{i}"] for i in linear]
    ax.errorbar(linear, [v[0] for v in loo], yerr=np.abs(np.array([v[1:] for v in loo]).T - [v[0] for v in loo]),
                fmt="o", color=figstyle.VERMILLION)
    ax.axhline(1, color=figstyle.GREY, lw=0.5, ls="--")
    ax.axhline(frac["conv + KV"][0], color=figstyle.GREY, lw=0.5, ls=":", label="no S at all")
    ax.set_xlabel("layer whose S is left out")
    ax.set_ylabel("fraction of the clue's effect")
    ax.set_title("Everything but one memory")
    ax.legend()
    ax = axes[2]
    for tag, colour in ((", KV", figstyle.BLUE), ("", figstyle.ORANGE)):
        label = "with KV" if tag else "without KV"
        ax.plot(CUTS, [frac[f"below {k}{tag}"][0] for k in CUTS], "o-", color=colour, label=f"below k, {label}")
        ax.plot(CUTS, [frac[f"from {k}{tag}"][0] for k in CUTS], "s--", color=colour, label=f"from k, {label}")
    ax.set_xlabel("k (layer)")
    ax.set_ylabel("fraction of the clue's effect")
    ax.set_title("S and conv by depth")
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    figstyle.panel_labels(axes)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cache_patch.json").write_text(json.dumps(res, indent=1))
    figstyle.save(fig, OUT / "cache_patch")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["collect", "analyse"])
    p.add_argument("--states", type=int, default=300)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()
    {"collect": collect, "analyse": analyse}[args.step](args)
