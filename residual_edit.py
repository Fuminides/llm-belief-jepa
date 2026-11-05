"""Test 4 with the edit made while the LLM reads: a latent edit added to the residual stream, so every later layer
recomputes its memory, its keys and values and its convolution state from it.

latent_edit.py writes edits into the recurrent memory S of finished turns. There even the real S after a clue,
patched into all 18 Gated DeltaNet layers, carries little of the clue's effect: the rest is in the KV cache of the
full-attention layers and in the short convolutions, and the pieces only act together. Here the edit is made upstream
instead. At a test state the LLM is about to read clue t (a_t); the latent model says that reading a' instead would
give ẑ = P(z_{t-1}, a'). While clue a_t runs, a shift is added to the output of layer L at all of its tokens, chosen so
that the encoder of layer L (running mean of the turn means, as in intervene.py) reads ẑ instead of z_t. The LLM's
belief after the edited clue is compared with its belief after really reading a', and with the latent model's
forecast R(ẑ) - R(z_t). The encoder only sees the turn mean, so a shift d of the mean can be made in many ways; a
first test spread it evenly over the clue's tokens, and even the oracle below then barely moved the LLM (slope 0.01-0.06
on the real change): the clue's meaning is carried by a few tokens. So every shift is put on the token of the value word
("blue" in "is not blue"), as in intervene.py, scaled by the number of tokens n of the clue so that the turn mean moves
by d; "oracle, spread" keeps the even version as the reference. Shifts, per layer L:
  latent     decoded: D(ẑ) - D(z_t), with D a decoder z -> running-mean state trained on natural episodes
  inverted   the smallest change of the running mean whose encoding is ẑ (Gauss-Newton, minimum-norm steps)
  oracle     the difference of the layer-L turn means of a' and a_t: what the encoder would see if the LLM had read a'
             (the most a uniform shift can do; the encoder only sees means)
  random     a random shift as large as the latent one
A shift d of the running mean at turn t is a shift (t + 1) d of the turn mean (the running mean is over the intro and
the t clues). Latent models trained on plain clues, the first 80% of the episodes; the states come from the other 20%.

  python residual_edit.py collect   # runs the LLM; saves data/residual_edit.npz
  python residual_edit.py analyse   # figure and JSON in results/qwen3.5-2b-memory/
"""
import argparse
import copy
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import figstyle
import game
import jepa
from collect import question_batch, read_beliefs, readout
from intervene import clue_word, word_position
from latent_edit import invert, train_decoder
from probe_layers import logit

MODEL = "Qwen/Qwen3.5-2B"
DATA_DIR = "data/qwen3.5-2b-memory"
DATA = Path("data/residual_edit.npz")
OUT = Path("results/qwen3.5-2b-memory")
LAYERS = [3, 6, 9, 12, 18]  # hidden_states index: the output of decoder layer L - 1
EDITS = ["oracle", "latent", "inverted", "random"]  # on the value word
SPREAD = "oracle, spread"


def latent_models(device):
    """Per layer: (latent model on running means, decoder, mu, sd); and the decoders' test R²."""
    from explain import get_model
    models, dec_r2 = [], []
    for L in LAYERS:
        h, a, b, _ = jepa.load(DATA_DIR, "running", L)
        n_train = round(len(h) * 0.8)
        mu, sd = h[:n_train].mean((0, 1)), h[:n_train].std((0, 1)) + 1e-6
        ht = torch.tensor((h - mu) / sd)
        model = get_model(ht, torch.tensor(a), torch.tensor(b), n_train,
                          Path(f"{DATA_DIR}/latent_running_layer{L}_mlp.pt"), "running", "mlp", 3000, device)
        dec, r2 = train_decoder(model, ht, n_train, Path(f"{DATA_DIR}/decoder_running_layer{L}_mlp.pt"), device)
        models.append((model, dec, torch.tensor(mu, device=device), torch.tensor(sd, device=device)))
        dec_r2.append(r2)
        print(f"layer {L}: decoder test R² {r2:.3f}", flush=True)
    return models, dec_r2


def collect(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=device).eval()
    intro_text, questions = readout(tok, "bold")
    batch = question_batch(questions, device)
    models, dec_r2 = latent_models(device)

    shards = [np.load(f) for f in sorted(glob.glob(f"{DATA_DIR}/shard_*.npz"))]
    actions = np.concatenate([s["actions"] for s in shards])
    pool = np.concatenate([s["pool"][:, :, LAYERS] for s in shards]).astype(np.float32)  # [E, T+1, layers, d]
    test = np.arange(round(len(actions) * 0.8), len(actions))
    rng = np.random.default_rng(args.seed)
    episodes = rng.choice(test, size=min(args.states, len(test)), replace=False)
    turns = rng.integers(1, actions.shape[1] + 1, size=len(episodes))  # the clue that is edited, 1..T
    clues = np.array([rng.choice([x for x in range(game.N_ACTIONS) if x != actions[e, t - 1]])
                      for e, t in zip(episodes, turns)])  # a', never the clue really read

    def read_clue(text, past, steer=None):
        """Run a clue on a copy of the cache, optionally adding (layer, shift, position or None for every token) to
        that layer's output; returns the beliefs after it and the turn mean of the output of each layer in LAYERS."""
        ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        handle = None
        if steer is not None:
            layer, shift, position = steer
            full = torch.zeros(ids.shape[1], shift.shape[0], device=device)
            full[slice(None) if position is None else position] = shift
            handle = llm.model.layers[layer - 1].register_forward_hook(lambda mod, inp, out: out + full.to(out.dtype))
        try:
            out = llm(ids, past_key_values=copy.deepcopy(past), use_cache=True, output_hidden_states=True)
        finally:
            if handle is not None:
                handle.remove()
        means = [out.hidden_states[L][0].float().mean(0) for L in LAYERS]
        return read_beliefs(llm, out.past_key_values, questions, *batch)[0], means

    conditions = ["base", "real"] + [f"{e}, layer {L}" for L in LAYERS for e in EDITS + [SPREAD]]
    beliefs, predicted, read_back, norms = [], [], [], []
    with torch.no_grad():
        for n, (e, t, a_new) in enumerate(zip(episodes, turns, clues)):
            bos = tok.bos_token
            ids = tok(intro_text, add_special_tokens=not (bos and intro_text.startswith(bos)),
                      return_tensors="pt").input_ids.to(device)
            past = llm(ids, use_cache=True).past_key_values
            for x in actions[e, :t - 1]:
                ids = tok(game.clue_text(x), add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                past = llm(ids, past_key_values=past, use_cache=True).past_key_values
            out = {}
            text = game.clue_text(actions[e, t - 1])
            n_tokens = len(tok(text, add_special_tokens=False).input_ids)
            word = word_position(tok, text, clue_word(actions[e, t - 1]))
            out["base"], base_means = read_clue(text, past)
            out["real"], real_means = read_clue(game.clue_text(a_new), past)
            pred_n, back_n, norm_n = [], [], []
            for j, (L, (model, dec, mu, sd)) in enumerate(zip(LAYERS, models)):
                before = torch.tensor(pool[e, :t, j], device=device)  # turn means up to clue t - 1 (intro included)
                running = lambda turn_mean: (before.sum(0) + turn_mean) / (t + 1)
                x_prev, x = (before.mean(0) - mu) / sd, (running(base_means[j]) - mu) / sd
                z = model.enc(x)
                zhat = model.step(model.enc(x_prev), torch.tensor(int(a_new), device=device))
                pred_n.append((model.read(zhat) - model.read(z)).tolist())
                dx_dec = dec(zhat) - dec(z)
                with torch.enable_grad():
                    dx_inv = invert(model.enc, x, zhat)
                back_n.append([(model.read(model.enc(x + dx)) - model.read(z)).tolist() for dx in (dx_dec, dx_inv)])
                shifts = {"oracle": real_means[j] - base_means[j], "latent": (t + 1) * dx_dec * sd,
                          "inverted": (t + 1) * dx_inv * sd}
                g = torch.Generator(device=device).manual_seed(int(args.seed * 100003 + n * 31 + j))
                rand = torch.randn(shifts["latent"].shape, generator=g, device=device)
                shifts["random"] = rand * shifts["latent"].norm() / rand.norm()
                norm_n.append([float(shifts[k].norm()) for k in EDITS] + [float(base_means[j].norm())])
                for k in EDITS:  # shifts of the turn mean, put on the value word
                    out[f"{k}, layer {L}"] = read_clue(text, past, (L, n_tokens * shifts[k], word))[0]
                out[f"{SPREAD}, layer {L}"] = read_clue(text, past, (L, shifts["oracle"], None))[0]
            beliefs.append([out[c] for c in conditions])
            predicted.append(pred_n)
            read_back.append(back_n)
            norms.append(norm_n)
            if (n + 1) % 25 == 0:
                print(f"{n + 1}/{len(episodes)} states", flush=True)
    DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez(DATA, beliefs=np.array(beliefs, dtype=np.float32), conditions=np.array(conditions),
             predicted=np.array(predicted, dtype=np.float32), read_back=np.array(read_back, dtype=np.float32),
             norms=np.array(norms, dtype=np.float32), episodes=episodes, turns=turns, clues=clues,
             replaced=actions[episodes, turns - 1], layers=np.array(LAYERS), decoder_r2=np.array(dec_r2))


def analyse(args):
    """Only states where a' is about another attribute than a_t, so the two effects of reading a' instead of a_t can be
    told apart: adding a' (shift of the belief in what a' asserts) and removing a_t (shift away from what a_t asserted).
    Any edit that disrupts the clue removes a_t; only an edit that carries a' adds it."""
    d = np.load(DATA)
    conditions = list(d["conditions"])
    layers = [int(x) for x in d["layers"]]
    llm = logit(d["beliefs"])  # [N, C, 4]
    change = llm - llm[:, :1]  # against reading the clue the episode really had
    new, old = np.array(game.ACTIONS)[d["clues"]], np.array(game.ACTIONS)[d["replaced"]]
    keep = new[:, 0] != old[:, 0]
    rows = np.arange(len(llm))[keep]
    sign = lambda acts: np.where(acts[keep, 1] == 0, 1.0, -1.0)  # the belief is P(value 0)
    add = sign(new)[:, None] * change[rows, :, new[keep, 0]]  # [n, C]
    remove = -sign(old)[:, None] * change[rows, :, old[keep, 0]]
    pred_add = sign(new)[:, None] * d["predicted"][rows, :, new[keep, 0]]  # [n, layers]
    res = {"states": int(keep.sum()), "decoder_test_r2": dict(zip(map(str, layers), d["decoder_r2"].tolist())),
           "real": {"add": float(add[:, 1].mean()), "remove": float(remove[:, 1].mean())}, "layers": {}}
    for j, L in enumerate(layers):
        block = {"forecast_add": float(pred_add[:, j].mean()),
                 "forecast_add_correlation_real": float(np.corrcoef(pred_add[:, j], add[:, 1])[0, 1]),
                 "encoder_reads_latent_edit_r2": jepa.r2(d["read_back"][:, j, 0], d["predicted"][:, j]),
                 "shift_norm": {k: float(d["norms"][:, j, i].mean()) for i, k in enumerate(EDITS + ["turn mean"])}}
        for k in EDITS + [SPREAD]:
            c = conditions.index(f"{k}, layer {L}")
            block[k] = {"add": float(add[:, c].mean()), "add_fraction": float(add[:, c].mean() / add[:, 1].mean()),
                        "remove": float(remove[:, c].mean()),
                        "remove_fraction": float(remove[:, c].mean() / remove[:, 1].mean()),
                        "add_correlation_real": float(np.corrcoef(add[:, c], add[:, 1])[0, 1]),
                        "add_correlation_forecast": float(np.corrcoef(add[:, c], pred_add[:, j])[0, 1])}
        res["layers"][str(L)] = block
    print(json.dumps(res, indent=1))

    figstyle.apply()
    fig, axes = plt.subplots(1, 3, figsize=(figstyle.DOUBLE, 2.7))
    colours = dict(zip(EDITS + [SPREAD], [figstyle.GREY, figstyle.BLUE, figstyle.GREEN, figstyle.ORANGE,
                                          figstyle.PURPLE]))
    for k in EDITS + [SPREAD]:
        for ax, metric in zip(axes[:2], ("add_fraction", "remove_fraction")):
            ax.plot(layers, [res["layers"][str(L)][k][metric] for L in layers], color=colours[k], marker="o",
                    ls=":" if k == SPREAD else "-", label=k)
    for ax, title in zip(axes[:2], ("Adds the new clue a'", "Removes the clue it replaces")):
        ax.axhline(1, color=figstyle.VERMILLION, ls="--", lw=1.2, label="reading a'")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(layers)
        ax.set(xlabel="Layer of the edit", ylabel="Fraction of the real shift")
        ax.set_title(title, loc="left")
    best = max(layers, key=lambda L: res["layers"][str(L)]["latent"]["add"])
    j = layers.index(best)
    got = add[:, conditions.index(f"latent, layer {best}")]
    axes[2].scatter(pred_add[:, j], add[:, 1], s=5, color=figstyle.VERMILLION, alpha=0.4, lw=0, label="reading a'")
    axes[2].scatter(pred_add[:, j], got, s=5, color=figstyle.BLUE, alpha=0.6, lw=0, label=f"latent edit, layer {best}")
    lim = np.percentile(np.abs(np.concatenate([pred_add[:, j], got, add[:, 1]])), 99) * 1.1
    axes[2].plot([-lim, lim], [-lim, lim], color="black", lw=0.5, ls=":")
    axes[2].set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="Latent forecast of the shift (log-odds)",
                ylabel="LLM's shift (log-odds)")
    axes[2].set_title(f"Towards a', layer {best}", loc="left")
    handles, labels = [], []
    for a in axes:
        for h_, l_ in zip(*a.get_legend_handles_labels()):
            if l_ not in labels:
                handles.append(h_)
                labels.append(l_)
    fig.legend(handles, labels, loc="outside lower center", ncol=4)
    figstyle.panel_labels(axes)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "residual_edit.json").write_text(json.dumps(res, indent=1))
    figstyle.save(fig, OUT / "residual_edit")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["collect", "analyse"])
    p.add_argument("--states", type=int, default=400)
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args()
    {"collect": collect, "analyse": analyse}[args.step](args)
