"""Test 4, interventional closure: edit the latent model's state, write the edit into the LLM's recurrent memory, and
check that the LLM's belief moves as the latent model says.

The interventions so far went one way: change the LLM, see whether the latent model follows (intervene.py). A latent
model that is a causal abstraction of the LLM (Geiger et al.) must also work the other way: a change made in latent
space, carried into the LLM, has to change the LLM's behaviour as the latent model predicts. Here the change is a clue
the LLM has not read. At a test state (episode, turn t) the predictor gives ẑ = P(z_t, a') for a random clue a'; ẑ is
turned into a change of the recurrent memory S of one Gated DeltaNet layer, written into the cache, and the LLM's
belief is read on the usual branches, without the clue ever being in the context. Conditions, per state:
  real          the LLM reads a' (what the edits try to reproduce)
  latent        the latent edit through a decoder D (z -> projected S, trained on natural episodes): D(ẑ) - D(z_t)
  inverted      the latent edit by inverting the encoder: the smallest change of the projected S whose encoding is ẑ
                (Gauss-Newton with minimum-norm steps)
  random        a random change of S as large as the latent one, in the same subspace
  oracle        the S of that layer after really reading a', patched in: the most an edit of this memory alone can do
  oracle, seen  the same, keeping only the part of the change the latent model sees (its 16 x 16 projection per head)
and over several layers at once: the latent edits of the four modelled layers together, the oracle on the same four,
and the oracle on all 18 Gated DeltaNet layers (how much of the clue's effect the recurrent memories carry at all). The
other place the clue can be is the KV cache of the 6 full-attention layers: "oracle, attention only" appends the clue's
keys and values there and leaves every memory as it was, and "oracle, memories and attention" does both. The third
piece of state is the short causal convolution of each Gated DeltaNet layer, which keeps the last tokens' projections,
so the first tokens of the question still see the last words of the clue: "oracle, convolutions only" copies those, and
"oracle, everything" copies all three, which must reproduce the real reading (a check of the patching).

The latent model reads S through R1' S R2 per head (collect.memory_reader); the smallest change of S with projection
dP is R1 (R1'R1)^-1 dP (R2'R2)^-1 R2'. The latent models are those of the earlier analyses (MLP encoder on S, plain
clues, the first 80% of the episodes); the states come from the other 20%.

  python latent_edit.py collect   # runs the LLM; saves data/latent_edit.npz
  python latent_edit.py analyse   # figure and JSON in results/qwen3.5-2b-memory/
"""
import argparse
import copy
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import figstyle
import game
import jepa
from collect import memory_projection, question_batch, read_beliefs, readout
from probe_layers import logit

MODEL = "Qwen/Qwen3.5-2B"
DATA_DIR = "data/qwen3.5-2b-memory"
DATA = Path("data/latent_edit.npz")
OUT = Path("results/qwen3.5-2b-memory")
EDITS = ["oracle", "oracle, seen", "latent", "inverted", "random"]  # per modelled layer
JOINT = ["latent, four layers", "oracle, four layers", "oracle, all memories", "oracle, attention only",
         "oracle, convolutions only", "oracle, memories and attention", "oracle, everything"]


def train_decoder(model, h, n_train, path, device, steps=3000):
    """D: latent -> normalised projected S, trained on the encoded training states (encoder frozen). Returns D and its
    R² on the test states."""
    with torch.no_grad():
        z = model.enc(h.to(device)).reshape(-1, model.read.in_features)
    x = h.reshape(-1, h.shape[-1]).to(device)
    n = n_train * h.shape[1]
    dec = jepa.mlp(z.shape[-1], 512, x.shape[-1]).to(device)
    if path.exists():
        dec.load_state_dict(torch.load(path, map_location=device))
    else:
        torch.manual_seed(0)
        opt = torch.optim.AdamW(dec.parameters(), lr=1e-3, weight_decay=1e-4)
        for idx in jepa.batches(n, steps):
            loss = F.mse_loss(dec(z[idx]), x[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        torch.save(dec.state_dict(), path)
    dec.eval()
    with torch.no_grad():
        r2 = jepa.r2(dec(z[n:]).cpu().numpy(), x[n:].cpu().numpy())
    return dec, r2


def latent_models(device):
    """Per modelled memory layer: (latent model, decoder, mu, sd), and the decoders' test R²."""
    from explain import get_model
    layers = [int(x) for x in np.load(sorted(glob.glob(f"{DATA_DIR}/shard_*.npz"))[0])["memory_layers"]]
    models, dec_r2 = [], []
    for i in range(len(layers)):
        h, a, b, _ = jepa.load(DATA_DIR, "memory", i)
        n_train = round(len(h) * 0.8)
        mu, sd = h[:n_train].mean((0, 1)), h[:n_train].std((0, 1)) + 1e-6
        ht = torch.tensor((h - mu) / sd)
        model = get_model(ht, torch.tensor(a), torch.tensor(b), n_train,
                          Path(f"{DATA_DIR}/latent_memory_layer{i}_mlp.pt"), "memory", "mlp", 3000, device)
        dec, r2 = train_decoder(model, ht, n_train, Path(f"{DATA_DIR}/decoder_memory_layer{i}_mlp.pt"), device)
        models.append((model, dec, torch.tensor(mu, device=device), torch.tensor(sd, device=device)))
        dec_r2.append(r2)
        print(f"memory layer {layers[i]}: decoder test R² {r2:.3f}", flush=True)
    return layers, models, dec_r2


def invert(enc, x0, target, iters=5, eps=1e-4):
    """Smallest change of the encoder's input that maps it to `target`, by Gauss-Newton with minimum-norm steps."""
    x = x0.clone()
    eye = torch.eye(target.shape[0], device=x.device)
    for _ in range(iters):
        jac = torch.func.jacrev(enc)(x)  # [d_z, d_in]
        x = x + jac.T @ torch.linalg.solve(jac @ jac.T + eps * eye, target - enc(x))
    return x - x0


def collect(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=device).eval()
    intro_text, questions = readout(tok, "bold")
    batch = question_batch(questions, device)
    ask = lambda past: read_beliefs(llm, past, questions, *batch)[0]
    r1, r2 = memory_projection(llm)
    left, right = r1 @ torch.linalg.inv(r1.T @ r1), torch.linalg.inv(r2.T @ r2) @ r2.T
    d = r1.shape[1]
    project = lambda S: (r1.T @ S @ r2).reshape(-1)
    lift = lambda dp: left @ dp.view(-1, d, d) @ right
    layers, models, dec_r2 = latent_models(device)
    linear = [i for i, kind in enumerate(llm.config.layer_types) if kind == "linear_attention"]
    attention = [i for i, kind in enumerate(llm.config.layer_types) if kind == "full_attention"]

    shards = [np.load(f) for f in sorted(glob.glob(f"{DATA_DIR}/shard_*.npz"))]
    actions = np.concatenate([s["actions"] for s in shards])
    stored = np.concatenate([s["memory"] for s in shards])  # [E, T+1, layers, 4096], to check the states match
    test = np.arange(round(len(actions) * 0.8), len(actions))
    rng = np.random.default_rng(args.seed)
    episodes = rng.choice(test, size=min(args.states, len(test)), replace=False)
    turns = rng.integers(actions.shape[1], size=len(episodes))  # state after t clues, t = 0..T-1
    clues = rng.integers(game.N_ACTIONS, size=len(episodes))  # the clue a' the edits stand in for

    def run(text, past):
        ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        return llm(ids, past_key_values=past, use_cache=True).past_key_values

    def patched(past, changes, kv_from=None, conv_from=None):
        """Beliefs with the recurrent states of some layers changed ({layer: dS}); with kv_from, the keys and values of
        the full-attention layers taken from that cache; with conv_from, the convolution states of the linear ones."""
        c = copy.deepcopy(past)
        for i, dS in changes.items():
            state = c.layers[i].recurrent_states[0]
            state[0] += dS.to(state.dtype)
        for i in attention if kv_from is not None else []:
            c.layers[i].keys, c.layers[i].values = kv_from.layers[i].keys.clone(), kv_from.layers[i].values.clone()
        for i in linear if conv_from is not None else []:
            c.layers[i].conv_states[0] = conv_from.layers[i].conv_states[0].clone()
        return ask(c)

    conditions = ["base", "real"] + [f"{e}, layer {L}" for L in layers for e in EDITS] + JOINT
    beliefs, predicted, read_back, norms, match = [], [], [], [], []
    with torch.no_grad():
        for n, (e, t, a_new) in enumerate(zip(episodes, turns, clues)):
            bos = tok.bos_token
            ids = tok(intro_text, add_special_tokens=not (bos and intro_text.startswith(bos)),
                      return_tensors="pt").input_ids.to(device)
            past = llm(ids, use_cache=True).past_key_values
            for x in actions[e, :t]:
                past = run(game.clue_text(x), past)
            real_past = run(game.clue_text(a_new), copy.deepcopy(past))
            S = {i: past.layers[i].recurrent_states[0][0].float() for i in linear}
            S_real = {i: real_past.layers[i].recurrent_states[0][0].float() for i in linear}
            out = {"base": ask(past), "real": ask(real_past)}
            pred_n, back_n, norm_n, match_n, latent_edits = [], [], [], [], {}
            for j, (L, (model, dec, mu, sd)) in enumerate(zip(layers, models)):
                p = project(S[L])
                match_n.append(float(np.corrcoef(p.cpu().numpy(), stored[e, t, j].astype(np.float32))[0, 1]))
                x = (p - mu) / sd
                z = model.enc(x)
                zhat = model.step(z, torch.tensor(int(a_new), device=device))
                pred_n.append((model.read(zhat) - model.read(z)).tolist())
                dx_dec = dec(zhat) - dec(z)
                with torch.enable_grad():
                    dx_inv = invert(model.enc, x, zhat)
                back_n.append([(model.read(model.enc(x + dx)) - model.read(z)).tolist() for dx in (dx_dec, dx_inv)])
                dS_real = S_real[L] - S[L]
                edits = {"oracle": dS_real, "oracle, seen": lift(project(dS_real)), "latent": lift(dx_dec * sd),
                         "inverted": lift(dx_inv * sd)}
                g = torch.Generator(device=device).manual_seed(int(args.seed * 100003 + n * 31 + j))
                rand = lift(torch.randn(p.shape, generator=g, device=device))
                edits["random"] = rand * edits["latent"].norm() / rand.norm()
                norm_n.append([float(edits[k].norm()) for k in EDITS] + [float(S[L].norm())])
                latent_edits[L] = edits["latent"]
                for k in EDITS:
                    out[f"{k}, layer {L}"] = patched(past, {L: edits[k]})
            out["latent, four layers"] = patched(past, latent_edits)
            out["oracle, four layers"] = patched(past, {L: S_real[L] - S[L] for L in layers})
            out["oracle, all memories"] = patched(past, {i: S_real[i] - S[i] for i in linear})
            out["oracle, attention only"] = patched(past, {}, kv_from=real_past)
            out["oracle, convolutions only"] = patched(past, {}, conv_from=real_past)
            all_S = {i: S_real[i] - S[i] for i in linear}
            out["oracle, memories and attention"] = patched(past, all_S, kv_from=real_past)
            out["oracle, everything"] = patched(past, all_S, kv_from=real_past, conv_from=real_past)
            beliefs.append([out[c] for c in conditions])
            predicted.append(pred_n)
            read_back.append(back_n)
            norms.append(norm_n)
            match.append(match_n)
            if (n + 1) % 25 == 0:
                print(f"{n + 1}/{len(episodes)} states; live vs stored projected S, correlation "
                      f"{np.mean(match):.4f}", flush=True)
    DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez(DATA, beliefs=np.array(beliefs, dtype=np.float32), conditions=np.array(conditions),
             predicted=np.array(predicted, dtype=np.float32), read_back=np.array(read_back, dtype=np.float32),
             norms=np.array(norms, dtype=np.float32), match=np.array(match, dtype=np.float32),
             episodes=episodes, turns=turns, clues=clues, layers=np.array(layers), decoder_r2=np.array(dec_r2))


def analyse(args):
    d = np.load(DATA)
    conditions = list(d["conditions"])
    layers = [int(x) for x in d["layers"]]
    llm = logit(d["beliefs"])  # [N, C, 4]
    change = llm - llm[:, :1]  # against the state before the edit
    attr, value, negated, source = np.array(game.ACTIONS)[d["clues"]].T
    sign = np.where(value == 0, 1.0, -1.0)  # the belief is P(value 0)
    N = len(attr)
    rows = np.arange(N)
    toward = sign[:, None] * change[rows, :, attr]  # [N, C]: shift towards what the clue asserts
    other = np.ones((N, 4), bool)
    other[rows, attr] = False
    spill = np.abs(change.transpose(0, 2, 1)[other].reshape(N, 3, -1)).mean(1)  # [N, C]
    real = toward[:, conditions.index("real")]
    pred = d["predicted"]  # [N, layers, 4]: the latent model's predicted change of the belief logits
    pred_toward = sign[:, None] * pred[rows, :, attr]

    res = {"states": N, "decoder_test_r2": dict(zip(map(str, layers), d["decoder_r2"].tolist())),
           "live_vs_stored_state_correlation": float(d["match"].mean()),
           "real": {"toward": float(real.mean()), "spillover": float(spill[:, 1].mean())}, "conditions": {}}
    for c, name in enumerate(conditions[2:], 2):
        res["conditions"][name] = {
            "toward": float(toward[:, c].mean()), "fraction_of_real": float(toward[:, c].mean() / real.mean()),
            "correlation_with_real": float(np.corrcoef(toward[:, c], real)[0, 1]),
            "spillover": float(spill[:, c].mean())}
    names = ["latent", "inverted"]
    norm_names = EDITS + ["S"]
    res["layers"] = {}
    for j, L in enumerate(layers):
        block = {"latent_forecast_toward": float(pred_toward[:, j].mean()),
                 "latent_forecast_r2_real": jepa.r2(pred[:, j], change[:, 1]),
                 "edit_norm": {k: float(d["norms"][:, j, norm_names.index(k)].mean()) for k in norm_names}}
        for k, name in enumerate(names):
            got = change[:, conditions.index(f"{name}, layer {L}")]
            # Closure: the LLM's change after the latent edit against what the latent model says the edit does
            block[f"{name}_closure_r2"] = jepa.r2(pred[:, j], got)
            block[f"{name}_closure_slope"] = float(np.polyfit(pred[:, j].ravel(), got.ravel(), 1)[0])
            block[f"{name}_encoder_reads_edit_r2"] = jepa.r2(d["read_back"][:, j, k], pred[:, j])
        res["layers"][str(L)] = block
    types = {f"{s} {'neg.' if g else 'aff.'}": (source == si) & (negated == g)
             for si, s in enumerate(game.SOURCE_NAMES) for g in (0, 1)}
    res["by_clue_type"] = {t: {"real": float(real[m].mean()), "n": int(m.sum()),
                               **{f"forecast, layer {L}": float(pred_toward[m, j].mean()) for j, L in enumerate(layers)},
                               **{f"latent edit, layer {L}": float(toward[m, conditions.index(f'latent, layer {L}')].mean())
                                  for L in layers},
                               "latent, four layers": float(toward[m, conditions.index("latent, four layers")].mean())}
                           for t, m in types.items()}
    print(json.dumps(res, indent=1))

    figstyle.apply()
    fig, axes = plt.subplots(1, 3, figsize=(figstyle.DOUBLE, 2.8), width_ratios=[1.3, 1.2, 1])
    colours = dict(zip(EDITS, [figstyle.GREY, figstyle.SKY, figstyle.BLUE, figstyle.GREEN, figstyle.ORANGE]))
    x = np.arange(len(layers))
    w = 0.16
    for k, name in enumerate(EDITS):
        vals = [res["conditions"][f"{name}, layer {L}"]["toward"] for L in layers]
        axes[0].bar(x + (k - 2) * w, vals, w * 0.95, color=colours[name],
                    label={"latent": "anchored JEPA, decoded", "inverted": "anchored JEPA, inverted"}.get(name, name))
    axes[0].axhline(res["real"]["toward"], color=figstyle.VERMILLION, ls="--", lw=1.2, label="reading the clue")
    axes[0].axhline(0, color="black", lw=0.5)
    axes[0].set_xticks(x, [f"layer {L}" for L in layers])
    axes[0].set(ylabel="Shift towards the clue (log-odds)")
    axes[0].set_title("Edits of one layer's memory S", loc="left")

    parts = ["latent, four layers", "oracle, four layers", "oracle, all memories", "oracle, convolutions only",
             "oracle, attention only", "oracle, memories and attention", "oracle, everything"]
    vals = [res["conditions"][p]["toward"] for p in parts] + [res["real"]["toward"]]
    names = ["anchored JEPA edit, S of 4 layers", "S of 4 layers", "S of all 18 layers", "convolution states", "attention KV",
             "S + attention KV", "S + KV + convolutions", "reading the clue"]
    bar_colours = [figstyle.BLUE] + [figstyle.GREY] * (len(parts) - 1) + [figstyle.VERMILLION]
    y = np.arange(len(vals))[::-1]
    axes[1].barh(y, vals, 0.7, color=bar_colours)
    axes[1].set_yticks(y, names)
    axes[1].axvline(0, color="black", lw=0.5)
    axes[1].set(xlabel="Shift towards the clue (log-odds)")
    axes[1].set_title("Which part of the cache carries it", loc="left")

    best = max(layers, key=lambda L: res["conditions"][f"latent, layer {L}"]["toward"])
    j = layers.index(best)
    got = toward[:, conditions.index(f"latent, layer {best}")]
    closure_panel(axes[2], pred_toward[:, j], real, got, best)
    handles, labels = [], []
    for a in axes[:2]:
        for h_, l_ in zip(*a.get_legend_handles_labels()):
            if l_ not in labels:
                handles.append(h_)
                labels.append(l_)
    fig.legend(handles, labels, loc="outside lower center", ncol=4)
    figstyle.panel_labels(axes)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latent_edit.json").write_text(json.dumps(res, indent=1))
    figstyle.save(fig, OUT / "latent_edit")
    fig, ax = plt.subplots(figsize=(3.3, 2.9))  # the manuscript's figure: panel C alone
    closure_panel(ax, pred_toward[:, j], real, got, best)
    figstyle.save(fig, OUT / "latent_edit_closure")


def closure_panel(ax, forecast, real, got, layer):
    """The anchored JEPA's forecast of the shift against the LLM's shift on reading the clue and on the edit alone."""
    ax.scatter(forecast, real, s=5, color=figstyle.VERMILLION, alpha=0.4, lw=0, label="LLM reads the clue")
    ax.scatter(forecast, got, s=5, color=figstyle.BLUE, alpha=0.6, lw=0, label="LLM gets only the edit of S")
    lim = np.percentile(np.abs(np.concatenate([forecast, got, real])), 99) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color="black", lw=0.5, ls=":")
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="Forecast shift (log-odds)", ylabel="LLM's actual shift (log-odds)")
    ax.set_title(f"Forecast against effect, layer {layer}", loc="left")
    ax.legend(loc="lower right", fontsize=6.5, markerscale=2.5)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["collect", "analyse"])
    p.add_argument("--states", type=int, default=400)
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args()
    {"collect": collect, "analyse": analyse}[args.step](args)
