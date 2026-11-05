"""Interventions in the LLM's residual stream: does the latent model see their effect on the belief?

Clue 4 of every episode is a colour clue. At layer 3, on the token of its colour word only, a direction is added:
  red vs blue       h(red) - h(blue) on that token, averaged over clue-like sentences
  warm vs cool      orange, yellow vs green, purple: colours absent from the game
  similar colours   crimson, scarlet vs navy, azure
  big vs small      another attribute of the game
  unrelated         early vs late ("the train is ...")
Each direction is scaled to the norm of the red-blue difference, times alpha: at alpha 1, red-blue added to "blue"
gives the layer-3 representation of "red". Prediction if the LLM reads it for its meaning: on "is blue" the belief in
red rises; on "is not blue" the same push means "is not red" and the belief in red falls.

The LLM's belief is read after clue 4 (the four attributes, and "orange or green"); the latent model (trained on
plain clues: running mean of the layer-3 turn means, MLP encoder) reads the intervened states. The criterion: its
readout should change when, and as much as, the LLM's belief does.

  python intervene.py collect        # runs the LLM; saves data/interventions.npz
  python intervene.py analyse        # latent model readouts; figure and JSON in results/qwen3.5-2b/
  python intervene.py collect-train  # episodes with one random intervention each, for consistency training
  python intervene.py consistency    # latent models trained with them, evaluated on the interventions above

Consistency training: the training interventions come from another family than the evaluated ones. On shape,
size and material clues, half the time the clue's own word direction (round-square, big-small, wooden-metal), else
a random direction; on colour clues only random directions, never red-blue. The readout loss on these intervened
episodes asks the encoder to read what the LLM believes after the intervention.
"""
import argparse
import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import figstyle
import game
import jepa
from collect import memory_reader, readout, run_episode
from probe_layers import logit

MODEL = "Qwen/Qwen3.5-2B"
LAYER = 3  # where the intervention is added
SAVED_LAYERS = [3, 6, 9, 12, 18, 24]  # layers whose states are kept, to test latent models on deeper layers
MEMORY_LAYERS = [2, 9, 17, 22]  # 0-based Gated DeltaNet layers whose recurrent state is kept
STEERED_CLUE = 4
TURNS = 6
EXTRA = [("orange", "green")]
DIRECTIONS = {"red vs blue": ("object", ["red"], ["blue"]),
              "warm vs cool": ("object", ["orange", "yellow"], ["green", "purple"]),
              "similar colours": ("object", ["crimson", "scarlet"], ["navy", "azure"]),
              "big vs small": ("object", ["big"], ["small"]),
              "unrelated": ("train", ["early"], ["late"])}
COLOUR_ACTIONS = [i for i, (attr, *_) in enumerate(game.ACTIONS) if attr == 0]
DATA = Path("data/interventions.npz")
TRAIN_DATA = Path("data/interventions_train.npz")
TRAIN_LAYERS = [3, 24]  # residual layers kept for consistency training: shallow and deep
TRAIN_DIRECTIONS = {"shape": ("object", ["round"], ["square"]), "size": ("object", ["big"], ["small"]),
                    "material": ("object", ["wooden"], ["metal"])}
OUT = Path("results/qwen3.5-2b")


def word_position(tok, text, word):
    """Index of the last token of `word` in the tokenised text (words can span several tokens)."""
    ids = tok(text, add_special_tokens=False).input_ids
    hits = [j - 1 for i in range(len(ids)) for j in range(i + 1, min(i + 4, len(ids)) + 1)
            if tok.decode(ids[i:j]).strip().lower() == word]
    assert hits, f"{word!r} not found in {text!r}"
    return hits[-1]


def clue_word(action):
    """The value word that appears in the clue's text ("blue" in "is not blue")."""
    attr, value, negated, _ = game.ACTIONS[action]
    return game.ATTRIBUTES[game.ATTR_NAMES[attr]][1 - value if negated else value]


@torch.no_grad()
def directions(model, tok, intro_text, pairs=DIRECTIONS):
    """Unit direction per entry of `pairs` (difference of layer-3 states on the word's token), the norm of each
    difference (the red-blue one is the unit of alpha), and the cosines between directions."""
    bos = tok.bos_token
    intro = tok(intro_text, add_special_tokens=not (bos and intro_text.startswith(bos)), return_tensors="pt").input_ids
    past = model(intro.to(model.device), use_cache=True).past_key_values
    templates = [t for _, t in game.TEMPLATES if "{anticlaim}" not in t]

    def mean_state(subject, words):
        states = []
        for word in words:
            for source in game.SOURCE_NAMES:
                for template in templates:
                    text = "\n" + template.format(source=source, claim=f"is {word}").replace("the object", f"the {subject}")
                    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
                    out = model(ids, past_key_values=copy.deepcopy(past), output_hidden_states=True)
                    states.append(out.hidden_states[LAYER][0, word_position(tok, text, word)].float())
        return torch.stack(states).mean(0)

    raw = {name: mean_state(subject, a) - mean_state(subject, b) for name, (subject, a, b) in pairs.items()}
    unit = {name: d / d.norm() for name, d in raw.items()}
    cosine = {f"{a} / {b}": float(unit[a] @ unit[b]) for i, a in enumerate(unit) for b in list(unit)[i + 1:]}
    return unit, {name: float(d.norm()) for name, d in raw.items()}, cosine


def collect(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    intro_text, questions = readout(tok, "bold", extra=EXTRA)
    unit, norms, cosine = directions(model, tok, intro_text)
    scale = norms["red vs blue"]
    print("red-blue norm", round(scale, 2), "| cosines:", {k: round(v, 2) for k, v in cosine.items()}, flush=True)
    actions = np.random.default_rng(args.seed).integers(game.N_ACTIONS, size=(args.episodes, TURNS))
    actions[:, STEERED_CLUE - 1] = [COLOUR_ACTIONS[e % len(COLOUR_ACTIONS)] for e in range(args.episodes)]
    positions = [word_position(tok, game.clue_text(a[STEERED_CLUE - 1]), clue_word(a[STEERED_CLUE - 1]))
                 for a in actions]
    conditions = [("none", 0.0)] + [(name, a) for name in DIRECTIONS for a in args.alphas]
    memory = memory_reader(model, MEMORY_LAYERS)
    pool, belief, mem = [], [], []
    for name, alpha in conditions:
        runs = []
        for a, pos in zip(actions, positions):
            steer = None if name == "none" else {
                STEERED_CLUE: (LAYER, (alpha * scale * unit[name]).to(torch.bfloat16), [pos])}
            runs.append(run_episode(model, tok, [game.clue_text(x) for x in a], intro_text, questions, steer=steer,
                                    memory=memory))
        pool.append(np.stack([r[1][:, SAVED_LAYERS] for r in runs]))
        belief.append(np.array([r[2] for r in runs], dtype=np.float32))
        mem.append(np.stack([r[4] for r in runs]))
        print(f"{name} alpha {alpha:+.1f} done", flush=True)
    DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez(DATA, pool=np.stack(pool), layers=np.array(SAVED_LAYERS), memory=np.stack(mem),
             memory_layers=np.array(MEMORY_LAYERS), belief=np.stack(belief), actions=actions,
             names=np.array([c[0] for c in conditions]), alphas=np.array([c[1] for c in conditions]),
             scale=scale, cosine=json.dumps(cosine))


def latent_readouts(states, kind, index, device="cuda"):
    """Readout, on the intervened states, of a latent model trained on natural plain clues (MLP encoder), and its
    readout R² on held-out natural episodes. kind "running": residual layer `index`, running mean of turn means;
    kind "memory": the recurrent state of saved memory layer number `index`."""
    from explain import get_model
    data_dir = "data/qwen3.5-2b" if kind == "running" else "data/qwen3.5-2b-memory"
    h, a, b, _ = jepa.load(data_dir, kind, index)
    n_train = round(len(h) * 0.8)
    mu, sd = h[:n_train].mean((0, 1)), h[:n_train].std((0, 1)) + 1e-6
    ht = torch.tensor((h - mu) / sd)
    model = get_model(ht, torch.tensor(a), torch.tensor(b), n_train,
                      Path(f"{data_dir}/latent_{kind}_layer{index}_mlp.pt"), kind, "mlp", 3000, device)
    if kind == "running":
        states = np.cumsum(states, 2) / np.arange(1, states.shape[2] + 1)[None, None, :, None]
    with torch.no_grad():
        natural = jepa.r2(model.read(model.enc(ht[n_train:].to(device)))[:, 1:].cpu().numpy(), logit(b[n_train:, 1:]))
        read = np.stack([model.read(model.enc(torch.tensor(((x - mu) / sd).astype(np.float32)).to(device))).cpu().numpy()
                         for x in states])  # [C, N, T+1, 4]
    return read, natural


def analyse(args):
    d = np.load(DATA)
    names, alphas, belief, actions = d["names"], d["alphas"], d["belief"], d["actions"]  # belief [C, N, T+1, 5]
    layers = list(d["layers"])
    pool = d["pool"].astype(np.float32)  # [C, N, T+1, layer, d]
    readouts = {("residual", layer): latent_readouts(pool[..., i, :], "running", layer) for i, layer in enumerate(layers)}
    if "memory" in d.files:
        mem = d["memory"].astype(np.float32)
        readouts.update({("memory", int(L)): latent_readouts(mem[..., i, :], "memory", i)
                         for i, L in enumerate(d["memory_layers"])})
    read = readouts[("residual", LAYER)][0]
    llm = logit(np.clip(belief, 1e-3, 1 - 1e-3))
    base = int(np.flatnonzero(names == "none")[0])
    t = STEERED_CLUE
    d_llm = llm[:, :, t] - llm[base, :, t]  # [C, N, 5]: colour (towards red), shape, size, material, orange
    d_lat = read[:, :, t] - read[base, :, t]  # [C, N, 4]
    negated = np.array([game.ACTIONS[a][2] for a in actions[:, t - 1]])
    blue_word = np.array([clue_word(a) == "blue" for a in actions[:, t - 1]])
    groups = {"affirmative": ~negated, "negated": negated}
    res = {"conditions": [f"{n} {a:+.1f}" for n, a in zip(names, alphas)], "scale": float(d["scale"]),
           "cosine_between_directions": json.loads(str(d["cosine"]))}
    for g, m in groups.items():
        res[g] = {"llm_colour": d_llm[:, m, 0].mean(1).tolist(), "latent_colour": d_lat[:, m, 0].mean(1).tolist(),
                  "llm_colour_word_blue": d_llm[:, m & blue_word, 0].mean(1).tolist(),
                  "llm_colour_word_red": d_llm[:, m & ~blue_word, 0].mean(1).tolist()}
    res["llm_other_attributes"] = np.abs(d_llm[:, :, 1:4]).mean(1).tolist()
    res["latent_other_attributes"] = np.abs(d_lat[:, :, 1:4]).mean(1).tolist()
    res["llm_orange"] = d_llm[:, :, 4].mean(1).tolist()
    res["llm_size"], res["latent_size"] = d_llm[:, :, 2].mean(1).tolist(), d_lat[:, :, 2].mean(1).tolist()
    # Agreement over conditions x (phrasing group for colour, plus the three other attributes)
    x = np.concatenate([np.stack([d_llm[:, m, 0].mean(1) for m in groups.values()], 1),
                        d_llm[:, :, 1:4].mean(1)], 1).ravel()
    y = np.concatenate([np.stack([d_lat[:, m, 0].mean(1) for m in groups.values()], 1),
                        d_lat[:, :, 1:4].mean(1)], 1).ravel()
    res["agreement_r2"], res["agreement_slope"] = jepa.r2(y, x), float(np.polyfit(x, y, 1)[0])
    print(json.dumps({k: res[k] for k in ("scale", "cosine_between_directions", "agreement_r2", "agreement_slope")},
                     indent=1))
    for c, label in enumerate(res["conditions"]):
        print(f"{label:22s} colour aff LLM {res['affirmative']['llm_colour'][c]:+.2f} latent "
              f"{res['affirmative']['latent_colour'][c]:+.2f} | neg LLM {res['negated']['llm_colour'][c]:+.2f} latent "
              f"{res['negated']['latent_colour'][c]:+.2f} | orange {res['llm_orange'][c]:+.2f}")

    figstyle.apply()
    fig, axes = plt.subplots(2, 2, figsize=(figstyle.DOUBLE, 5))
    axes = axes.ravel()
    colours = dict(zip(DIRECTIONS, [figstyle.VERMILLION, figstyle.ORANGE, figstyle.PURPLE, figstyle.GREEN,
                                    figstyle.GREY]))

    def curve(ax, values_llm, values_lat=None):
        for name in DIRECTIONS:
            m = (names == name) | (names == "none")
            order = np.argsort(alphas[m])
            ax.plot(alphas[m][order], np.asarray(values_llm)[m][order], color=colours[name], marker="o",
                    label=f"{name}, LLM")
            if values_lat is not None:
                ax.plot(alphas[m][order], np.asarray(values_lat)[m][order], color=colours[name], marker="s", ls="--",
                        mfc="white", label=f"{name}, latent model")
        ax.axhline(0, color="black", lw=0.5)
        ax.set(xlabel="Strength (red-blue norms)", ylabel="Change in belief (log-odds)")

    for ax, g in zip(axes[:2], groups):
        curve(ax, res[g]["llm_colour"], res[g]["latent_colour"])
        ax.set_title(f"Red, {g} colour clue (\"is {'not ' if g == 'negated' else ''}…\")", loc="left")
    curve(axes[2], res["llm_orange"])
    axes[2].set_title("Orange vs green (outside the game), LLM", loc="left")
    curve(axes[3], res["llm_size"], res["latent_size"])
    axes[3].set_title("Big (size belief), same colour clues", loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    figstyle.panel_labels(axes)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "interventions.json").write_text(json.dumps(res, indent=1))
    figstyle.save(fig, OUT / "interventions")
    layer_sweep(names, alphas, d_llm, readouts, negated, base)


def layer_sweep(names, alphas, d_llm, readouts, negated, base):
    """Per state read by the latent model (residual stream at a layer, or the recurrent memory of a Gated DeltaNet
    layer): agreement with the LLM over conditions with |alpha| <= 1, the latent response to red-blue on negated
    clues (the LLM's flips sign), and its size response to big-small (the LLM's stays near 0). Memory of decoder
    layer L is plotted at depth L + 1, the residual stream right after that layer."""
    t = STEERED_CLUE
    moderate = np.abs(alphas) <= 1
    red = (names == "red vs blue") & moderate
    big = (names == "big vs small") & moderate

    def groups(delta):  # [C, N, k] -> [C, 5]: colour on affirmative clues, colour on negated, shape, size, material
        return np.column_stack([delta[:, ~negated, 0].mean(1), delta[:, negated, 0].mean(1),
                                delta[:, :, 1:4].mean(1)])

    llm_g = groups(d_llm)
    out = {"llm_negated_red_slope": float(np.polyfit(alphas[red], llm_g[red, 1], 1)[0]),
           "llm_big_small_size_response": float(np.abs(llm_g[big, 3]).mean())}
    for (kind, layer), (read, natural) in readouts.items():
        block = out.setdefault(kind, {"layers": [], "depth": [], "natural_readout_r2": [], "agreement_r2": [],
                                      "negated_red_slope": [], "big_small_size_response": []})
        block["layers"].append(int(layer))
        block["depth"].append(int(layer) + (1 if kind == "memory" else 0))
        block["natural_readout_r2"].append(float(natural))
        for k, v in intervention_metrics(read, names, alphas, d_llm, negated, base).items():
            block[k].append(v)
    print(json.dumps(out, indent=1))
    (OUT / "interventions_layers.json").write_text(json.dumps(out, indent=1))
    plot_layers(out)


def plot_layers(out):
    """The depth sweep figure from its JSON, so the plot can be redone without the interventions."""
    figstyle.apply()
    fig, axes = plt.subplots(1, 3, figsize=(figstyle.DOUBLE, 2.6))
    styles = {"residual": (figstyle.BLUE, "o", "anchored JEPA, residual stream"),
              "memory": (figstyle.GREEN, "D", "anchored JEPA, recurrent memory S")}
    lowest = min(min(b["agreement_r2"]) for k, b in out.items() if k in styles)
    bottom = max(-1, lowest - 0.1)
    for kind, (colour, marker, label) in styles.items():
        if kind not in out:
            continue
        b = out[kind]
        axes[0].plot(b["depth"], b["agreement_r2"], color=colour, marker=marker, label=label)
        axes[0].plot(b["depth"], b["natural_readout_r2"], color=colour, marker=marker, ls=":", mfc="white", alpha=0.7,
                     label=f"{label}, no intervention")
        axes[1].plot(b["depth"], b["negated_red_slope"], color=colour, marker=marker, label=label)
        axes[2].plot(b["depth"], b["big_small_size_response"], color=colour, marker=marker, label=label)
    clipped = [f"layer {d}: {v:.1f}" for k in styles if k in out for d, v in zip(out[k]["depth"], out[k]["agreement_r2"])
               if v < bottom]
    if clipped:  # off the axis: say where
        axes[0].text(0.35, 0.3, "below the axis: " + ", ".join(clipped), transform=axes[0].transAxes, fontsize=7)
    axes[0].axhline(0, color="black", lw=0.5)
    axes[0].set(xlabel="Depth (layer)", ylabel="R² with the LLM's belief changes", ylim=(bottom, 1))
    axes[0].set_title("Agreement under interventions", loc="left")
    axes[1].axhline(out["llm_negated_red_slope"], color=figstyle.VERMILLION, lw=1.2, ls="--", label="LLM")
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set(xlabel="Depth (layer)", ylabel="Slope (log-odds per red-blue norm)")
    axes[1].set_title("Red-blue on negated clues", loc="left")
    axes[2].axhline(out["llm_big_small_size_response"], color=figstyle.VERMILLION, lw=1.2, ls="--", label="LLM")
    axes[2].set(xlabel="Depth (layer)", ylabel="Mean |size change| (log-odds)")
    axes[2].set_title("Big-small on a colour word", loc="left")
    handles, labels = [], []
    for a in axes:
        for h, l in zip(*a.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    figstyle.panel_labels(axes)
    figstyle.save(fig, OUT / "interventions_layers")


def intervention_metrics(read, names, alphas, d_llm, negated, base):
    """A latent model's readouts on the intervened episodes against the LLM, over conditions with |alpha| <= 1:
    agreement (R² of the readout changes on the LLM's belief changes), slope of red-blue on negated colour clues,
    and the size response to big-small on a colour word."""
    t = STEERED_CLUE
    moderate = np.abs(alphas) <= 1
    red = (names == "red vs blue") & moderate
    big = (names == "big vs small") & moderate

    def groups(delta):  # [C, N, k] -> [C, 5]: colour on affirmative clues, colour on negated, shape, size, material
        return np.column_stack([delta[:, ~negated, 0].mean(1), delta[:, negated, 0].mean(1),
                                delta[:, :, 1:4].mean(1)])

    llm_g, lat_g = groups(d_llm), groups(read[:, :, t] - read[base, :, t])
    return {"agreement_r2": jepa.r2(lat_g[moderate].ravel(), llm_g[moderate].ravel()),
            "negated_red_slope": float(np.polyfit(alphas[red], lat_g[red, 1], 1)[0]),
            "big_small_size_response": float(np.abs(lat_g[big, 3]).mean())}


def collect_train(args):
    """Episodes of 8 plain clues with one intervention each at layer 3, on the value word of a random clue."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    intro_text, questions = readout(tok, "bold")
    scale = directions(model, tok, intro_text)[1]["red vs blue"]
    unit = directions(model, tok, intro_text, TRAIN_DIRECTIONS)[0]
    rng = np.random.default_rng(args.seed + 100)
    gen = torch.Generator().manual_seed(args.seed + 100)
    actions = rng.integers(game.N_ACTIONS, size=(args.train_episodes, 8))
    pool, belief, meta = [], [], []
    for e, a in enumerate(actions):
        clue = int(rng.integers(1, 9))
        action = int(a[clue - 1])
        attr = game.ACTIONS[action][0]
        if attr == 0 or rng.random() < 0.5:
            kind, d = "random", torch.randn(model.config.hidden_size, generator=gen)
            d = (d / d.norm()).to(model.device)
        else:
            kind, d = game.ATTR_NAMES[attr], unit[game.ATTR_NAMES[attr]]
        alpha = float(rng.uniform(-2, 2))
        pos = word_position(tok, game.clue_text(action), clue_word(action))
        r = run_episode(model, tok, [game.clue_text(x) for x in a], intro_text, questions,
                        steer={clue: (LAYER, (alpha * scale * d).to(torch.bfloat16), [pos])})
        pool.append(r[1][:, TRAIN_LAYERS])
        belief.append(r[2])
        meta.append((clue, kind, alpha))
        if (e + 1) % 250 == 0:
            print(f"{e + 1}/{args.train_episodes} episodes", flush=True)
    np.savez(TRAIN_DATA, pool=np.stack(pool), layers=np.array(TRAIN_LAYERS), belief=np.array(belief, dtype=np.float32),
             actions=actions, clue=np.array([m[0] for m in meta]), kind=np.array([m[1] for m in meta]),
             alpha=np.array([m[2] for m in meta]))


def consistency(args):
    """Latent models trained on natural episodes alone and with the intervened ones, at a shallow and a deep layer:
    readout on natural test episodes, and the intervention metrics on the evaluation interventions (a family never
    seen in training)."""
    from explain import get_model
    tr_data, ev = np.load(TRAIN_DATA), np.load(DATA)
    names, alphas, actions = ev["names"], ev["alphas"], ev["actions"]
    llm = logit(np.clip(ev["belief"], 1e-3, 1 - 1e-3))
    base = int(np.flatnonzero(names == "none")[0])
    d_llm = llm[:, :, STEERED_CLUE] - llm[base, :, STEERED_CLUE]
    negated = np.array([game.ACTIONS[a][2] for a in actions[:, STEERED_CLUE - 1]])
    running = lambda x: np.cumsum(x, -2) / np.arange(1, x.shape[-2] + 1)[:, None]
    out = {"llm": {"negated_red_slope": float(np.polyfit(alphas[(names == "red vs blue") & (np.abs(alphas) <= 1)],
                                                          d_llm[(names == "red vs blue") & (np.abs(alphas) <= 1)][:, negated, 0].mean(1), 1)[0])}}
    for li, layer in enumerate(TRAIN_LAYERS):
        h, a, b, _ = jepa.load("data/qwen3.5-2b", "running", layer)
        n_train = round(len(h) * 0.8)
        mu, sd = h[:n_train].mean((0, 1)), h[:n_train].std((0, 1)) + 1e-6
        norm = lambda x: torch.tensor(((x - mu) / sd).astype(np.float32))
        h_int = norm(running(tr_data["pool"][:, :, li].astype(np.float32)))
        a_int = torch.tensor(tr_data["actions"])
        b_int = torch.tensor(tr_data["belief"].clip(1e-3, 1 - 1e-3))
        ht, at, bt = norm(h), torch.tensor(a), torch.tensor(b)
        natural_only = get_model(ht, at, bt, n_train, Path(f"data/qwen3.5-2b/latent_running_layer{layer}_mlp.pt"),
                                 "running", "mlp", 3000, "cuda")
        path = Path(f"data/qwen3.5-2b/latent_running_layer{layer}_mlp_consistency.pt")
        if path.exists():
            with_int = jepa.LatentModel(h.shape[-1]).cuda()
            with_int.load_state_dict(torch.load(path))
            with_int.eval()
        else:
            with_int = jepa.train_latent(torch.cat([ht[:n_train], h_int]), torch.cat([at[:n_train], a_int]),
                                         torch.cat([bt[:n_train], b_int]), True, args.steps, "cuda", 0)
            torch.save(with_int.state_dict(), path)
        ev_states = norm(running(ev["pool"][..., list(ev["layers"]).index(layer), :].astype(np.float32)))
        for label, model in (("natural only", natural_only), ("with interventions", with_int)):
            with torch.no_grad():
                natural = jepa.r2(model.read(model.enc(ht[n_train:].cuda()))[:, 1:].cpu().numpy(), logit(b[n_train:, 1:]))
                read = np.stack([model.read(model.enc(x.cuda())).cpu().numpy() for x in ev_states])
            out[f"layer {layer}, {label}"] = {"natural_readout_r2": float(natural),
                                              **intervention_metrics(read, names, alphas, d_llm, negated, base)}
    out["llm"]["big_small_size_response"] = float(np.abs(
        d_llm[(names == "big vs small") & (np.abs(alphas) <= 1)][:, :, 2].mean(1)).mean())
    print(json.dumps(out, indent=1))

    figstyle.apply()
    fig, axes = plt.subplots(1, 4, figsize=(figstyle.DOUBLE, 2.5))
    labels = [k for k in out if k != "llm"]
    x = np.arange(len(TRAIN_LAYERS))
    for ax, (metric, title, llm_value) in zip(axes, (
            ("natural_readout_r2", "Natural readout R²", None), ("agreement_r2", "Agreement R²", None),
            ("negated_red_slope", "Negated red-blue slope", out["llm"]["negated_red_slope"]),
            ("big_small_size_response", "Size response", out["llm"]["big_small_size_response"]))):
        for i, (variant, colour, hatch) in enumerate((("natural only", "white", "////"),
                                                      ("with interventions", figstyle.BLUE, None))):
            vals = [out[f"layer {layer}, {variant}"][metric] for layer in TRAIN_LAYERS]
            ax.bar(x + (i - 0.5) * 0.38, np.maximum(vals, -1) if metric == "agreement_r2" else vals, 0.36,
                   color=colour, edgecolor=figstyle.BLUE, hatch=hatch, lw=0.8, label=f"latent model, {variant}")
        if llm_value is not None:
            ax.axhline(llm_value, color=figstyle.VERMILLION, lw=1.2, ls="--", label="LLM")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(x, [f"layer {layer}" for layer in TRAIN_LAYERS])
        ax.set_title(title, loc="left")
    axes[1].set_ylim(-1, 1)
    clipped = [f"{k.split(',')[0]}, {k.split(', ')[1]}: {v['agreement_r2']:.1f}" for k, v in out.items()
               if k != "llm" and v["agreement_r2"] < -1]
    if clipped:  # bars cut at -1: say where they end
        axes[1].text(0.97, 0.06, "below the axis:\n" + "\n".join(clipped), transform=axes[1].transAxes, fontsize=6,
                     ha="right", va="bottom")
    handles, lab = [], []
    for a in axes:
        for h_, l_ in zip(*a.get_legend_handles_labels()):
            if l_ not in lab:
                handles.append(h_)
                lab.append(l_)
    fig.legend(handles, lab, loc="outside lower center", ncol=3)
    figstyle.panel_labels(axes)
    (OUT / "consistency.json").write_text(json.dumps(out, indent=1))
    figstyle.save(fig, OUT / "consistency")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["collect", "analyse", "collect-train", "consistency"])
    p.add_argument("--episodes", type=int, default=160)
    p.add_argument("--alphas", type=float, nargs="+", default=[-2, -1, -0.5, 0.5, 1, 2])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--train-episodes", type=int, default=1500)
    p.add_argument("--steps", type=int, default=3000)
    args = p.parse_args()
    {"collect": collect, "analyse": analyse, "collect-train": collect_train, "consistency": consistency}[args.step](args)
