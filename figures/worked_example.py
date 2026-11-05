"""Two worked examples with real data, for explaining the project.

A. One test episode of plain clues (Qwen3.5-2B): the LLM's belief in red after each clue, the Bayesian belief, and
   the latent model on the recurrent memory of layer 17: its readout of the true state and its forecast from the
   state after clue 2, driven by the clue ids only. The GRU on clue ids for comparison.
B. One intervention: clue 4 is "... says the object is not blue." and the red-blue direction is added on "blue"
   at layer 3. Change in the belief in red for the LLM, the latent model on the layer-3 residual stream, and the
   latent model on the memory of layer 17.

  python -m figures.worked_example    # figure and JSON in results/qwen3.5-2b/
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from figures import figstyle
from games import game
from analysis import intervene
from latent import jepa
from analysis.probe_layers import logit

EPISODE = 1610  # a test episode: contradiction, negations and under-confidence on colour
FORECAST_FROM = 2
MEMORY = ("data/qwen3.5-2b-memory", 2)  # memory layer number 2 of the saved ones = decoder layer 17
OUT = Path("results/qwen3.5-2b")


def short(action):
    _, value, negated, source = game.ACTIONS[action]
    word = intervene.clue_word(action)
    return f"{game.SOURCE_NAMES[source]}:\n{'not ' if negated else ''}{word}"


def episode_example(device="cuda"):
    data_dir, index = MEMORY
    h, a, b, _ = jepa.load(data_dir, "memory", index)
    n_train = round(len(h) * 0.8)
    mu, sd = h[:n_train].mean((0, 1)), h[:n_train].std((0, 1)) + 1e-6
    model = jepa.LatentModel(h.shape[-1]).to(device)
    model.load_state_dict(torch.load(f"{data_dir}/latent_memory_layer{index}_mlp.pt"))
    model.eval()
    torch.manual_seed(0)
    gru = jepa.train_gru(torch.tensor(a[:n_train]), torch.tensor(b[:n_train]), 3000, device, 0)
    he = torch.tensor(((h[EPISODE:EPISODE + 1] - mu) / sd).astype(np.float32)).to(device)
    ae = torch.tensor(a[EPISODE:EPISODE + 1]).to(device)
    with torch.no_grad():
        z = model.enc(he)
        readout = torch.sigmoid(model.read(z))[0, :, 0].cpu().numpy()
        zf, forecast = z[:, FORECAST_FROM], []
        for t in range(FORECAST_FROM, ae.shape[1]):
            zf = model.step(zf, ae[:, t])
            forecast.append(torch.sigmoid(model.read(zf))[0, 0].item())
        gru_p = torch.sigmoid(gru(ae))[0, :, 0].cpu().numpy()
    bayes = np.array([game.bayes_belief(a[EPISODE, :t].tolist()) for t in range(a.shape[1] + 1)])
    return {"clues": [game.clue_text(x).strip() for x in a[EPISODE]], "short": [short(x) for x in a[EPISODE]],
            "llm_p_red": b[EPISODE, :, 0].tolist(), "bayes_p_red": bayes[:, 0].tolist(),
            "latent_readout_p_red": readout.tolist(), "latent_forecast_p_red": forecast,
            "gru_p_red": [float(b[EPISODE, 0, 0])] + gru_p.tolist(), "llm_all_attributes_end": b[EPISODE, -1].tolist()}


def intervention_example():
    d = np.load(intervene.DATA)
    names, alphas, actions = d["names"], d["alphas"], d["actions"]
    t = intervene.STEERED_CLUE
    clue = actions[:, t - 1]
    target = np.array([game.ACTIONS[x][0] == 0 and game.ACTIONS[x][2] and intervene.clue_word(x) == "blue"
                       for x in clue])  # "... is not blue."
    red = (names == "red vs blue") | (names == "none")
    order = np.argsort(alphas[red])
    llm = logit(np.clip(d["belief"][..., 0], 1e-3, 1 - 1e-3))
    base = int(np.flatnonzero(names == "none")[0])
    out = {"clue_examples": sorted({game.clue_text(x).strip() for x in clue[target]}), "episodes": int(target.sum()),
           "alpha": alphas[red][order].tolist(),
           "llm": (llm[red][:, target, t] - llm[base, target, t]).mean(1)[order].tolist()}
    pool = d["pool"].astype(np.float32)
    layer3 = intervene.latent_readouts(pool[..., list(d["layers"]).index(3), :], "running", 3)[0]
    memory = intervene.latent_readouts(d["memory"][..., MEMORY[1], :].astype(np.float32), "memory", MEMORY[1])[0]
    for label, read in (("latent, layer-3 residual", layer3), ("latent, memory of layer 17", memory)):
        out[label] = (read[red][:, target, t, 0] - read[base, target, t, 0]).mean(1)[order].tolist()
    return out


def main():
    ep, iv = episode_example(), intervention_example()
    figstyle.apply()
    fig = plt.figure(figsize=(figstyle.DOUBLE, 5.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], width_ratios=[1.1, 1])
    ax = fig.add_subplot(grid[0, :])
    turns = np.arange(len(ep["llm_p_red"]))
    ax.plot(turns, ep["llm_p_red"], color=figstyle.VERMILLION, marker="o", label="LLM")
    ax.plot(turns, ep["bayes_p_red"], color=figstyle.GREY, marker="s", ls=":", label="Bayes")
    ax.plot(turns, ep["latent_readout_p_red"], color=figstyle.GREEN, marker="D", ls="--", mfc="white",
            label="latent (memory S, layer 17): reads the state")
    ax.plot(turns[FORECAST_FROM + 1:], ep["latent_forecast_p_red"], color=figstyle.GREEN, marker="D",
            label=f"latent (memory S, layer 17): forecast from turn {FORECAST_FROM}")
    ax.plot(turns, ep["gru_p_red"], color=figstyle.BLUE, lw=1, ls="-.", label="GRU on clue ids")
    ax.axvline(FORECAST_FROM, color=figstyle.GREY, lw=0.6, ls=":")
    ax.axhline(0.5, color="black", lw=0.4)
    ax.set_xticks(turns, ["start"] + ep["short"], fontsize=7)
    ax.set(ylim=(0, 1), ylabel="P(red)")
    ax.set_title("A. One episode: belief in red after each clue", loc="left")
    legend_a = ax.get_legend_handles_labels()
    bx = fig.add_subplot(grid[1, 0])
    bx.plot(iv["alpha"], iv["llm"], color=figstyle.VERMILLION, marker="o", label="LLM")
    bx.plot(iv["alpha"], iv["latent, layer-3 residual"], color=figstyle.BLUE, marker="s", ls="--",
            label="latent (layer-3 residual)")
    bx.plot(iv["alpha"], iv["latent, memory of layer 17"], color=figstyle.GREEN, marker="D",
            label="latent (memory S, layer 17)")
    bx.axhline(0, color="black", lw=0.5)
    bx.set(xlabel="Push towards red (red-blue norms)", ylabel="Change in P(red) (log-odds)")
    bx.set_title("B. \"… is not blue.\" pushed towards red", loc="left")
    lx = fig.add_subplot(grid[1, 1])
    lx.axis("off")
    first = lx.legend(*legend_a, loc="upper left", title="A", alignment="left", fontsize=7)
    lx.add_artist(first)
    lx.legend(*bx.get_legend_handles_labels(), loc="lower left", title="B", alignment="left", fontsize=7)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "worked_example.json").write_text(json.dumps({"episode": ep, "intervention": iv}, indent=1))
    figstyle.save(fig, OUT / "worked_example")
    print(json.dumps({"episode": ep, "intervention": iv}, indent=1))


if __name__ == "__main__":
    main()
