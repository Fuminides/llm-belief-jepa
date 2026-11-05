"""Generalisation tests: the latent model against the GRUs outside what they were trained on.

  horizon    train on the first 4 clues of each episode; forecast the beliefs after clues 5-8. The latent model
             starts from the true state after clue 4, the GRUs must run on longer sequences than they saw.
  templates  (paraphrased data) train on episodes without two templates (one level 2, one level 3); read the
             belief after clues phrased with them. The template-aware GRU has never seen those templates; the
             latent model reads the state they leave. Turns before any held-out clue are the in-distribution
             reference.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from figures import figstyle
from games import game
from latent import jepa
from analysis.probe_layers import logit

SPLIT = 4  # clues seen in training for the horizon test
HELD_OUT = (5, 8)  # "If you ask {source}, ..." (level 2) and "{source} denies that ..." (level 3)


def standardise(h, seen):
    """Standardise every state with the statistics of the states seen in training."""
    mu, sd = seen.mean((0, 1)), seen.std((0, 1)) + 1e-6
    return torch.tensor((h - mu) / sd)


@torch.no_grad()
def rollout_from(model, h, a, start, device):
    """Belief logits after clues start+1..T, predicted from the state after clue `start` and the clue ids."""
    z = model.enc(h.to(device))[:, start]
    out = []
    for t in range(start, a.shape[1]):
        z = model.step(z, a[:, t].to(device))
        out.append(model.read(z))
    return torch.stack(out, 1).cpu().numpy()


def horizon_test(h, a, b, tpl, n_test, steps, encoder, device):
    E = len(h)
    tr, te = np.arange(E - n_test), np.arange(E - n_test, E)
    h = standardise(h, h[tr, :SPLIT + 1])
    a, b = torch.tensor(a), torch.tensor(b)
    target = logit(b[te].numpy())
    res = {}
    short = jepa.train_latent(h[tr, :SPLIT + 1], a[tr, :SPLIT], b[tr, :SPLIT + 1], True, steps, device, 0, encoder)
    full = jepa.train_latent(h[tr], a[tr], b[tr], True, steps, device, 0, encoder)
    for name, model in (("latent model", short), ("latent model, trained on 8 clues", full)):
        pred = rollout_from(model, h[te], a[te], SPLIT, device)
        res[name] = [jepa.r2(pred[:, k], target[:, SPLIT + 1 + k]) for k in range(pred.shape[1])]
    grus = {"GRU on clue ids": (a, game.N_ACTIONS)}
    if tpl is not None:
        grus["GRU on clue ids + phrasing"] = (a * len(game.TEMPLATES) + torch.tensor(tpl),
                                              game.N_ACTIONS * len(game.TEMPLATES))
    for name, (ids, n_ids) in grus.items():
        for label, length in ((name, SPLIT), (f"{name}, trained on 8 clues", a.shape[1])):
            gru = jepa.train_gru(ids[tr, :length], b[tr, :length + 1], steps, device, 0, n_ids=n_ids)
            with torch.no_grad():
                g = gru(ids[te].to(device)).cpu().numpy()
            res[label] = [jepa.r2(g[:, t - 1], target[:, t]) for t in range(SPLIT + 1, a.shape[1] + 1)]
    return res


def templates_test(h, a, b, tpl, n_test, steps, encoder, device):
    E = len(h)
    held = np.isin(tpl, HELD_OUT)
    tr = np.flatnonzero(~held[:E - n_test].any(1))
    te = np.arange(E - n_test, E)[held[E - n_test:].any(1)]
    h = standardise(h, h[tr])
    a, b = torch.tensor(a), torch.tensor(b)
    after = np.cumsum(held[te], 1) > 0  # [episodes, T]: turn t+1 comes after a held-out clue
    target = logit(b[te, 1:].numpy())
    model = jepa.train_latent(h[tr], a[tr], b[tr], True, steps, device, 0, encoder)
    with torch.no_grad():
        preds = {"latent model (readout of state)": model.read(model.enc(h[te].to(device)))[:, 1:].cpu().numpy()}
    with_phrasing = a * len(game.TEMPLATES) + torch.tensor(tpl)
    for name, ids, n_ids in (("GRU on clue ids", a, game.N_ACTIONS),
                             ("GRU on clue ids + phrasing", with_phrasing, game.N_ACTIONS * len(game.TEMPLATES))):
        gru = jepa.train_gru(ids[tr], b[tr], steps, device, 0, n_ids=n_ids)
        with torch.no_grad():
            preds[name] = gru(ids[te].to(device)).cpu().numpy()
    res = {"train_episodes": len(tr), "test_episodes": len(te)}
    for name, pred in preds.items():
        res[name] = {"before any held-out clue": jepa.r2(pred[~after], target[~after]),
                     "after a held-out clue": jepa.r2(pred[after], target[after])}
    return res


def plot(res, stem):
    figstyle.apply()
    panels = ["horizon"] + (["templates"] if "templates" in res else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(figstyle.DOUBLE if len(panels) == 2 else 3.8, 2.8),
                             squeeze=False)
    axes = axes[0]
    styles = {"latent model": (figstyle.BLUE, "o", "latent"), "GRU on clue ids": (figstyle.VERMILLION, "s", "GRU ids"),
              "GRU on clue ids + phrasing": (figstyle.ORANGE, "D", "GRU ids + phrasing")}
    ax = axes[0]
    turns = np.arange(SPLIT + 1, SPLIT + 1 + len(res["horizon"]["latent model"]))
    for name, (colour, marker, short) in styles.items():
        if name in res["horizon"]:
            ax.plot(turns, res["horizon"][name], color=colour, marker=marker, label=f"{short}, trained on 4")
            ax.plot(turns, res["horizon"][f"{name}, trained on 8 clues"], color=colour, marker=marker, ls=":",
                    mfc="white", label=f"{short}, trained on 8")
    ax.set_xticks(turns)
    ax.set(xlabel="Turn forecast (clues beyond 4 are new)", ylabel="Test R² (belief logits)")
    ax.set_title("Longer horizon", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    if "templates" in res:
        ax = axes[1]
        t = res["templates"]
        names = [n for n in t if isinstance(t[n], dict)]
        x = np.arange(len(names))
        for i, (part, hatch) in enumerate((("before any held-out clue", None), ("after a held-out clue", "////"))):
            ax.bar(x + (i - 0.5) * 0.38, [t[n][part] for n in names], 0.36, color="white" if hatch else figstyle.BLUE,
                   edgecolor=figstyle.BLUE, hatch=hatch, lw=0.8, label=part)
        ax.axhline(0, color="black", lw=0.5)
        short = {"latent model (readout of state)": "latent\n(readout)", "GRU on clue ids": "GRU\nids",
                 "GRU on clue ids + phrasing": "GRU ids\n+ phrasing"}
        ax.set_xticks(x, [short[n] for n in names])
        ax.set(ylabel="Test R² (belief logits)")
        ax.set_title(f"Held-out templates ({t['train_episodes']} training episodes)", loc="left")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    figstyle.panel_labels(axes)
    figstyle.save(fig, stem)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--states", choices=["acts", "pool", "running", "memory"], default="running")
    p.add_argument("--encoder", choices=["mlp", "attention"], default="mlp")
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    h, a, b, tpl = jepa.load(args.data, args.states, args.layer)
    n_test = round(len(h) * args.test_frac)
    res = {"args": vars(args), "horizon": horizon_test(h, a, b, tpl, n_test, args.steps, args.encoder, device)}
    if tpl is not None:
        res["templates"] = templates_test(h, a, b, tpl, n_test, args.steps, args.encoder, device)
    print(json.dumps({k: v for k, v in res.items() if k != "args"}, indent=1))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"generalize_{args.states}_layer{args.layer}_{args.encoder}"
    Path(f"{stem}.json").write_text(json.dumps(res, indent=1))
    plot(res, stem)


if __name__ == "__main__":
    main()
