"""How many numbers should the latent have? The usual latent model with d_z = 4 ... 128, with and without VICReg.

VICReg's variance term asks every latent dimension for a standard deviation of at least 1, so a latent bigger than
the state it models must spread its spare dimensions over something: noise, or copies of what the other dimensions
hold. Without VICReg only the readout loss keeps the latent from collapsing, and an unneeded dimension may shrink.
Each model is scored by Test 1 (forecast R² per horizon) and by the R² of the one-step change of the belief, and
its test latents by their effective rank (participation ratio of the covariance eigenvalues, (Σλ)² / Σλ²), the
number of principal components that hold 95% of the variance, and the share of the variance the belief
log-odds explain linearly. Same data as anchored.py: the memory of layer 14 of Qwen3.5-2B, 80/20 split; --data
takes another collection (e.g. with fewer attributes, collect.py --attributes) and --out its results.

  python -m latent.capacity --part 0/4   # trains every fourth model (CPU: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4)
  python -m latent.capacity              # trains what is missing, scores all: results/qwen3.5-2b-memory/capacity.*
  python -m latent.capacity --data data/qwen3.5-2b-memory-2attr --out results/qwen3.5-2b-memory-2attr/capacity
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from figures import figstyle
from games import game
from latent import jepa
from latent.anchored import DATA_DIR, LAYER, OUT, one_step_change
from analysis.probe_layers import logit

D_Z = [4, 8, 16, 32, 64, 128]
SEEDS = [0, 1]


def spectrum(z, lo):
    """Effective rank, components for 95% of the variance, and the share of the variance the belief explains."""
    z = z.reshape(-1, z.shape[-1])
    lam = np.clip(np.linalg.eigvalsh(np.cov(z.T)), 0, None)[::-1]
    lo = lo.reshape(-1, lo.shape[-1])
    explained = LinearRegression().fit(lo, z).predict(lo)
    return {"participation_ratio": float(lam.sum() ** 2 / (lam ** 2).sum()),
            "components_95": int(np.searchsorted(np.cumsum(lam) / lam.sum(), 0.95) + 1),
            "belief_share": float(1 - ((z - explained) ** 2).sum() / ((z - z.mean(0)) ** 2).sum()),
            "std": z.std(0).round(3).tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--part", default=None, help="i/n: only train the models i, i+n, i+2n, ... and stop")
    p.add_argument("--data", default=DATA_DIR)
    p.add_argument("--out", default=str(OUT / "capacity"))
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    memory_layers = list(np.load(f"{args.data}/shard_000000.npz")["memory_layers"])
    h, a, b, _ = jepa.load(args.data, "memory", memory_layers.index(LAYER))
    game.use_attributes(b.shape[-1])  # the readout has one output per attribute
    n = round(len(h) * 0.8)
    mu, sd = h[:n].mean((0, 1)), h[:n].std((0, 1)) + 1e-6
    h, a, b = torch.tensor((h - mu) / sd), torch.tensor(a), torch.tensor(b)
    tr, te = slice(None, n), slice(n, None)

    runs = [(d, v, s) for d in D_Z for v in (True, False) for s in SEEDS]
    if args.part:
        i, k = map(int, args.part.split("/"))
        runs = runs[i::k]
    res = {"data": args.data, "layer": LAYER, "attributes": b.shape[-1], "train_episodes": n, "test_episodes": len(h) - n, "runs": []}
    for d, use_vicreg, seed in runs:
        path = Path(f"{args.data}/capacity_layer{LAYER}_dz{d}{'' if use_vicreg else '_novicreg'}_seed{seed}.pt")
        if path.exists():
            model = jepa.LatentModel(h.shape[-1], d_z=d).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
        else:
            model = jepa.train_latent(h[tr], a[tr], b[tr], True, 3000, device, seed, d_z=d, use_vicreg=use_vicreg)
            torch.save(model.state_dict(), path)
            print("trained", path, flush=True)
        if args.part:
            continue
        with torch.no_grad():
            forecast = jepa.evaluate({"latent model": model}, {}, h[te], a[te], b[te], device)
            z = model.enc(h[te].to(device)).cpu().numpy()
            run = {"d_z": d, "vicreg": use_vicreg, "seed": seed, "forecast": forecast["latent model"],
                   "readout": forecast["readout of true state"],
                   "one_step_change": one_step_change(model, h[te], a[te], b[te], device),
                   **spectrum(z, logit(b[te].numpy()))}
        res["runs"].append(run)
        print(f"d_z {d}, {'VICReg' if use_vicreg else 'no VICReg'}, seed {seed}: forecast mean "
              f"{np.mean(run['forecast']):.3f}, one-step change {run['one_step_change']:.3f}, effective rank "
              f"{run['participation_ratio']:.1f}, 95% in {run['components_95']}, belief share {run['belief_share']:.2f}",
              flush=True)
    if args.part:
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    plot(res, out)


def plot(res, out):
    figstyle.apply()
    fig, axes = plt.subplots(1, 3, figsize=(figstyle.DOUBLE, 2.4))
    panels = (("forecast", "mean forecast R² (horizons 1-8)"), ("one_step_change", "R² of the one-step change"),
              ("participation_ratio", "effective rank of the latent"))
    for ax, (key, label) in zip(axes, panels):
        for use_vicreg, color, name in ((True, figstyle.BLUE, "VICReg"), (False, figstyle.VERMILLION, "no VICReg")):
            runs = [r for r in res["runs"] if r["vicreg"] == use_vicreg]
            score = lambda r: float(np.mean(r[key]))
            # a rollout that diverged scores R² far below 0: drawn as × on the axis, left out of the mean
            ok = [r for r in runs if key == "participation_ratio" or score(r) > 0]
            ax.scatter([r["d_z"] for r in ok], [score(r) for r in ok], s=8, color=color, alpha=0.5, lw=0)
            ax.scatter([r["d_z"] for r in runs if r not in ok], [0] * (len(runs) - len(ok)), s=14, marker="x",
                       color=color, clip_on=False, zorder=3)
            ax.plot(D_Z, [np.mean([score(r) for r in ok if r["d_z"] == d] or [np.nan]) for d in D_Z], "o-",
                    color=color, ms=3, label=name)
        if key != "participation_ratio":
            ax.set_ylim(0, 1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(D_Z, [str(d) for d in D_Z])
        ax.set(xlabel="latent size d_z", ylabel=label)
    axes[2].plot(D_Z, D_Z, ls="--", color=figstyle.GREY, lw=1, label="d_z")
    axes[2].set_yscale("log", base=2)
    axes[0].legend()
    axes[2].legend()
    figstyle.panel_labels(axes)
    figstyle.save(fig, out)


if __name__ == "__main__":
    main()
