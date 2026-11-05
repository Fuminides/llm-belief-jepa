"""One belief or several (H2), second half: the K-head anchored latent.

  z = [b^(1), ..., b^(K), u]    b^(j): 4 log-odds read from the recurrent memory S of layer l_j by its own encoder;
                                u: k free numbers from all K memories together, with a small L2 penalty
  b̂ = sum_j w_j(z) b^(j)        the stated belief, a gate (softmax over heads, per attribute) that depends on z
  ẑ_{t+1} = z_t + P(z_t, a_t)   the predictor of anchored.py

The stated-belief loss is on the gated sum (encoded and predicted latents), as in anchored.py; a per-head loss with
weight HEAD_WEIGHT keeps every head a belief about the same four attributes, so that where heads differ the
difference is in what each layer supports and not in a head drifting off. With HEAD_WEIGHT 0 the heads are free
(the collapse check: do they still converge?). Scored on the test episodes: the forecast of the stated belief
(jepa.evaluate), how far the heads disagree (RMS pairwise difference in log-odds, and against the RMS of the stated
belief), each head's R² to the stated belief and to each observer of observers.py, and the gate's mean weight per
head on the turns where the attribute is contradicted against the others.

  python heads.py --layers 1 9 14 22            # results/qwen3.5-2b-memory-all/heads_l1-9-14-22.{json,png,pdf}
  python heads.py --layers 14                    # K = 1, the reference
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import figstyle
import game
import jepa
from observers import contradicted, observers
from probe_layers import logit

A = len(game.ATTR_NAMES)


class HeadsEncoder(nn.Module):
    """h [B, T+1, K * d] -> z [B, T+1, K * A + k]."""

    def __init__(self, K, d, k, d_hidden=512):
        super().__init__()
        self.K, self.d, self.k = K, d, k
        self.heads = nn.ModuleList(jepa.mlp(d, d_hidden, A) for _ in range(K))
        self.free = jepa.mlp(K * d, d_hidden, k) if k else None

    def forward(self, h):
        parts = [head(h[..., j * self.d:(j + 1) * self.d]) for j, head in enumerate(self.heads)]
        if self.free is not None:
            parts.append(self.free(h))
        return torch.cat(parts, -1)


class Gate(nn.Module):
    """z -> the stated belief: softmax weights over the heads, per attribute, from z."""

    def __init__(self, K, d_z):
        super().__init__()
        self.K = K
        self.lin = nn.Linear(d_z, K * A)

    def weights(self, z):
        return self.lin(z).reshape(*z.shape[:-1], self.K, A).softmax(-2)

    def forward(self, z):
        heads = z[..., :self.K * A].reshape(*z.shape[:-1], self.K, A)
        return (self.weights(z) * heads).sum(-2)


class HeadsModel(jepa.LatentModel):
    def __init__(self, K, d, k):
        super().__init__(K * d, d_z=K * A + k)
        self.K, self.k = K, k
        self.enc = HeadsEncoder(K, d, k)
        self.read = Gate(K, K * A + k)

    def heads_of(self, z):
        return z[..., :self.K * A].reshape(*z.shape[:-1], self.K, A)


def train(h, a, b, K, d, k, steps, device, seed, head_weight, u_penalty=0.01):
    torch.manual_seed(seed)
    model = HeadsModel(K, d, k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for idx in jepa.batches(len(h), steps):
        hb, ab, bb = h[idx].to(device), a[idx].to(device), b[idx].to(device)
        z, preds = model.rollouts(hb, ab)
        target = z.detach()
        pred_loss = sum(F.mse_loss(p, target[:, j + 1:]) for j, p in enumerate(preds)) / len(preds)
        read_loss = jepa.logit_mse(model.read(z), bb) + sum(
            jepa.logit_mse(model.read(p), bb[:, j + 1:]) for j, p in enumerate(preds)) / len(preds)
        loss = pred_loss + read_loss
        if head_weight:
            lb = torch.logit(bb)[..., None, :].expand(-1, -1, K, -1)
            loss = loss + head_weight * F.mse_loss(model.heads_of(z), lb)
        if k:
            loss = loss + u_penalty * (z[..., -k:] ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model.eval()


def r2(pred, target):
    return float(1 - ((target - pred) ** 2).sum() / ((target - target.mean()) ** 2).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/qwen3.5-2b-memory-all")
    p.add_argument("--out", default="results/qwen3.5-2b-memory-all")
    p.add_argument("--layers", type=int, nargs="+", default=[1, 9, 14, 22])
    p.add_argument("--k", type=int, default=4, help="free numbers")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--head-weight", type=float, nargs="+", default=[0.1, 0.0])
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shards = [np.load(f) for f in sorted(glob.glob(f"{args.data}/shard_*.npz"))]
    layers = [int(l) for l in shards[0]["memory_layers"]]
    idx = [layers.index(l) for l in args.layers]
    h = np.concatenate([s["memory"][:, :, idx] for s in shards]).astype(np.float32)  # [E, T+1, K, d]
    E, T1, K, d = h.shape
    a = np.concatenate([s["actions"] for s in shards])
    b = np.concatenate([s["llm_belief"] for s in shards]).clip(1e-3, 1 - 1e-3)
    n = round(E * 0.8)
    mu, sd = h[:n].mean((0, 1)), h[:n].std((0, 1)) + 1e-6
    h = torch.tensor(((h - mu) / sd).reshape(E, T1, K * d))
    a, b = torch.tensor(a), torch.tensor(b)
    tr, te = slice(None, n), slice(n, None)
    stated = logit(b.numpy())
    obs = observers(a.numpy())
    contra = contradicted(a.numpy())[te]  # [E_te, T+1, A]
    res = {"layers": args.layers, "k": args.k, "train_episodes": n, "test_episodes": E - n, "runs": {}}
    tag = "-".join(map(str, args.layers))
    for head_weight in args.head_weight:
        runs = []
        for seed in args.seeds:
            path = Path(f"{args.data}/heads_l{tag}_k{args.k}_hw{head_weight:g}_seed{seed}.pt")
            model = HeadsModel(K, d, args.k).to(device)
            if path.exists():
                model.load_state_dict(torch.load(path, map_location=device))
                model.eval()
            else:
                model = train(h[tr], a[tr], b[tr], K, d, args.k, args.steps, device, seed, head_weight)
                torch.save(model.state_dict(), path)
            forecast = jepa.evaluate({"latent model": model}, {}, h[te], a[te], b[te], device)
            with torch.no_grad():
                z = model.enc(h[te].to(device))
                heads = model.heads_of(z).cpu().numpy()  # [E_te, T+1, K, A]
                w = model.read.weights(z).cpu().numpy()   # [E_te, T+1, K, A]
            st = stated[te]
            pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
            run = {"seed": seed, "forecast": forecast["latent model"], "readout": forecast["readout of true state"],
                   "rms_stated": float(np.sqrt((st[:, 1:] ** 2).mean())),
                   "rms_head_disagreement": float(np.sqrt(np.mean([((heads[:, 1:, i] - heads[:, 1:, j]) ** 2).mean()
                                                                   for i, j in pairs]))) if pairs else 0.0,
                   "head_r2_to_stated": [r2(heads[:, 1:, j], st[:, 1:]) for j in range(K)],
                   "head_r2_to_observer": {name: [r2(heads[:, 1:, j], o[te][:, 1:]) for j in range(K)]
                                           for name, o in obs.items()},
                   "head_best_observer": [],
                   "gate_mean": w[:, 1:].mean((0, 1, 3)).tolist(),
                   "gate_mean_contradicted": [float(w[:, 1:, j][contra[:, 1:] > 0].mean()) for j in range(K)],
                   "gate_mean_not_contradicted": [float(w[:, 1:, j][contra[:, 1:] == 0].mean()) for j in range(K)],
                   "gate_entropy": float(-(w[:, 1:] * np.log(w[:, 1:] + 1e-9)).sum(-2).mean())}
            for j in range(K):  # which observer each head is nearest, after the stated belief itself
                scores = {name: v[j] for name, v in run["head_r2_to_observer"].items()}
                run["head_best_observer"].append(max(scores, key=scores.get))
            runs.append(run)
            print(f"head weight {head_weight:g}, seed {seed}: forecast mean {np.mean(run['forecast']):.3f}; heads "
                  f"disagree by {run['rms_head_disagreement']:.3f} log-odds (stated RMS {run['rms_stated']:.2f}); head R² "
                  f"to stated {np.round(run['head_r2_to_stated'], 3).tolist()}; gate {np.round(run['gate_mean'], 2).tolist()} "
                  f"contradicted {np.round(run['gate_mean_contradicted'], 2).tolist()} not "
                  f"{np.round(run['gate_mean_not_contradicted'], 2).tolist()}", flush=True)
        res["runs"][f"head weight {head_weight:g}"] = runs
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"heads_l{tag}"
    stem.with_suffix(".json").write_text(json.dumps(res, indent=1))
    if K > 1:
        plot(res, stem)


def plot(res, stem):
    figstyle.apply()
    fig, axes = plt.subplots(1, 3, figsize=(figstyle.DOUBLE, 2.5))
    K = len(res["layers"])
    names = [f"layer {l}" for l in res["layers"]]
    colours = [figstyle.BLUE, figstyle.ORANGE, figstyle.GREEN, figstyle.PURPLE, figstyle.VERMILLION, figstyle.SKY][:K]
    for ax, (label, runs) in zip(axes[:2], res["runs"].items()):
        obs_names = list(runs[0]["head_r2_to_observer"])
        x = np.arange(len(obs_names))
        width = 0.8 / K
        for j in range(K):
            vals = np.array([np.mean([r["head_r2_to_observer"][o][j] for r in runs]) for o in obs_names])
            ax.bar(x + (j - (K - 1) / 2) * width, vals, width * 0.9, color=colours[j], label=names[j])
        ax.set_xticks(x, obs_names, rotation=45, ha="right", rotation_mode="anchor", fontsize=6)
        ax.set(ylabel="$R^2$ of the head to the observer", ylim=(0, 1))
        dis = np.mean([r["rms_head_disagreement"] for r in runs])
        ax.set_title(f"Heads, {label} (disagree by {dis:.2f} log-odds)", loc="left", fontsize=8)
    axes[0].legend(fontsize=6, loc="lower left")
    ax = axes[2]
    runs = next(iter(res["runs"].values()))
    for j in range(K):
        ax.plot([0, 1], [np.mean([r["gate_mean_not_contradicted"][j] for r in runs]),
                         np.mean([r["gate_mean_contradicted"][j] for r in runs])], marker="o", color=colours[j], label=names[j])
    ax.set_xticks([0, 1], ["attribute not\ncontradicted", "contradicted"], fontsize=7)
    ax.set(ylabel="Mean gate weight", ylim=(0, 1), xlim=(-0.3, 1.3))
    ax.set_title("The gate and contradiction", loc="left")
    figstyle.panel_labels(axes)
    figstyle.save(fig, stem)


if __name__ == "__main__":
    main()
