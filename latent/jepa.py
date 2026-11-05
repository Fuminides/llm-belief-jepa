"""Latent world model of the LLM's belief updates (first version) and Test 1, forecasting.

  z_t = E(h_t)               encoder on one layer's state
  ẑ_{t+1} = z_t + P(z_t, a_t)  predictor, conditioned on the clue
  b̂_t = R(z_t)               readout of the LLM's belief (4 logits)

Loss: multi-step latent prediction against the stop-gradient encoded states, the readout on encoded and on
predicted latents (squared error on the LLM's belief logits, the quantity Test 1 scores; cross-entropy on
probabilities left the readout well below a linear probe), and VICReg variance/covariance terms against collapse. Test 1: from z_t and the next k clues, predict the LLM's belief at t+k (R² on logits, per k).
Baselines: a GRU over clue ids only (no activations; the LLM is deterministic given them, so it is the bar to
beat on data efficiency; with paraphrased clues, a second GRU also sees each clue's template), the same model
without the action input (RET-style), the readout of the true state
E(h_{t+k}), which needs the future activations, and a linear (ridge) probe of the true state as its reference.
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import RidgeCV
from torch import nn

from figures import figstyle
from games import game
from analysis.probe_layers import logit


def mlp(d_in, d_hidden, d_out):
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_hidden), nn.GELU(),
                         nn.Linear(d_hidden, d_out))


class TurnAttention(nn.Module):
    """Encoder over the turn states so far: each turn attends to itself and the earlier turns (causal)."""

    def __init__(self, d_state, d_z, d_hidden, d_model=256):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_state, d_model), nn.LayerNorm(d_model))
        self.q, self.k, self.v = (nn.Linear(d_model, d_model) for _ in range(3))
        self.out = mlp(2 * d_model, d_hidden, d_z)

    def forward(self, h):  # [B, T+1, d_state] -> [B, T+1, d_z]
        x = self.proj(h)
        scores = self.q(x) @ self.k(x).transpose(1, 2) / x.shape[-1] ** 0.5
        causal = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device).tril()
        pooled = scores.masked_fill(~causal, float("-inf")).softmax(-1) @ self.v(x)
        return self.out(torch.cat([x, pooled], -1))


class LatentModel(nn.Module):
    def __init__(self, d_state, d_z=64, d_hidden=512, use_action=True, encoder="mlp"):
        super().__init__()
        self.use_action = use_action
        self.enc = TurnAttention(d_state, d_z, d_hidden) if encoder == "attention" else mlp(d_state, d_hidden, d_z)
        self.act = nn.Embedding(game.N_ACTIONS, 32)
        self.pred = mlp(d_z + 32, d_hidden, d_z)
        self.read = nn.Linear(d_z, len(game.ATTR_NAMES))

    def step(self, z, a):
        e = self.act(a) * self.use_action
        return z + self.pred(torch.cat([z, e], -1))

    def rollouts(self, h, a):
        """z [B, T+1, D] and, per horizon k = 1..T, predictions of z_{t+k} from z_t for t = 0..T-k."""
        z = self.enc(h)
        T = a.shape[1]
        zhat, preds = z[:, :-1], []
        for k in range(1, T + 1):
            zhat = self.step(zhat[:, :T + 1 - k], a[:, k - 1:])
            preds.append(zhat)
        return z, preds


class ActionGRU(nn.Module):
    """Predicts the LLM's belief after each clue from the clue ids alone."""

    def __init__(self, n_ids=game.N_ACTIONS, d_hidden=128):
        super().__init__()
        self.act = nn.Embedding(n_ids, 32)
        self.gru = nn.GRU(32, d_hidden, batch_first=True)
        self.read = nn.Linear(d_hidden, len(game.ATTR_NAMES))

    def forward(self, a):  # [B, T] -> belief logits after each clue [B, T, 4]
        return self.read(self.gru(self.act(a))[0])


def vicreg(z):
    z = z.reshape(-1, z.shape[-1])
    z = z - z.mean(0)
    var = F.relu(1 - torch.sqrt(z.var(0) + 1e-4)).mean()
    cov = (z.T @ z) / (len(z) - 1)
    off = cov - torch.diag(torch.diag(cov))
    return var, (off ** 2).sum() / z.shape[1]


def logit_mse(logits, p):
    return F.mse_loss(logits, torch.logit(p))


def r2(pred, target):
    return float(1 - ((target - pred) ** 2).sum() / ((target - target.mean()) ** 2).sum())


def batches(n, steps):
    """The same number of gradient steps whatever the training-set size, so data efficiency is compared fairly."""
    for _ in range(steps):
        yield torch.randint(n, (min(128, n),))


def train_latent(h, a, b, use_action, steps, device, seed, encoder="mlp", d_z=64, use_vicreg=True, use_readout=True):
    """use_readout=False trains by prediction and VICReg alone (the plain JEPA), leaving the readout untrained."""
    torch.manual_seed(seed)
    model = LatentModel(h.shape[-1], d_z=d_z, use_action=use_action, encoder=encoder).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for idx in batches(len(h), steps):
        hb, ab, bb = h[idx].to(device), a[idx].to(device), b[idx].to(device)
        z, preds = model.rollouts(hb, ab)
        target = z.detach()
        pred_loss = sum(F.mse_loss(p, target[:, k + 1:]) for k, p in enumerate(preds)) / len(preds)
        read_loss = (logit_mse(model.read(z), bb) + sum(
            logit_mse(model.read(p), bb[:, k + 1:]) for k, p in enumerate(preds)) / len(preds)) if use_readout else 0
        var, cov = vicreg(z)
        loss = pred_loss + read_loss + (var + 0.04 * cov if use_vicreg else 0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


def train_gru(a, b, steps, device, seed, n_ids=game.N_ACTIONS):
    torch.manual_seed(seed)
    model = ActionGRU(n_ids).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for idx in batches(len(a), steps):
        loss = logit_mse(model(a[idx].to(device)), b[idx, 1:].to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


@torch.no_grad()
def evaluate(models, grus, h, a, b, device):
    """grus: {name: (model, its input ids for these episodes)}."""
    """R² of the predicted belief logits per horizon k, pooled over start turns, attributes and episodes."""
    h, a = h.to(device), a.to(device)
    target = logit(b.numpy())
    T = a.shape[1]
    res = {}
    for name, model in models.items():
        z, preds = model.rollouts(h, a)
        res[name] = [r2(model.read(p).cpu().numpy(), target[:, k + 1:]) for k, p in enumerate(preds)]
        if name == "latent model":
            res["readout of true state"] = r2(model.read(z[:, 1:]).cpu().numpy(), target[:, 1:])
    # Horizon k targets turns k..T in both cases. A GRU always sees the whole clue history, so its forecast of
    # turn t does not depend on the start turn.
    for name, (gru, ids) in grus.items():
        g = gru(ids.to(device)).cpu().numpy()
        res[name] = [r2(g[:, k - 1:], target[:, k:]) for k in range(1, T + 1)]
    return res


@torch.no_grad()
def forecasts(models, grus, h, a, b, device):
    """Like evaluate, but returns the predictions: name -> list over horizons k of (prediction, target) arrays
    [E, T - k + 1, 4], so that R² can be recomputed on resampled episodes."""
    h, a = h.to(device), a.to(device)
    target = logit(b.numpy())
    T = a.shape[1]
    out = {}
    for name, model in models.items():
        z, preds = model.rollouts(h, a)
        out[name] = [(model.read(p).cpu().numpy(), target[:, k + 1:]) for k, p in enumerate(preds)]
        if name == "latent model":
            out["readout of true state"] = [(model.read(z[:, 1:]).cpu().numpy(), target[:, 1:])]
    for name, (gru, ids) in grus.items():
        g = gru(ids.to(device)).cpu().numpy()
        out[name] = [(g[:, k - 1:], target[:, k:]) for k in range(1, T + 1)]
    return out


def bootstrap_difference(runs, n_boot=1000, seed=0):
    """runs: per seed, the forecasts of the latent model and of the GRU. The 95% interval, over resampled test
    episodes, of R²(latent) - R²(GRU) per horizon and of its mean over horizons; each resample is scored with
    every seed's models and the seeds averaged, so the interval carries the sampling noise of the test set."""
    rng = np.random.default_rng(seed)
    E = runs[0]["latent model"][0][0].shape[0]
    K = len(runs[0]["latent model"])
    diffs = np.zeros((n_boot, K))
    for i in range(n_boot):
        idx = rng.integers(0, E, E)
        for run in runs:
            for k in range(K):
                (pl, tl), (pg, tg) = run["latent model"][k], run["GRU on clue ids"][k]
                diffs[i, k] += (r2(pl[idx], tl[idx]) - r2(pg[idx], tg[idx])) / len(runs)
    lo, hi = np.percentile(diffs, [2.5, 97.5], axis=0)
    mlo, mhi = np.percentile(diffs.mean(1), [2.5, 97.5])
    return {"per_horizon_ci95": np.stack([lo, hi], 1).tolist(), "mean_over_horizons_ci95": [float(mlo), float(mhi)]}


def load(data_dir, key, layer):
    """key: acts (last token), pool (turn mean), running (mean of the turn means so far, intro included) or memory
    (projected recurrent state of linear-attention layers; `layer` then indexes the saved memory layers)."""
    shards = [np.load(f) for f in sorted(glob.glob(f"{data_dir}/shard_*.npz"))]
    h = np.concatenate([s["pool" if key == "running" else key][:, :, layer] for s in shards]).astype(np.float32)
    if key == "running":
        h = (np.cumsum(h, 1) / np.arange(1, h.shape[1] + 1)[None, :, None]).astype(np.float32)
    a = np.concatenate([s["actions"] for s in shards])
    b = np.concatenate([s["llm_belief"] for s in shards]).clip(1e-3, 1 - 1e-3)
    templates = np.concatenate([s["templates"] for s in shards]) if "templates" in shards[0].files else None
    return h, a, b, templates


def plot(all_res, stem):
    figstyle.apply()
    fig, axes = plt.subplots(1, 2, figsize=(figstyle.DOUBLE, 2.6))
    largest = all_res[-1]
    styles = {"latent model": (figstyle.BLUE, "o", "-"), "latent model, no action": (figstyle.SKY, "^", "--"),
              "GRU on clue ids": (figstyle.VERMILLION, "s", "-"),
              "GRU on clue ids + phrasing": (figstyle.ORANGE, "D", "--")}
    styles = {k: v for k, v in styles.items() if k in largest}
    ks = np.arange(1, len(largest["GRU on clue ids"]) + 1)
    sd = lambda r, name: np.array(r.get("sd over seeds", {}).get(name, np.zeros(len(r[name]))))
    for name, (colour, marker, ls) in styles.items():
        axes[0].plot(ks, largest[name], color=colour, marker=marker, ls=ls, label=name)
        axes[0].fill_between(ks, np.array(largest[name]) - sd(largest, name), np.array(largest[name]) + sd(largest, name),
                             color=colour, alpha=0.2, lw=0)
    axes[0].axhline(largest["readout of true state"], color=figstyle.GREY, ls=":", lw=1,
                    label="readout of true state")
    axes[0].axhline(largest["linear probe of true state"], color=figstyle.GREY, ls="-.", lw=1,
                    label="linear probe of true state")
    axes[0].set(xlabel="Horizon k (clues ahead)", ylabel="Test R² (belief logits)")
    axes[0].set_title(f"Forecast, {largest['train_episodes']} training episodes", loc="left")
    n = [r["train_episodes"] for r in all_res]
    for name, (colour, marker, ls) in styles.items():
        means = np.array([np.mean(r[name]) for r in all_res])
        spread = np.array([np.mean(sd(r, name)) for r in all_res])
        axes[1].plot(n, means, color=colour, marker=marker, ls=ls, label=name)
        axes[1].fill_between(n, means - spread, means + spread, color=colour, alpha=0.2, lw=0)
    axes[1].set(xscale="log", xlabel="Training episodes", ylabel="Mean test R² over horizons")
    axes[1].set_xticks(n, n)
    axes[1].minorticks_off()
    axes[1].set_title("Data efficiency", loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    figstyle.panel_labels(axes)
    figstyle.save(fig, stem)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--states", choices=["acts", "pool", "running", "memory"], default="acts")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--encoder", choices=["mlp", "attention"], default="mlp",
                   help="attention: causal attention over the turn states (use with --states pool)")
    p.add_argument("--train-episodes", type=int, nargs="+", default=[100, 200, 400, 800, 1600])
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="*", help="train every model with each seed; the figure and the JSON "
                   "then give the mean, the spread over seeds and a bootstrap of the margin over the GRU")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    h, a, b, templates = load(args.data, args.states, args.layer)
    n_test = round(len(h) * args.test_frac)
    tr, te = slice(0, len(h) - n_test), slice(len(h) - n_test, None)
    mu, sd = h[tr].mean((0, 1)), h[tr].std((0, 1)) + 1e-6
    h = torch.tensor((h - mu) / sd)
    a, b = torch.tensor(a), torch.tensor(b)
    with_phrasing = None if templates is None else a * len(game.TEMPLATES) + torch.tensor(templates)

    seeds = args.seeds or [args.seed]
    all_res = []
    for n in args.train_episodes:
        n = min(n, tr.stop)
        sub = slice(0, n)
        runs = []
        for seed in seeds:
            models = {"latent model": train_latent(h[sub], a[sub], b[sub], True, args.steps, device, seed,
                                                   args.encoder),
                      "latent model, no action": train_latent(h[sub], a[sub], b[sub], False, args.steps, device,
                                                              seed, args.encoder)}
            grus = {"GRU on clue ids": (train_gru(a[sub], b[sub], args.steps, device, seed), a[te])}
            if with_phrasing is not None:
                grus["GRU on clue ids + phrasing"] = (train_gru(with_phrasing[sub], b[sub], args.steps, device, seed,
                                                                n_ids=game.N_ACTIONS * len(game.TEMPLATES)),
                                                      with_phrasing[te])
            runs.append(forecasts(models, grus, h[te], a[te], b[te], device))
        # Mean over seeds of the pooled R² per horizon (a single value for the readout of the true state), the
        # per-seed values, and the bootstrap of the margin over the GRU.
        per_seed = {name: [[r2(p, t) for p, t in run[name]] for run in runs] for name in runs[0]}
        res = {name: (np.mean(v, 0).tolist() if len(v[0]) > 1 else float(np.mean(v))) for name, v in per_seed.items()}
        res["sd over seeds"] = {name: (np.std(v, 0).tolist() if len(v[0]) > 1 else float(np.std(v)))
                                for name, v in per_seed.items()}
        res["per seed"] = per_seed
        res["seeds"] = seeds
        res["latent minus GRU"] = bootstrap_difference(runs)
        flat = lambda x: x[:, 1:].reshape(-1, x.shape[-1]).numpy()
        ridge = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(flat(h[sub]), logit(flat(b[sub])))
        res["linear probe of true state"] = r2(ridge.predict(flat(h[te])), logit(flat(b[te])))
        res["train_episodes"] = n
        all_res.append(res)
        print(n, {k: np.round(v, 3).tolist() if isinstance(v, list) else round(v, 3)
                  for k, v in res.items() if isinstance(v, (list, float))},
              "latent - GRU, mean over horizons, 95%:", np.round(res["latent minus GRU"]["mean_over_horizons_ci95"], 3),
              flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / (f"jepa_{args.states}_layer{args.layer}" + ("_attention" if args.encoder == "attention" else ""))
    Path(f"{stem}.json").write_text(json.dumps({"args": vars(args), "results": all_res}, indent=1))
    plot(all_res, stem)


if __name__ == "__main__":
    main()
