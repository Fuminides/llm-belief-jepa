"""A latent model whose first four axes are the LLM's beliefs, plus k free numbers: how much state is there besides
the belief?

  z = [b, u]    b: 4 numbers read out by the identity, so each is the log-odds the LLM would answer for one
                attribute (colour, shape, size, material); u: k free numbers with no readout
  ẑ_{t+1} = z_t + P(z_t, a_t)

The readout loss on encoded and on predicted latents anchors b to the LLM's measured belief, so b cannot collapse
and needs no VICReg. u is only worth keeping if it helps predict future beliefs. Left free, u grew without bound in a
first run (standard deviations in the hundreds, with rollouts diverging on test episodes, also with a VICReg variance
floor on u), so u now carries a small L2 penalty, which lets an unneeded dimension shrink to zero, and gradients are
clipped. A variant standardises u instead (batch normalisation without parameters, as a whitened encoder would), so
every free dimension keeps unit variance and only the forecast tells whether it helps. The readout being fixed also removes the rotation freedom of the usual
latent (Klindt et al. 2026): the belief axes have names.

Compared on the memory of layer 14 (the one the LLM needs most, cache_patch.py) with a dimension-matched plain
latent (4 + k numbers, trained by prediction and VICReg alone, no belief loss, read out afterwards by a ridge probe
fitted on the frozen latent: the usual JEPA recipe) and with the usual 64-number latent model, by Test 1 (forecast R² per horizon) and by the R² of the one-step change of the belief. What u holds is read
with linear probes of known variables of the clue history, from b alone and from b and u together.

  python -m latent.anchored   # results/qwen3.5-2b-memory/anchored.{json,png,pdf}
"""
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
from latent import jepa
from latent.explain import get_model
from analysis.probe_layers import logit

DATA_DIR = "data/qwen3.5-2b-memory-mid"
LAYER = 14
OUT = Path("results/qwen3.5-2b-memory")
KS = [0, 1, 2, 4, 8, 16]
SEEDS = [0, 1]


class Belief(nn.Module):
    def forward(self, z):
        return z[..., :len(game.ATTR_NAMES)]


class Standardised(nn.Module):
    """An encoder whose last k outputs are batch-normalised without parameters."""

    def __init__(self, enc, k):
        super().__init__()
        self.enc, self.norm = enc, nn.BatchNorm1d(k, affine=False)

    def forward(self, h):
        z = self.enc(h)
        k = self.norm.num_features
        u = self.norm(z[..., -k:].reshape(-1, k)).reshape(*z.shape[:-1], k)
        return torch.cat([z[..., :-k], u], -1)


class AnchoredModel(jepa.LatentModel):
    def __init__(self, d_state, k, standardise=False):
        super().__init__(d_state, d_z=len(game.ATTR_NAMES) + k)
        self.read = Belief()
        if standardise and k:
            self.enc = Standardised(self.enc, k)


def train(h, a, b, k, steps, device, seed, standardise=False, u_penalty=0.01):
    torch.manual_seed(seed)
    model = AnchoredModel(h.shape[-1], k, standardise).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for idx in jepa.batches(len(h), steps):
        hb, ab, bb = h[idx].to(device), a[idx].to(device), b[idx].to(device)
        z, preds = model.rollouts(hb, ab)
        target = z.detach()
        pred_loss = sum(F.mse_loss(p, target[:, j + 1:]) for j, p in enumerate(preds)) / len(preds)
        read_loss = jepa.logit_mse(model.read(z), bb) + sum(
            jepa.logit_mse(model.read(p), bb[:, j + 1:]) for j, p in enumerate(preds)) / len(preds)
        loss = pred_loss + read_loss
        if k and not standardise:
            loss = loss + u_penalty * (z[..., -k:] ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model.eval()


def train_plain(h, a, b, d_z, steps, device, seed):
    """The dimension-matched unsupervised baseline: a plain latent of d_z numbers trained by prediction and VICReg
    only. Its readout is a ridge regression to the belief logits fitted afterwards on the frozen latent, stored in
    the model's linear readout so that it is scored like the others."""
    model = jepa.train_latent(h, a, b, True, steps, device, seed, d_z=d_z, use_readout=False)
    with torch.no_grad():
        z = model.enc(h.to(device)).cpu().numpy()
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(z.reshape(-1, d_z), logit(b.numpy()).reshape(-1, 4))
    model.read.weight.data = torch.tensor(ridge.coef_, dtype=torch.float32, device=device)
    model.read.bias.data = torch.tensor(ridge.intercept_, dtype=torch.float32, device=device)
    return model


@torch.no_grad()
def one_step_change(model, h, a, b, device):
    """R² of the LLM's change of belief after the next clue, predicted from the encoded state and the clue."""
    z = model.enc(h.to(device))
    pred = model.read(model.step(z[:, :-1], a.to(device))).cpu().numpy()
    lo = logit(b.numpy())
    return jepa.r2(pred - lo[:, :-1], lo[:, 1:] - lo[:, :-1])


def history_variables(a):
    """Known variables of each state [E, T+1]: name -> array [E, T+1, m]."""
    E, T = a.shape
    attr, value, negated, source = np.moveaxis(np.array(game.ACTIONS)[a], -1, 0)
    counts = np.zeros((E, T + 1, 4))
    last = np.zeros((E, T + 1, 4))  # the clue just read: +1 / -1 on its attribute by the value it asserts
    since = np.full((E, T + 1, 4), T + 1.0)  # clues since the last one on each attribute
    seen = np.zeros((E, T + 1, 4, 2))
    for t in range(1, T + 1):
        counts[:, t] = counts[:, t - 1]
        since[:, t] = since[:, t - 1] + 1
        seen[:, t] = seen[:, t - 1]
        rows = np.arange(E)
        counts[rows, t, attr[:, t - 1]] += 1
        last[rows, t, attr[:, t - 1]] = np.where(value[:, t - 1] == 0, 1.0, -1.0)
        since[rows, t, attr[:, t - 1]] = 0
        seen[rows, t, attr[:, t - 1], value[:, t - 1]] = 1
    turn = np.broadcast_to(np.arange(T + 1)[None, :, None], (E, T + 1, 1)).astype(float)
    last_negated = np.zeros((E, T + 1, 1))
    last_negated[:, 1:, 0] = negated
    last_source = np.zeros((E, T + 1, 1))
    last_source[:, 1:, 0] = source
    bayes = logit(np.array([[game.bayes_belief(row[:t].tolist()) for t in range(T + 1)] for row in a]))
    return {"turn": turn, "clues per attribute": counts, "clue just read (signed, per attribute)": last,
            "clues since the last on each attribute": np.minimum(since, T + 1),
            "contradicted attributes": seen.prod(-1).sum(-1, keepdims=True),
            "clue just read is negated": last_negated, "clue just read is from Bob": last_source,
            "Bayesian log-odds": bayes}


def probe(x_train, y_train, x_test, y_test):
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(x_train, y_train)
    return jepa.r2(ridge.predict(x_test).reshape(y_test.shape), y_test)  # one-column targets come back flat


def main():
    device = "cuda"
    memory_layers = list(np.load(f"{DATA_DIR}/shard_000000.npz")["memory_layers"])
    i = memory_layers.index(LAYER)
    h, a, b, _ = jepa.load(DATA_DIR, "memory", i)
    n = round(len(h) * 0.8)
    mu, sd = h[:n].mean((0, 1)), h[:n].std((0, 1)) + 1e-6
    h, a, b = torch.tensor((h - mu) / sd), torch.tensor(a), torch.tensor(b)
    tr, te = slice(None, n), slice(n, None)
    known = history_variables(a.numpy())

    variants = {f"anchored, k={k}": (k, False) for k in KS}
    variants.update({f"anchored, k={k}, standardised u": (k, True) for k in KS if k in (2, 4, 8)})
    res = {"layer": LAYER, "train_episodes": n, "test_episodes": len(h) - n, "variants": {}}
    for name, (k, standardise) in variants.items():
        runs = []
        for seed in SEEDS:
            path = Path(f"{DATA_DIR}/anchored_layer{LAYER}_k{k}{'_standardised' if standardise else ''}_seed{seed}.pt")
            if path.exists():
                model = AnchoredModel(h.shape[-1], k, standardise).to(device)
                model.load_state_dict(torch.load(path, map_location=device))
                model.eval()
            else:
                model = train(h[tr], a[tr], b[tr], k, 3000, device, seed, standardise)
                torch.save(model.state_dict(), path)
            forecast = jepa.evaluate({"latent model": model}, {}, h[te], a[te], b[te], device)
            with torch.no_grad():
                z = model.enc(h.to(device)).cpu().numpy()
            run = {"forecast": forecast["latent model"], "readout": forecast["readout of true state"],
                   "one_step_change": one_step_change(model, h[te], a[te], b[te], device),
                   "u_std": z[n:, :, 4:].reshape(-1, k).std(0).tolist() if k else []}
            if k:  # what u adds to b about the clue history
                flat = lambda x: x.reshape(-1, x.shape[-1])
                run["probes"] = {var: {"from b": probe(flat(z[:n, :, :4]), flat(y[:n]), flat(z[n:, :, :4]), flat(y[n:])),
                                       "from b and u": probe(flat(z[:n]), flat(y[:n]), flat(z[n:]), flat(y[n:]))}
                                 for var, y in known.items()}
            runs.append(run)
            print(f"{name}, seed {seed}: forecast k=1 {run['forecast'][0]:.3f}, mean {np.mean(run['forecast']):.3f}; "
                  f"one-step change {run['one_step_change']:.3f}; u std {np.round(run['u_std'], 2).tolist()}", flush=True)
        res["variants"][name] = runs

    # the dimension-matched baseline: a plain latent of 4 + k numbers, no belief loss, probe fitted afterwards
    for k in KS:
        d = len(game.ATTR_NAMES) + k
        runs = []
        for seed in SEEDS:
            path = Path(f"{DATA_DIR}/plain_layer{LAYER}_d{d}_seed{seed}.pt")
            if path.exists():
                model = jepa.LatentModel(h.shape[-1], d_z=d).to(device)
                model.load_state_dict(torch.load(path, map_location=device))
                model.eval()
            else:
                model = train_plain(h[tr], a[tr], b[tr], d, 3000, device, seed)
                torch.save(model.state_dict(), path)
            forecast = jepa.evaluate({"latent model": model}, {}, h[te], a[te], b[te], device)
            runs.append({"forecast": forecast["latent model"], "readout": forecast["readout of true state"],
                         "one_step_change": one_step_change(model, h[te], a[te], b[te], device)})
            print(f"plain, {d} numbers, seed {seed}: forecast mean {np.mean(runs[-1]['forecast']):.3f}; "
                  f"one-step change {runs[-1]['one_step_change']:.3f}", flush=True)
        res["variants"][f"plain, {d} numbers"] = runs

    # the usual latent model on the same memory: 64 numbers, learned readout, VICReg
    usual = get_model(h, a, b, n, Path(f"{DATA_DIR}/latent_memory_layer{i}_mlp.pt"), "memory", "mlp", 3000, device)
    forecast = jepa.evaluate({"latent model": usual}, {}, h[te], a[te], b[te], device)
    res["usual latent model (64 numbers, VICReg)"] = {
        "forecast": forecast["latent model"], "readout": forecast["readout of true state"],
        "one_step_change": one_step_change(usual, h[te], a[te], b[te], device)}
    print(f"usual: forecast mean {np.mean(forecast['latent model']):.3f}; one-step change "
          f"{res['usual latent model (64 numbers, VICReg)']['one_step_change']:.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "anchored.json").write_text(json.dumps(res, indent=1))
    plot(res)


def plot(res):
    figstyle.apply()
    fig, axes = plt.subplots(1, 3, figsize=(figstyle.DOUBLE, 2.5))
    plain = [f"anchored, k={k}" for k in KS]
    mean = lambda name, key: np.mean([np.mean(r[key]) for r in res["variants"][name]])
    for ax, key, label in ((axes[0], "forecast", "mean forecast R² (horizons 1-8)"),
                           (axes[1], "one_step_change", "R² of the one-step change")):
        ax.plot(KS, [mean(v, key) for v in plain], "o-", color=figstyle.BLUE, label="anchored JEPA, L2 on u")
        std = [k for k in KS if f"anchored, k={k}, standardised u" in res["variants"]]
        ax.plot(std, [mean(f"anchored, k={k}, standardised u", key) for k in std], "s--", color=figstyle.ORANGE,
                label="anchored JEPA, standardised u")
        ax.plot(KS, [mean(f"plain, {len(game.ATTR_NAMES) + k} numbers", key) for k in KS], "^:", color=figstyle.GREY,
                label="plain JEPA, no belief loss")
        ax.set(xlabel="free dimensions k", ylabel=label)
        ax.set_xticks(KS)
    axes[0].legend(loc="center", fontsize=6.5)  # the anchored curves sit at the top, the plain one at the bottom
    ax = axes[2]
    run = res["variants"][f"anchored, k={max(KS)}"][0]
    names = list(run["probes"])
    y = np.arange(len(names))
    ax.barh(y - 0.2, [run["probes"][v]["from b"] for v in names], 0.4, color=figstyle.GREY, label="from b")
    ax.barh(y + 0.2, [run["probes"][v]["from b and u"] for v in names], 0.4, color=figstyle.BLUE, label="from b and u")
    ax.set_yticks(y, names, fontsize=6)
    ax.invert_yaxis()
    ax.set(xlabel=f"probe R² (k={max(KS)})", xlim=(0, 1))
    ax.legend(fontsize=6, loc="lower right")
    figstyle.panel_labels(axes)
    figstyle.save(fig, OUT / "anchored")


if __name__ == "__main__":
    main()
