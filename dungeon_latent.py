"""The anchored latent model on the dungeon: the belief is what the model expects each move to do.

  z = [b, u]    b: 4 x 5 numbers, per direction the logits of the consequence answer (safe, damage, die, slide,
                wall), read out by a softmax per direction; u: k free numbers with a small L2 penalty
  ẑ_{t+1} = z_t + P(z_t, a_t)

as anchored.py on the clue game, with the dungeon's state and action. The state h_t is the recurrent memory of one
linear-attention layer at the decision point of turn t (dungeon_states.py --consequences). The action is what is
appended to the chat between two decision points: the move, and what the narration says happened (moved, bumped,
slid, hurt, the forbidden word); with --view also the four cells around the player after the move, by terrain, which
the next message shows. The readout loss is the KL divergence from the model's answers to the decoded ones, on
encoded and on predicted latents over rollouts of up to W turns; training windows of W + 1 decision points are cut
from the games, which are split 80/20.

Scored on the test games by the one-step forecast of the answers (mean KL from the model's next answers, against
keeping this turn's answers, and as R² of the log-probabilities' change) for k = 0, 1, 4, 16, and by the learning
curve of dungeon_learning.py computed from the predicted answers instead of the model's own.

  python dungeon_latent.py --model Qwen/Qwen3.5-27B --difficulty medium_hidden   # results/dungeon/latent/...
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dungeon import CONSEQUENCES, MOVES, Dungeon, step
from dungeon_learning import Learner, summarise
from dungeon_states import DATA
from jepa import batches, mlp
from anchored import probe

EVENTS = ("move", "bump", "slide", "damage", "lethal", "goal", "time", "no_lives", "word")
KS = [0, 1, 4, 16]
W = 6
OUT = Path("results/dungeon/latent")
NB = len(MOVES) * len(CONSEQUENCES)


def load(difficulty, name, layer, view):
    """Per game: memory [T, d] at the layer, answers [T, 4, 5], actions [T - 1, a] between decision points."""
    root = DATA / difficulty / "consequences" / name.replace("/", "__")
    records = json.loads((root / "games.json").read_text())
    d = Dungeon(difficulty=difficulty)
    terrains = list(d.terrain)
    games = []
    for path in sorted(glob.glob(f"{root}/shard_*.npz")):
        shard = np.load(path)
        for g, seed in enumerate(shard["seeds"]):
            T = int(shard["turns"][g])
            rec, level = records[str(seed)], d.generate(int(seed))
            acts = []
            for t in range(T - 1):
                turn = rec["log"][t]
                a = np.zeros(len(MOVES) + 1 + len(EVENTS) + (len(MOVES) * len(terrains) if view else 0), np.float32)
                a[list(MOVES).index(turn["move"]) if turn["move"] in MOVES else len(MOVES)] = 1
                for e in turn["events"]:
                    a[len(MOVES) + 1 + EVENTS.index(e)] = 1
                if view:
                    cell = tuple(rec["log"][t + 1]["cell"])
                    for m, move in enumerate(MOVES):
                        nxt = step(cell, move)
                        name_ = level[nxt] if d.inside(nxt) else "wall"
                        a[len(MOVES) + 1 + len(EVENTS) + m * len(terrains) + terrains.index(name_)] = 1
                acts.append(a)
            games.append({"seed": int(seed), "h": shard["memory"][g, :T, layer].astype(np.float32),
                          "q": shard["consequence"][g, :T], "a": np.array(acts), "record": rec})
    return games


def windows(games):
    h, a, q = [], [], []
    for g in games:
        for s in range(0, len(g["h"]) - W):
            h.append(g["h"][s:s + W + 1]), a.append(g["a"][s:s + W]), q.append(g["q"][s:s + W + 1])
    return torch.tensor(np.array(h)), torch.tensor(np.array(a)), torch.tensor(np.array(q))


class Model(nn.Module):
    def __init__(self, d_state, d_action, k, d_hidden=512):
        super().__init__()
        self.k = k
        self.enc = mlp(d_state, d_hidden, NB + k)
        self.act = nn.Linear(d_action, 32)
        self.pred = mlp(NB + k + 32, d_hidden, NB + k)

    def read(self, z):
        """log P(answer) per direction, [..., 4, 5]."""
        return F.log_softmax(z[..., :NB].reshape(*z.shape[:-1], len(MOVES), len(CONSEQUENCES)), -1)

    def step(self, z, a):
        return z + self.pred(torch.cat([z, self.act(a)], -1))

    def rollouts(self, h, a):
        z = self.enc(h)
        zhat, preds = z[:, :-1], []
        for j in range(1, a.shape[1] + 1):
            zhat = self.step(zhat[:, :a.shape[1] + 1 - j], a[:, j - 1:])
            preds.append(zhat)
        return z, preds


def kl(q, logp):
    """KL(q || p) per direction, averaged."""
    return (q * (torch.log(q.clamp_min(1e-6)) - logp)).sum(-1).mean()


def target_b(q):
    """What b should be: the answers' log-probabilities, centred per direction so that log_softmax gives them back and
    the scale of the logits is pinned (a KL loss alone leaves it free and the rollouts diverge). [..., NB]."""
    lq = torch.log(q.clamp_min(1e-4))
    return (lq - lq.mean(-1, keepdim=True)).reshape(*q.shape[:-2], NB)


def train(h, a, q, k, steps, device, seed, u_penalty=0.01):
    torch.manual_seed(seed)
    model = Model(h.shape[-1], a.shape[-1], k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for idx in batches(len(h), steps):
        hb, ab, qb = h[idx].to(device), a[idx].to(device), q[idx].to(device)
        z, preds = model.rollouts(hb, ab)
        target = z.detach()
        bq = target_b(qb)
        loss = (sum(F.mse_loss(p, target[:, j + 1:]) for j, p in enumerate(preds)) / len(preds)
                + F.mse_loss(z[..., :NB], bq)
                + sum(F.mse_loss(p[..., :NB], bq[:, j + 1:]) for j, p in enumerate(preds)) / len(preds))
        if k:
            loss = loss + u_penalty * (z[..., -k:] ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model.eval()


@torch.no_grad()
def score(model, games, mu, sd, device):
    """One-step forecast over every transition of the test games."""
    kls, keep, change, target = [], [], [], []
    for g in games:
        if len(g["h"]) < 2:
            continue
        h = torch.tensor((g["h"] - mu) / sd).to(device)
        q = torch.tensor(g["q"]).to(device)
        z = model.enc(h)
        logp = model.read(model.step(z[:-1], torch.tensor(g["a"]).to(device)))
        kls.append(float(kl(q[1:], logp)) * (len(q) - 1))
        keep.append(float(kl(q[1:], torch.log(q[:-1].clamp_min(1e-6)))) * (len(q) - 1))
        lq = torch.log(q.clamp_min(1e-6))
        change.append((logp - lq[:-1]).cpu().numpy().ravel()), target.append((lq[1:] - lq[:-1]).cpu().numpy().ravel())
    n = sum(len(g["h"]) - 1 for g in games if len(g["h"]) >= 2)
    change, target = np.concatenate(change), np.concatenate(target)
    return {"kl_next": sum(kls) / n, "kl_keep": sum(keep) / n,
            "r2_change": float(1 - ((target - change) ** 2).sum() / ((target - target.mean()) ** 2).sum())}


def walk(d, kinds, hidden, record):
    """Per turn of a game, before the move: the Bayesian learner's observations of each hidden kind so far and its
    posterior per kind, and the moves into a hidden kind as (direction, kind, seen, true answer, learner's P of it)."""
    level, learner, log, out = d.generate(record["seed"]), Learner(d, kinds), record["log"], []
    for t, turn in enumerate(log):
        cell, moves = tuple(turn["cell"]), []
        for m, move in enumerate(MOVES):
            nxt = step(cell, move)
            if d.blocks(level, nxt) or level[nxt] not in kinds:
                continue
            truth = CONSEQUENCES.index(turn["truth"][m])
            moves.append((m, level[nxt], learner.seen[level[nxt]], truth,
                          float(learner.answer(level, cell, move)[truth]) if hidden else 1.0))
        out.append({"seen": np.array([learner.seen[k] for k in kinds], np.float32),
                    "post": np.concatenate([learner.post[k] for k in kinds]).astype(np.float32), "moves": moves})
        end = tuple(log[t + 1]["cell"]) if t + 1 < len(log) else tuple(record["final"])
        nxt = step(cell, turn["move"]) if turn["move"] in MOVES else None
        learner.update(level, None if nxt is None or d.blocks(level, nxt) else nxt, end, turn["events"])
    return out


def curve_rows(walked, answers):
    """dungeon_learning's rows of the learning curve, from any [T, 4, 5] answers (the model's own or forecast)."""
    return [{"kind": kind, "seen": seen, "model": float(answers[t, m, truth]), "bayes": bayes}
            for t, w in enumerate(walked) for m, kind, seen, truth, bayes in w["moves"]]


@torch.no_grad()
def forecast_answers(model, g, mu, sd, device):
    """[T, 4, 5]: the answers the latent forecasts for each decision point from the previous one and the action
    between; the first decision point keeps the model's own answers."""
    z = model.enc(torch.tensor((g["h"] - mu) / sd).to(device))
    p = torch.exp(model.read(model.step(z[:-1], torch.tensor(g["a"]).to(device)))).cpu().numpy()
    return np.concatenate([g["q"][:1], p])


@torch.no_grad()
def learning_probes(model, train_games, test_games, mu, sd, device, k):
    """Held-out R² of the Bayesian learner's state read from b and from b and u by a ridge probe."""
    z = lambda games: np.concatenate([model.enc(torch.tensor((g["h"] - mu) / sd).to(device)).cpu().numpy()
                                      for g in games])
    ztr, zte, out = z(train_games), z(test_games), {}
    for name in ("observations of each kind", "posterior per kind"):
        key = "seen" if name.startswith("obs") else "post"
        ytr, yte = (np.concatenate([g[key] for g in games]) for games in (train_games, test_games))
        out[name] = {"from b": probe(ztr[:, :NB], ytr, zte[:, :NB], yte)}
        if k:
            out[name]["from b and u"] = probe(ztr, ytr, zte, yte)
    return out


def state_variables(d, kinds, games):
    """Per state of each game, the known variables of the dungeon state: the turn, lives left, hazards in view, the
    Bayesian learner's observations of each kind and its posterior that fire hurts (walk() must have run)."""
    fire = kinds.index("fire") if "fire" in kinds else 0
    for g in games:
        level, log, lives, rows = d.generate(g["record"]["seed"]), g["record"]["log"], d.player["lives"], []
        for t, w in enumerate(g["walk"]):
            cell = tuple(log[t]["cell"])
            in_view = sum(level[c] in kinds for c in d.visible(level, cell) if c != cell)
            rows.append([t, lives, in_view, *w["seen"], w["post"][fire * len(w["post"]) // len(kinds) + 1]])
            lives -= "damage" in log[t]["events"]
        g["vars"] = np.array(rows, np.float32)
    return ["turn", "lives left", "hazards in view"] + [f"observations of {k}" for k in kinds] + ["P(fire hurts), Bayesian learner"]


def map_stage(args, games, train_games, test_games, mu, sd, device, d, kinds, h, a, q):
    """The free numbers of the dungeon latent (k = 16): a t-SNE map of the held-out states coloured by the known
    variables, each with the R² a ridge probe of u reaches, and k-means clusters against them by NMI."""
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    from sklearn.metrics import normalized_mutual_info_score
    import figstyle
    k = 16
    model = train(h, a, q, k, args.steps, device, 0)
    names = state_variables(d, kinds, games)
    with torch.no_grad():
        z = lambda gs: np.concatenate([model.enc(torch.tensor((g["h"] - mu) / sd).to(device)).detach().cpu().numpy() for g in gs])
    utr, ute = z(train_games)[:, NB:], z(test_games)[:, NB:]
    vtr, vte = (np.concatenate([g["vars"] for g in gs]) for gs in (train_games, test_games))
    r2 = {n: float(probe(utr, vtr[:, i], ute, vte[:, i])) if vtr[:, i].std() > 0 else None for i, n in enumerate(names)}
    # the map and the clusters over every state (the encoder is fixed; the probes above stay held-out)
    u_all, v_all = np.concatenate([utr, ute]), np.concatenate([vtr, vte])
    labels = KMeans(8, n_init=10, random_state=0).fit_predict(u_all)
    nmi = {n: float(normalized_mutual_info_score(np.round(v_all[:, i], 1) if "P(" in n else v_all[:, i].astype(int), labels))
           for i, n in enumerate(names)}
    xy = TSNE(perplexity=30, init="pca", random_state=0).fit_transform(u_all)
    res = {"k": k, "states": len(u_all), "test_states": len(ute), "probe_r2_from_u": r2, "nmi_kmeans8": nmi}
    print(json.dumps(res, indent=1), flush=True)
    stem = OUT / f"{args.model.split('/')[-1]}_{args.difficulty}{'_view' if args.view else ''}_map"
    Path(f"{stem}.json").write_text(json.dumps(res, indent=1))
    figstyle.apply()
    show = ["turn", "hazards in view", "observations of fire", "observations of ice"]
    fig, axes = plt.subplots(1, len(show), figsize=(figstyle.DOUBLE, 2.3))
    for ax, n in zip(axes.ravel(), show):
        i = names.index(n)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=v_all[:, i], s=2, cmap="viridis", lw=0, rasterized=True)
        ax.set_title(f"{n} ($R^2$ {r2[n]:.2f})", fontsize=7.5, loc="left")
        ax.set(xticks=[], yticks=[])
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
    figstyle.panel_labels(axes.ravel())
    figstyle.save(fig, stem)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-27B")
    p.add_argument("--difficulty", default="medium_hidden")
    p.add_argument("--layer", type=int, default=3, help="index into the shards' layers")
    p.add_argument("--view", action="store_true", help="the action also holds the four cells around after the move")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--map", action="store_true", help="instead of the scores: the map of the free numbers (k = 16)")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    games = load(args.difficulty, args.model, args.layer, args.view)
    n = min(round(0.8 * len(games)), len(games) - 1)
    train_games, test_games = games[:n], games[n:]
    mu = np.concatenate([g["h"] for g in train_games]).mean(0)
    sd = np.concatenate([g["h"] for g in train_games]).std(0) + 1e-6
    for g in games:
        g["h_std"] = (g["h"] - mu) / sd
    d = Dungeon(difficulty=args.difficulty)
    kinds = list(d.config["level"]["density"]) + ([d.config["level"]["cue"]["terrain"]] if d.config["level"].get("cue") else [])
    hidden = d.config.get("effects") == "hidden"
    for g in games:
        g["walk"] = walk(d, kinds, hidden, g["record"])[:len(g["h"])]
        g["seen"] = np.stack([w["seen"] for w in g["walk"]])
        g["post"] = np.stack([w["post"] for w in g["walk"]])
    h, a, q = windows([{**g, "h": g["h_std"]} for g in train_games])
    print(f"{len(train_games)} training games, {len(h)} windows; {len(test_games)} test games", flush=True)
    if args.map:
        return map_stage(args, games, train_games, test_games, mu, sd, device, d, kinds, h, a, q)
    res = {"model": args.model, "difficulty": args.difficulty, "layer": args.layer, "view": args.view, "kinds": kinds,
           "learning_llm_test": summarise([r for g in test_games for r in curve_rows(g["walk"], g["q"])], kinds),
           "runs": []}
    for k in KS:
        for seed in (0, 1):
            model = train(h, a, q, k, args.steps, device, seed)
            s = score(model, test_games, mu, sd, device)
            rows = [r for g in test_games if len(g["h"]) >= 2 for r in curve_rows(g["walk"], forecast_answers(model, g, mu, sd, device))]
            pr = learning_probes(model, train_games, test_games, mu, sd, device, k)
            res["runs"].append({"k": k, "seed": seed, **s, "learning_forecast_test": summarise(rows, kinds), "probes": pr})
            print(f"k {k:2d} seed {seed}: KL to the next answers {s['kl_next']:.4f} (keeping this turn's {s['kl_keep']:.4f}), "
                  f"R² of the change {s['r2_change']:.3f}; probes " + ", ".join(f"{n}: {v}" for n, v in pr.items()), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"{args.model.split('/')[-1]}_{args.difficulty}{'_view' if args.view else ''}"
    Path(f"{stem}.json").write_text(json.dumps(res, indent=1))  # with_suffix would eat ".5-27B..."
    plot(res, stem)


def plot(res, stem, k=16):
    """The learning curve of the LLM's own answers on the test games against the one the latent forecasts (mean over
    seeds at width k) and the Bayesian learner, per hidden kind; and what the probes read of the learner's state."""
    import matplotlib.pyplot as plt
    import figstyle
    from dungeon_learning import cell_title
    figstyle.apply()
    d = Dungeon(difficulty=res["difficulty"])
    runs = [r for r in res["runs"] if r["k"] == k]
    kinds = [kd for kd in res["kinds"] if kd in res["learning_llm_test"]]
    fig, axes = plt.subplots(1, len(kinds) + 1, figsize=(figstyle.DOUBLE, 2.5), gridspec_kw={"width_ratios": [1] * len(kinds) + [1.3]})
    for ax, kind in zip(axes, kinds):
        llm = res["learning_llm_test"][kind]["by_seen"]
        xs = sorted(int(x) for x in llm)
        mean = lambda key: [np.mean([r["learning_forecast_test"][kind]["by_seen"][str(x)]["model"] for r in runs]) for x in xs]
        ax.plot(xs, [llm[str(x)]["bayes"] for x in xs], color=figstyle.GREEN, lw=1, label="Bayesian learner")
        ax.plot(xs, [llm[str(x)]["model"] for x in xs], "o-", color=figstyle.VERMILLION, ms=3, label="LLM, its own answers")
        ax.plot(xs, mean("model"), "s--", color=figstyle.BLUE, ms=3, label="anchored JEPA, forecast answers")
        ax.set(xticks=[0, 1, 2, 3], xticklabels=["0", "1", "2", "3+"], ylim=(0, 1), xlim=(-0.4, 3.4))
        cell_title(ax, d, kind)
    axes[0].set_ylabel("P(true consequence)")
    axes[len(kinds) // 2].set_xlabel("Times the kind's effect was observed so far")  # one label for the cell panels
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=6.5)
    ax = axes[-1]
    names = list(runs[0]["probes"])
    w = 0.38
    for i, (src, colour) in enumerate((("from b", figstyle.GREY), ("from b and u", figstyle.BLUE))):
        ax.bar(np.arange(len(names)) + (i - 0.5) * w, [np.mean([r["probes"][n][src] for r in runs]) for n in names], w * 0.9,
               color=colour, label=src)
    ax.set_xticks(range(len(names)), [n.replace(" of each kind", "\nof each kind").replace(" per kind", "\nper kind") for n in names], fontsize=7)
    ax.set(ylabel="Held-out $R^2$ of the probe", ylim=(0, 1), title="The Bayesian learner's state")
    ax.legend(fontsize=6, loc="upper left", frameon=False)
    figstyle.panel_labels(axes)
    figstyle.save(fig, stem)


if __name__ == "__main__":
    main()
