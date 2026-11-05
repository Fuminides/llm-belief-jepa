"""Planning in the latent (the working doc's test list, H3): search clue sequences for a target belief.

From the LLM's state after the first 4 clues of a test episode, find 4 more clues that bring its stated belief to a
target: the belief it actually reached after clue 8 of the same episode, so the episode's own clues 5-8 are one
exact solution. Planners, each a beam search (width 64) over the 32 clues, differing in the model of the LLM they
search with:
  latent        the latent model on the memory of layer 14 (explain.py's, data/qwen3.5-2b-memory-mid): encode the
                real state after clue 4, roll the predictor forward, decode the belief
  bayes         a perfect Bayesian: the LLM's stated log-odds after clue 4 plus log(r / (1 - r)) per clue
  observer      the implied observer of fit_bayes.py, fitted on the training episodes (the reliabilities the LLM acts
                as if each clue type had; no recency)
  greedy        the latent model with a beam of width 1
  random        4 random clues
  actual        the episode's own clues 5-8 (the reference)
Scored by the RMS error, in log-odds over the four attributes, between the LLM's belief after the plan and the target.
Without the LLM, the belief after the plan is that of a simulator, the GRU on clue ids trained on the LLM's beliefs
(--simulator gru; test R² about 0.93), which shares nothing with the latent planner but is still a model; --llm
replays every plan on the real model (GPU, about 10 minutes for 200 episodes with Qwen3.5-2B).

  python plan_latent.py                         # simulator only: results/qwen3.5-2b-memory-mid/planning.*
  python plan_latent.py --llm Qwen/Qwen3.5-2B   # the plans replayed on the LLM
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import figstyle
import game
import jepa
from fit_bayes import design, fit
from probe_layers import logit

DATA = "data/qwen3.5-2b-memory-mid"
LAYER = 14
OUT = Path("results/qwen3.5-2b-memory-mid")
START, AHEAD, BEAM = 4, 4, 64
PLANNERS = ["latent", "observer", "bayes", "greedy", "random", "actual"]


def beam_search(step, read, state, target, width):
    """Clue sequences of length AHEAD minimising |read(state after them) − target|; step(states, clue ids) -> states."""
    beams, seqs = state[None], [[]]
    for _ in range(AHEAD):
        n = len(seqs)
        if torch.is_tensor(beams):
            states = beams.repeat_interleave(game.N_ACTIONS, 0)
        else:
            states = np.repeat(beams, game.N_ACTIONS, 0)
        clues = np.tile(np.arange(game.N_ACTIONS), n)
        nxt = step(states, clues)
        err = ((read(nxt) - target) ** 2).mean(-1)
        keep = np.argsort(err)[:width]
        beams = nxt[torch.as_tensor(keep)] if torch.is_tensor(nxt) else nxt[keep]
        seqs = [seqs[k // game.N_ACTIONS] + [int(clues[k])] for k in keep]
    return seqs[0]


def additive(weights):
    """Step and read for an observer that adds a weight per clue type to the asserted value's log-odds."""
    attr, value, negated, source = np.array(game.ACTIONS).T
    sign = np.where(value == 0, 1, -1)

    def step(lo, clues):
        lo = lo.copy()
        lo[np.arange(len(clues)), attr[clues]] += sign[clues] * weights[source[clues] * 2 + negated[clues]]
        return lo
    return step, lambda lo: lo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--llm", help="replay the plans on this model (GPU)")
    p.add_argument("--steps", type=int, default=3000)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shards = [np.load(f) for f in sorted(glob.glob(f"{DATA}/shard_*.npz"))]
    li = list(shards[0]["memory_layers"]).index(LAYER)
    h = np.concatenate([s["memory"][:, :, li] for s in shards]).astype(np.float32)
    a = np.concatenate([s["actions"] for s in shards])
    b = np.concatenate([s["llm_belief"] for s in shards]).clip(1e-3, 1 - 1e-3)
    E, T = a.shape
    n_train = round(E * 0.8)
    mu, sd = h[:n_train].mean((0, 1)), h[:n_train].std((0, 1)) + 1e-6
    hs = torch.tensor((h - mu) / sd)
    model = jepa.LatentModel(h.shape[-1]).to(device)
    model.load_state_dict(torch.load(f"{DATA}/latent_memory_layer{li}_mlp.pt", map_location=device))
    model.eval()
    gru_path = Path(f"{DATA}/planning_gru.pt")
    gru = jepa.ActionGRU().to(device)
    if gru_path.exists():
        gru.load_state_dict(torch.load(gru_path, map_location=device))
    else:
        gru = jepa.train_gru(torch.tensor(a[:n_train]), torch.tensor(b[:n_train]), args.steps, device, 0)
        torch.save(gru.state_dict(), gru_path)
    gru.eval()
    lo = logit(b)
    with torch.no_grad():
        gru_r2 = jepa.r2(gru(torch.tensor(a[n_train:]).to(device)).cpu().numpy(), lo[n_train:, 1:])
    A = len(game.ATTR_NAMES)
    obs_w = fit(design(a[:n_train]), lo[:n_train, 1:])[A:A + 4]
    r = np.array([game.SOURCES[s] for s in game.SOURCE_NAMES])
    bayes_w = np.repeat(np.log(r / (1 - r)), 2)
    print(f"GRU simulator test R² {gru_r2:.3f}; observer weights {np.round(obs_w, 2)}", flush=True)

    rng = np.random.default_rng(0)
    eps = np.arange(n_train, E)[:args.episodes]
    plans = {k: [] for k in PLANNERS}
    with torch.no_grad():
        z0 = model.enc(hs[eps, START].to(device))
    for i, e in enumerate(eps):
        target = lo[e, START + AHEAD]
        latent_step = lambda zs, clues: model.step(zs, torch.as_tensor(clues, device=device))
        latent_read = lambda zs: model.read(zs).cpu().numpy()
        with torch.no_grad():
            plans["latent"].append(beam_search(latent_step, latent_read, z0[i], target, BEAM))
            plans["greedy"].append(beam_search(latent_step, latent_read, z0[i], target, 1))
        for name, w in (("observer", obs_w), ("bayes", bayes_w)):
            plans[name].append(beam_search(*additive(w), lo[e, START], target, BEAM))
        plans["random"].append(rng.integers(game.N_ACTIONS, size=AHEAD).tolist())
        plans["actual"].append(a[e, START:START + AHEAD].tolist())

    targets = lo[eps, START + AHEAD]
    res = {"episodes": len(eps), "start_turn": START, "ahead": AHEAD, "beam": BEAM, "gru_simulator_r2": gru_r2,
           "observer_weights": obs_w.tolist(), "plans": {k: v for k, v in plans.items()}, "error": {}}
    with torch.no_grad():
        for name, seqs in plans.items():
            full = np.concatenate([a[eps, :START], np.array(seqs)], 1)
            sim = gru(torch.tensor(full).to(device)).cpu().numpy()[:, -1]
            res["error"].setdefault("gru simulator", {})[name] = rms(sim, targets)
            # what each model-based planner expected
            if name in ("latent", "greedy"):
                zs = z0
                for k in range(AHEAD):
                    zs = model.step(zs, torch.tensor(np.array(seqs)[:, k]).to(device))
                res["error"].setdefault("planner's own model", {})[name] = rms(model.read(zs).cpu().numpy(), targets)
    if args.llm:
        res["error"]["LLM"] = replay(args.llm, a[eps, :START], plans, targets, device)
    print(json.dumps(res["error"], indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "planning.json").write_text(json.dumps(res, indent=1))
    plot(res)


def rms(pred, target):
    """Mean over episodes of the RMS error over attributes, with its standard error."""
    per = np.sqrt(((pred - target) ** 2).mean(-1))
    return [float(per.mean()), float(per.std() / np.sqrt(len(per)))]


def replay(name, prefixes, plans, targets, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from collect import readout, run_episode
    tok = AutoTokenizer.from_pretrained(name)
    llm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                                               device_map=device).eval()
    intro_text, questions = readout(tok, "bold")
    out = {}
    for planner, seqs in plans.items():
        final = []
        for prefix, seq in zip(prefixes, seqs):
            clues = [game.clue_text(x) for x in list(prefix) + list(seq)]
            beliefs = run_episode(llm, tok, clues, intro_text, questions)[2]
            final.append(logit(np.array(beliefs[-1])))
        out[planner] = rms(np.array(final), targets)
        print(planner, out[planner], flush=True)
    return out


def plot(res):
    figstyle.apply()
    judges = [j for j in ("gru simulator", "LLM") if j in res["error"]]
    fig, ax = plt.subplots(figsize=(figstyle.DOUBLE / 2, 2.4))
    colours = {"gru simulator": figstyle.GREY, "LLM": figstyle.BLUE}
    w = 0.8 / len(judges)
    for i, j in enumerate(judges):
        vals = np.array([res["error"][j][k] for k in PLANNERS])
        ax.bar(np.arange(len(PLANNERS)) + (i - (len(judges) - 1) / 2) * w, vals[:, 0], w * 0.9, yerr=1.96 * vals[:, 1],
               color=colours[j], label=f"belief after the plan: {j.replace('gru', 'GRU')}",
               error_kw={"lw": 0.6, "capsize": 1.5})
    ax.set_xticks(range(len(PLANNERS)), PLANNERS)
    ax.set(ylabel="RMS error to target (log-odds)")
    ax.set_title(f"Planning {res['ahead']} clues from turn {res['start_turn']}", loc="left")
    ax.legend(loc="upper left", fontsize=6)
    figstyle.save(fig, OUT / "planning")


if __name__ == "__main__":
    main()
