"""The anchored latent on paraphrased clues (the working doc's test list, H1): does a free number hold the wording?

Wording is noise the bottleneck should discard, unless the LLM's own update depends on it; with paraphrased clues it
does (a level-3 clue weighs 0.31 log-odds against 0.79 for level 0, results/qwen3.5-2b-paraphrase/). The anchored
latent of anchored.py, z = [b, u] with b the LLM's four belief log-odds, is trained on the memory of one layer of a
collection made with --paraphrase, with k = 0 and k free numbers. Reported: the forecast and the one-step change of
the belief with and without u (the gain is state beyond the belief), and linear probes, from b alone and from b and
u together, of the wording of the clue just read (its template, one-hot; its designed level; its negation cues),
next to the clue-history variables of anchored.py. A wording probe that gains from u is the measure of how much the
latent keeps the phrasing because the LLM's update needs it.

  python -m latent.anchored_paraphrase --data data/qwen3.5-2b-paraphrase-memory-mid   # needs a collection with templates:
  # collect.py --model Qwen/Qwen3.5-2B --format bold --dtype bfloat16 --episodes 2000 --paraphrase
  #            --memory-layers 5 8 9 10 12 14 --out data/qwen3.5-2b-paraphrase-memory-mid
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from games import game
from latent import jepa
from latent.anchored import AnchoredModel, history_variables, one_step_change, probe, train
from latent.explain import NEGATION


def wording_variables(a, tpl):
    """Wording of the clue just read, per state [E, T+1, m] (zeros before the first clue)."""
    E, T = a.shape
    onehot = np.zeros((E, T + 1, len(game.TEMPLATES)))
    onehot[:, 1:][np.indices((E, T))[0], np.indices((E, T))[1], tpl] = 1
    level = np.zeros((E, T + 1, 1))
    level[:, 1:, 0] = np.array([lvl for lvl, _ in game.TEMPLATES])[tpl]
    negations = np.zeros((E, T + 1, 1))
    negations[:, 1:, 0] = [[sum(w in NEGATION for w in re.findall(r"[a-z]+", game.clue_text(x, t).lower()))
                            for x, t in zip(ra, rt)] for ra, rt in zip(a, tpl)]
    return {"template of the clue just read (one-hot)": onehot, "its convolutedness level": level,
            "its negation cues": negations}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--layer", type=int, default=14)
    p.add_argument("--ks", type=int, nargs="+", default=[0, 4, 16])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--out", default="results/qwen3.5-2b-paraphrase-memory-mid")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    memory_layers = list(np.load(f"{args.data}/shard_000000.npz")["memory_layers"])
    h, a, b, tpl = jepa.load(args.data, "memory", memory_layers.index(args.layer))
    if tpl is None:
        raise SystemExit(f"{args.data} has no templates: collect it with --paraphrase")
    n = round(len(h) * 0.8)
    mu, sd = h[:n].mean((0, 1)), h[:n].std((0, 1)) + 1e-6
    h, a_t, b = torch.tensor((h - mu) / sd), torch.tensor(a), torch.tensor(b)
    tr, te = slice(None, n), slice(n, None)
    known = {**wording_variables(a, tpl), **history_variables(a)}
    flat = lambda x: x.reshape(-1, x.shape[-1])
    res = {"data": args.data, "layer": args.layer, "train_episodes": n, "test_episodes": len(h) - n, "variants": {}}
    for k in args.ks:
        runs = []
        for seed in args.seeds:
            path = Path(f"{args.data}/anchored_layer{args.layer}_k{k}_seed{seed}.pt")
            if path.exists():
                model = AnchoredModel(h.shape[-1], k).to(device)
                model.load_state_dict(torch.load(path, map_location=device))
                model.eval()
            else:
                model = train(h[tr], a_t[tr], b[tr], k, args.steps, device, seed)
                torch.save(model.state_dict(), path)
            with torch.no_grad():
                forecast = jepa.evaluate({"latent model": model}, {}, h[te], a_t[te], b[te], device)
                z = model.enc(h.to(device)).cpu().numpy()
                run = {"forecast": forecast["latent model"], "readout": forecast["readout of true state"],
                       "one_step_change": one_step_change(model, h[te], a_t[te], b[te], device)}
            if k:
                run["probes"] = {var: {"from b": probe(flat(z[:n, :, :4]), flat(y[:n]), flat(z[n:, :, :4]), flat(y[n:])),
                                       "from b and u": probe(flat(z[:n]), flat(y[:n]), flat(z[n:]), flat(y[n:]))}
                                 for var, y in known.items()}
            runs.append(run)
            print(f"k={k}, seed {seed}: forecast mean {np.mean(run['forecast']):.3f}, one-step change "
                  f"{run['one_step_change']:.3f}" + ("".join(f"\n  {v}: {r['from b']:.3f} -> {r['from b and u']:.3f}"
                                                            for v, r in run.get("probes", {}).items())), flush=True)
        res["variants"][f"k={k}"] = runs
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "anchored_paraphrase.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
