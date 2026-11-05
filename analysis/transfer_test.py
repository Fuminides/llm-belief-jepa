"""Transfer of the implied observer between themes (H3): fit the clue weights on one game, predict the other.

The implied observer (fit_bayes.py) is additive in log-odds: an intercept per attribute plus a weight per clue type
(source x phrasing). The intercepts are about the words of each game, so they are refitted on the target; the
clue weights are carried over as they are, and also with the two sources swapped (in the clue game the first
source, Alice, is the reliable one; in "trust" the first, the audit tool; in "trust_reversed" the second, the user,
so a model that keys on the stated reliability transfers with the swap and one that keys on the role does not).
R² is compared with a full refit on the target and with the true Bayesian observer. The same for the recency fit.
With a theme that has a decision, the decision's log-odds are also regressed on the signed clue counts of all the
attributes, to see how much the three facts irrelevant to the stated rule move it.

  CLUE_THEME=trust python -m analysis.transfer_test --source data/qwen3.5-2b-memory --target data/qwen3.5-2b-trust \
      --out results/qwen3.5-2b-trust
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

from games import game
from analysis.fit_bayes import CLUE_TYPES, design, design_recency, fit, r2
from analysis.probe_layers import logit


def load(path):
    shards = [np.load(f) for f in sorted(glob.glob(f"{path}/shard_*.npz"))]
    keys = ["llm_belief", "bayes_belief", "actions"] + [k for k in ("llm_decision", "bayes_decision", "prior") if k in shards[0]]
    out = {k: np.concatenate([s[k] for s in shards]) for k in keys}
    if "prior" in out and np.ptp(out["prior"]) == 0:  # a constant stated prior is in the intercepts
        del out["prior"]
    return out


def swap_sources(x, A, per_source):
    """x with the columns of the two sources exchanged (per_source columns each, after the A intercepts, repeated
    for every recency block)."""
    x = x.copy()
    k = 2 * per_source
    for start in range(A, x.shape[-1], k):
        block = x[..., start:start + k].copy()
        x[..., start:start + per_source], x[..., start + per_source:start + k] = block[..., per_source:], block[..., :per_source]
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="shards fitted on (the clue game)")
    p.add_argument("--target", required=True, help="shards predicted (the trust game)")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    src, tgt = load(args.source), load(args.target)
    A = src["llm_belief"].shape[-1]
    assert tgt["llm_belief"].shape[-1] == A
    game.use_attributes(A)
    res = {"source": args.source, "target": args.target, "episodes": [len(src["actions"]), len(tgt["actions"])]}
    for name, make in (("additive", design), ("recency", design_recency)):
        # A per-episode stated prior (collect.py --prior random) enters the source fit as a last regressor, whose
        # coefficient is then dropped: the carried-over weights are the clue weights only.
        xs, ys = make(src["actions"], src.get("prior")), logit(src["llm_belief"][:, 1:])
        xt, yt = make(tgt["actions"], tgt.get("prior")), logit(tgt["llm_belief"][:, 1:])
        coef_full = fit(xs, ys)
        coef_s = coef_full[:-1] if "prior" in src else coef_full
        if "prior" in tgt:  # the target's stated prior gets its own weight, fitted with the intercepts
            xt = np.concatenate([xt[..., :A], xt[..., -1:], xt[..., A:-1]], -1)
        n_intercepts = A + ("prior" in tgt)
        assert xt.shape[-1] - n_intercepts == len(coef_s) - A, "source and target designs differ"
        out = {"r2_refit_on_target": float(r2(yt, xt @ fit(xt, yt))),
               "r2_true_bayes_on_target": float(r2(yt, logit(tgt["bayes_belief"][:, 1:]))),
               "r2_source_on_itself": float(r2(ys, xs @ coef_full))}
        for label, x in (("transfer", xt), ("transfer_sources_swapped", swap_sources(xt, n_intercepts, 2))):
            # Source weights, target intercepts: fit the intercepts on the residual after the carried-over weights.
            carried = x[..., n_intercepts:] @ coef_s[A:]
            intercepts = fit(x[..., :n_intercepts], yt - carried)
            out["r2_" + label] = float(r2(yt, carried + x[..., :n_intercepts] @ intercepts))
        res[name] = out
        print(name, json.dumps({k: round(v, 3) for k, v in out.items()}))
    if "llm_decision" in tgt:
        # Decision log-odds on the signed counts of every attribute's clue types (+ intercept).
        E, T = tgt["actions"].shape
        counts = design(tgt["actions"])[..., A:]  # [E, T, A, 4]: attribute a's counts in row a
        x = np.concatenate([np.ones((E, T, 1)), counts.reshape(E, T, -1)], -1)
        y = logit(tgt["llm_decision"][:, 1:])
        w = fit(x, y)
        weights = w[1:].reshape(A, len(CLUE_TYPES))
        dec = {"r2": float(r2(y, x @ w)), "intercept": float(w[0]),
               "weights_by_attribute": {a: dict(zip([t.replace("\n", " ") for t in CLUE_TYPES], weights[i].round(3).tolist()))
                                        for i, a in enumerate(game.ATTR_NAMES)},
               "mean_abs_weight_by_attribute": dict(zip(game.ATTR_NAMES, np.abs(weights).mean(1).round(3).tolist())),
               "agreement_with_bayes_decision": float(((tgt["llm_decision"] > 0.5) == tgt["bayes_decision"]).mean()),
               "p_first_answer_where_bayes_says_first": float(tgt["llm_decision"][tgt["bayes_decision"] == 1].mean()),
               "p_first_answer_where_bayes_says_second": float(tgt["llm_decision"][tgt["bayes_decision"] == 0].mean())}
        res["decision"] = dec
        print("decision", json.dumps({k: v for k, v in dec.items() if k != "weights_by_attribute"}))
        print("  weights by attribute:", json.dumps(dec["weights_by_attribute"]))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "transfer.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
