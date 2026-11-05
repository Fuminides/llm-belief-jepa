"""Compare readout formats on a few episodes before a full collection run.

Besides the clue reactions, each episode ends with a sentence that names one value and carries no evidence. A
readout that measures belief should not move; a shift away from the named value means the answer is suppressed
because the word was just mentioned.

With a theme that has a decision (--theme trust), the decision question is checked too: its answer mass, how often
the answer agrees with the Bayes-optimal decision, and whether it follows the posterior of the decision attribute.
"""
import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import game
from collect import FORMATS, readout, run_episode
from diagnose import summarise
from probe_layers import logit

MENTION = "\n{who} has no idea whether {subject} is {value}."


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--turns", type=int, default=4)
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--formats", nargs="+", choices=FORMATS, default=FORMATS)
    p.add_argument("--theme", choices=list(game.THEMES), default=game.THEME, help="the words of the game (game.THEMES)")
    args = p.parse_args()
    game.use(args.theme)
    A = len(game.ATTR_NAMES)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    # Straight onto the device: on unified memory (GB10) a CPU copy first would hold the weights twice.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype) if device == "cuda" else torch.float32, device_map=device
    ).eval()

    rng = np.random.default_rng(0)
    actions = rng.integers(game.N_ACTIONS, size=(args.episodes, args.turns))
    bayes = np.array([[game.bayes_belief(a[:t].tolist()) for t in range(args.turns + 1)] for a in actions], dtype=np.float32)
    attr, value = rng.integers(len(game.ATTR_NAMES), size=args.episodes), rng.integers(2, size=args.episodes)
    who = "Carol" if game.THEME == "clue" else "A bystander"
    mentions = [MENTION.format(who=who, subject=game.SUBJECT, value=game.ATTRIBUTES[game.ATTR_NAMES[a]][v])
                for a, v in zip(attr, value)]
    e = np.arange(args.episodes)
    for fmt in args.formats:
        intro_text, questions = readout(tok, fmt)
        runs = [run_episode(model, tok, [game.clue_text(x) for x in a] + [m], intro_text, questions)
                for a, m in zip(actions, mentions)]
        llm = np.array([r[2] for r in runs], dtype=np.float32)  # [E, T+2, A(+1)]; the last state follows the mention
        mass = np.array([r[3] for r in runs], dtype=np.float32)
        print(fmt, "(both option orders averaged)")
        summarise(llm[:, :-1, :A], bayes, actions)
        shift = np.where(value == 0, 1, -1) * (logit(llm[e, -1, attr]) - logit(llm[e, -2, attr]))
        print(f"  shift towards a value named without evidence (log-odds, 0 = no lexical effect): "
              f"{shift.mean():.3f} ± {1.96 * shift.std() / np.sqrt(len(shift)):.3f}")
        print("  probability of the answer tokens:", mass[:, :, :A].mean().round(3), "per attribute",
              mass[:, :, :A].mean((0, 1)).round(3), flush=True)
        if game.DECISION:
            first, second = game.DECISION["answers"]
            d, k = llm[:, :-1, A], game.DECISION["attribute"]
            optimal = np.array([[game.bayes_decision(a[:t].tolist()) for t in range(args.turns + 1)] for a in actions])
            print(f"  decision: P({first}) with no clue {d[:, 0].mean():.3f}; mean {d.mean():.3f} where Bayes says "
                  f"{first} ({optimal.mean():.2f} of the states) and {d[optimal == 0].mean():.3f} where it says {second}")
            print(f"  agreement of the more probable answer with the Bayes-optimal decision: "
                  f"{((d > 0.5) == optimal).mean():.3f}; corr with the posterior of {game.ATTR_NAMES[k]}: "
                  f"{np.corrcoef(d.ravel(), bayes[:, :, k].ravel())[0, 1]:.3f}")
            print("  probability of the decision answer tokens:", mass[:, :-1, A].mean().round(3), flush=True)


if __name__ == "__main__":
    main()
