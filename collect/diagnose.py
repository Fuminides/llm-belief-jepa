"""Does the LLM react to clues? Summaries computed from collected shards or from in-memory arrays."""
import glob
import sys

import numpy as np

from games import game
def summarise(llm, bayes, actions):
    """llm, bayes: [E, T+1, 4]; actions: [E, T]. Prints prior bias and how beliefs shift after each clue."""
    E, T = actions.shape
    print("  no-clue belief (Bayes 0.5):", llm[:, 0].mean(0).round(3))
    print("  corr with Bayes, turn 1 / last:",
          round(np.corrcoef(llm[:, 1].ravel(), bayes[:, 1].ravel())[0, 1], 3),
          round(np.corrcoef(llm[:, -1].ravel(), bayes[:, -1].ravel())[0, 1], 3))

    # Shift of the belief on the attribute the clue is about, signed towards the asserted value.
    acts = np.array([game.ACTIONS[a] for a in actions.ravel()]).reshape(E, T, 4)
    attr, value, negated, source = (acts[..., i] for i in range(4))
    e, t = np.indices((E, T))
    shift = llm[e, t + 1, attr] - llm[e, t, attr]
    signed = np.where(value == 0, shift, -shift)
    print("  mean shift towards the asserted value:", signed.mean().round(3), "(Bayes:",
          np.where(value == 0, bayes[e, t + 1, attr] - bayes[e, t, attr], -(bayes[e, t + 1, attr] - bayes[e, t, attr])).mean().round(3), ")")
    print("  affirmative / negated:", signed[negated == 0].mean().round(3), signed[negated == 1].mean().round(3))
    print(f"  {game.SOURCE_NAMES[0]} / {game.SOURCE_NAMES[1]}:", signed[source == 0].mean().round(3),
          signed[source == 1].mean().round(3))


if __name__ == "__main__":
    shards = [np.load(f) for f in sorted(glob.glob(f"{sys.argv[1] if len(sys.argv) > 1 else 'data'}/shard_*.npz"))]
    summarise(*(np.concatenate([s[k] for s in shards]) for k in ("llm_belief", "bayes_belief", "actions")))
    if "value_mass" in shards[0].files:
        print("  probability of the two value tokens, per attribute:",
              np.concatenate([s["value_mass"] for s in shards]).mean((0, 1)).round(3))
