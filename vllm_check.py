"""Does vLLM read the same moves as transformers, and how fast does it think? Runs in the vllm environment
(~/micromamba/envs/vllm, vLLM 0.32); transformers' numbers are the saved dungeon games (play_dungeon.py).

  1. The first turn of the saved argmax games, without thinking: the four moves' probabilities from vLLM's top
     log-probabilities of the first answer token, against the saved ones.
  2. Thinking on the same first turns, all games at once, up to a budget: tokens per second in total.

  PATH=~/micromamba/envs/vllm/bin:$PATH gpu-solo ~/micromamba/envs/vllm/bin/python vllm_check.py --model Qwen/Qwen3.5-2B
(vLLM compiles kernels with the environment's ninja on first use, hence the PATH; gpu-solo keeps it out of MPS.)
"""
import argparse
import json
import time

import numpy as np

from dungeon import MOVES, QUESTION, Dungeon
from play_dungeon import OUT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--difficulty", default="medium")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--budget", type=int, default=2000)
    p.add_argument("--memory", type=float, default=0.5, help="share of GPU memory vLLM may take")
    args = p.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    saved = json.loads((OUT / args.difficulty / f"{args.model.replace('/', '__')}.json").read_text())
    dungeon = Dungeon(saved["config"])
    games = saved["results"]["argmax"]["games"][:args.games]
    texts = []
    for game in games:
        level = dungeon.generate(game["seed"])
        state = dungeon.start(level)
        texts.append("\n".join([dungeon.rules(), dungeon.status(state), dungeon.observe(level, level.start), QUESTION]))

    tok = AutoTokenizer.from_pretrained(args.model)
    variants = {m: {tok(s + w, add_special_tokens=False).input_ids[0] for s in ("", " ") for w in (m, m.capitalize())}
                for m in MOVES}
    llm = LLM(args.model, dtype="bfloat16", gpu_memory_utilization=args.memory, max_model_len=16384,
              enable_prefix_caching=True, seed=0)

    def prompts(thinking):
        return [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True,
                                        enable_thinking=thinking) for t in texts]

    out = llm.generate(prompts(False), SamplingParams(max_tokens=1, temperature=0, logprobs=20))
    diffs, same = [], []
    for game, o in zip(games, out):
        top = o.outputs[0].logprobs[0]  # token id -> Logprob
        p_moves = np.array([sum(np.exp(top[i].logprob) for i in ids if i in top) for ids in variants.values()])
        p_moves /= p_moves.sum()
        saved_p = np.array(game["log"][0]["p"])
        diffs.append(np.abs(p_moves - saved_p).max())
        same.append(p_moves.argmax() == saved_p.argmax())
    print(f"first move without thinking, vLLM against the saved transformers games ({len(games)} turns): "
          f"same argmax {np.mean(same):.2f}, max |dp| mean {np.mean(diffs):.4f}, max {np.max(diffs):.4f}", flush=True)

    start = time.time()
    out = llm.generate(prompts(True), SamplingParams(max_tokens=args.budget, temperature=0.6, top_p=0.95, top_k=20, seed=0))
    seconds = time.time() - start
    lengths = [len(o.outputs[0].token_ids) for o in out]
    finished = [o.outputs[0].finish_reason == "stop" for o in out]
    print(f"thinking: {len(out)} first turns, budget {args.budget}: {sum(lengths)} tokens in {seconds:.0f} s = "
          f"{sum(lengths) / seconds:.0f} tokens/s in total; {np.mean(finished):.2f} finished, "
          f"median length {np.median(lengths):.0f}", flush=True)
    print(out[0].outputs[0].text[-400:])


if __name__ == "__main__":  # vLLM starts its engine in a spawned process
    main()
