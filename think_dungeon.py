"""Whole dungeon games with thinking, played through vLLM (run in the vllm environment; see vllm_check.py).

As in play_dungeon.py, a game is one chat with one turn per move. By default past turns keep only the move, as Qwen's
template renders them, so each turn plans afresh; --keep N keeps the thinking of the last N past turns (--keep all: of
every turn), each exactly as the model wrote it and closed it, before its move. Each turn the model first thinks: Qwen's thinking sampling up to a budget of tokens, stopped
at </think>, or closed with Qwen's budget sentence when the budget runs out. Then the move is read from the first
answer token as in play_dungeon.py (the four moves' probabilities from vLLM's top 20 log-probabilities,
renormalised), and the most probable move is played. All games advance one turn at a time together, so each vLLM
call is one batch.

Prompts go to vLLM as token ids, and each turn's thinking ids are saved (data/dungeon_think/, not in git), so the
states can be read later with transformers from exactly the tokens the model saw: the prompt (prompt_ids, from the
turns' texts, the moves, the thinking and keep), then the thinking, then the closing ids.

After every turn the moves, their infos and the thinking so far go to a checkpoint (<model>.ckpt.json next to the
thinking ids). A run started again with the same arguments replays it (slides are seeded by the level, so the same
moves give the same games) and goes on from the next turn; the checkpoint is removed once the results are written.

  PATH=~/micromamba/envs/vllm/bin:$PATH gpu-solo ~/micromamba/envs/vllm/bin/python think_dungeon.py \\
      --model Qwen/Qwen3.5-9B --budget 1500 --games 50 [--keep 1]
  # writes results/dungeon/<difficulty>/think<budget>[_keep<N>]/<model>.json and data/dungeon_think/...
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from dungeon import MOVES, Dungeon
from play_dungeon import OUT, describe, stats

CLOSE = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
DATA = Path("data/dungeon_think")


def prompt_ids(tok, texts, moves, thoughts, closings, keep):
    """The chat's token ids for the turn after len(moves) moves: Qwen's template (past turns keep only the move),
    except that the last `keep` past turns (None: all) keep their thinking ids and closing ids before the move, as
    the model produced them after the generation prompt."""
    def ids(s):
        return tok(s, add_special_tokens=False).input_ids
    out, n = [], len(moves)
    for j, text in enumerate(texts):
        out += ids(f"<|im_start|>user\n{text.strip()}<|im_end|>\n<|im_start|>assistant\n")
        if j == n:
            out += ids("<think>\n")
        elif keep is None or j >= n - keep:
            out += ids("<think>\n") + thoughts[j] + closings[j] + ids(f"{moves[j]}<|im_end|>\n")
        else:
            out += ids(f"{moves[j]}<|im_end|>\n")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--difficulty", default="medium")
    p.add_argument("--games", type=int, default=50)
    p.add_argument("--seed", type=int, default=0, help="levels are seeds seed, seed+1, ... (as in play_dungeon.py)")
    p.add_argument("--budget", type=int, default=1500, help="most thinking tokens per turn")
    p.add_argument("--keep", default="0", help="past turns whose thinking stays in the chat: a number or all")
    p.add_argument("--memory", type=float, default=0.75, help="share of GPU memory vLLM may take")
    args = p.parse_args()
    keep = None if args.keep == "all" else int(args.keep)
    run = f"think{args.budget}" + (f"_keep{args.keep}" if keep != 0 else "")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt

    tok = AutoTokenizer.from_pretrained(args.model)
    end_think = tok.convert_tokens_to_ids("</think>")
    finish_ids = tok("</think>\n\n", add_special_tokens=False).input_ids
    close_ids = tok(CLOSE, add_special_tokens=False).input_ids
    variants = [{tok(s + w, add_special_tokens=False).input_ids[0] for s in ("", " ") for w in (m, m.capitalize())}
                for m in MOVES]
    dungeon = Dungeon(difficulty=args.difficulty)
    # The longest games' views take about 9k tokens; each kept thought adds up to a budget and its closing.
    most = dungeon.config["player"]["turns"] or dungeon.config["max_turns"]
    kept = most if keep is None else min(keep, most)
    llm = LLM(args.model, dtype="bfloat16", gpu_memory_utilization=args.memory,
              max_model_len=16384 + kept * (args.budget + len(close_ids) + 8),
              enable_prefix_caching=True, seed=args.seed)

    levels = [dungeon.generate(args.seed + i) for i in range(args.games)]
    games = {i: dungeon.game(level) for i, level in enumerate(levels)}
    turns = {i: next(g) for i, g in games.items()}
    texts = {i: [] for i in games}
    moves = {i: [] for i in games}
    infos = {i: [] for i in games}
    thoughts = {i: [] for i in games}  # per game, per turn: the thinking token ids
    closings = {i: [] for i in games}  # and the ids that closed it (</think> or the budget sentence)
    records, start, n_turn = {}, time.time(), 0

    def play(i, move, info, th):
        thoughts[i].append(th)
        closings[i].append(finish_ids if info["finished"] else close_ids)
        moves[i].append(move)
        infos[i].append(info)
        try:
            turns[i] = games[i].send((move, info))
        except StopIteration as end:
            records[i] = end.value
            del turns[i]

    name = args.model.replace("/", "__")
    data = DATA / args.difficulty / run
    data.mkdir(parents=True, exist_ok=True)
    ckpt = data / f"{name}.ckpt.json"
    if ckpt.exists():
        saved = json.loads(ckpt.read_text())
        if (saved["seed"], saved["games"]) != (args.seed, args.games):
            raise SystemExit(f"{ckpt} is for seed {saved['seed']} and {saved['games']} games")
        n_turn = saved["turn"]
        for i in games:  # all games still going have made n_turn moves, so the sampling seeds carry on
            for move, info, th in zip(saved["moves"][str(i)], saved["infos"][str(i)], saved["thoughts"][str(i)]):
                texts[i].append(turns[i]["text"])
                play(i, move, info, th)
        print(f"resumed from {ckpt}: turn {n_turn}, {len(records)} games over", flush=True)
    while turns:
        active = sorted(turns)
        prompts = []
        for i in active:
            texts[i].append(turns[i]["text"])
            prompts.append(prompt_ids(tok, texts[i], moves[i], thoughts[i], closings[i], keep))
        think = llm.generate(
            [TokensPrompt(prompt_token_ids=ids) for ids in prompts],
            [SamplingParams(max_tokens=args.budget, temperature=0.6, top_p=0.95, top_k=20, stop_token_ids=[end_think],
                            seed=args.seed * 100003 + levels[i].seed * 1009 + n_turn) for i in active],
            use_tqdm=False)
        thinking, finished = [], []
        for o in think:
            ids = list(o.outputs[0].token_ids)
            finished.append(o.outputs[0].finish_reason == "stop")
            thinking.append(ids[:-1] if ids and ids[-1] == end_think else ids)
        read = llm.generate(
            [TokensPrompt(prompt_token_ids=pr + th + (finish_ids if f else close_ids))
             for pr, th, f in zip(prompts, thinking, finished)],
            SamplingParams(max_tokens=1, temperature=0, logprobs=20), use_tqdm=False)
        for i, pr, th, f, o in zip(active, prompts, thinking, finished, read):
            top = o.outputs[0].logprobs[0]
            p_moves = np.array([sum(np.exp(top[t].logprob) for t in ids if t in top) for ids in variants])
            mass = float(p_moves.sum())
            p_moves = p_moves / mass if mass > 0 else np.full(len(MOVES), 1 / len(MOVES))
            move = list(MOVES)[int(p_moves.argmax())]
            info = {"p": [round(float(x), 4) for x in p_moves], "mass": round(mass, 4), "tokens": len(pr),
                    "think_tokens": len(th), "finished": f}
            play(i, move, info, th)
        n_turn += 1
        tmp = ckpt.with_suffix(".tmp")
        tmp.write_text(json.dumps({"seed": args.seed, "games": args.games, "turn": n_turn, "moves": moves,
                                   "infos": infos, "thoughts": thoughts}))
        tmp.replace(ckpt)
        print(f"turn {n_turn}: {len(active)} games, thinking {np.mean([len(t) for t in thinking]):.0f} tokens "
              f"(finished {np.mean(finished):.2f}), {len(records)} games over, {time.time() - start:.0f} s", flush=True)

    played = [records[i] for i in sorted(records)]
    print(f"{run}  {describe(played)}  ({time.time() - start:.0f} s)", flush=True)
    out = OUT / args.difficulty / run / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "difficulty": args.difficulty, "seed": args.seed,
                               "budget": args.budget, "keep": args.keep, "config": dungeon.config,
                               "results": {run: {"games": played, **stats(played)}}}))
    (data / f"{name}.json").write_text(json.dumps({"model": args.model, "close": CLOSE, "finish": "</think>\n\n",
                                                   "keep": args.keep, "moves": {str(levels[i].seed): moves[i] for i in moves},
                                                   "thinking": {str(levels[i].seed): thoughts[i] for i in thoughts}}))
    ckpt.unlink()
    print("saved", out, "and", data / f"{name}.json")


if __name__ == "__main__":  # vLLM starts its engine in a spawned process
    main()
