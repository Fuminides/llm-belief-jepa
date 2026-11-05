"""Does thinking keep a model out of the pits it walked into? The turns of its saved dungeon games (play_dungeon.py,
argmax) on which a model stepped into a pit next to it are asked again with thinking on, with the same chat up to
that turn.

Thinking is sampled as Qwen recommends for it (temperature 0.6, top-p 0.95, top-k 20) up to a budget of tokens;
if it has not ended by then, it is closed with Qwen's budget sentence. The move is then read as in the games: the
first token of the answer, either case, with or without a leading space. The same chats without thinking are read
again as a check against the saved games. All samples of all turns go through the model as one batch.

  python -m analysis.think_pits --model Qwen/Qwen3.5-27B --budget 2000 --samples 3
  # writes results/dungeon/<difficulty>/think_pits/<model>.json
"""
import argparse
import json
import time

import numpy as np

from games.dungeon import MOVES, Dungeon, step
from games.play_dungeon import LM, OUT, Replay

CLOSE = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"


def pit_turns(r):
    """The fatal turns of the saved argmax games that ended by walking (not sliding) into a pit: the chat up to
    that turn, the move taken, and where every move would have led."""
    dungeon, turns = Dungeon(r["config"]), []
    for game in r["results"]["argmax"]["games"]:
        last = game["log"][-1]
        if game["outcome"] != "lethal" or "slide" in last["events"]:
            continue
        level = dungeon.generate(game["seed"])
        replay = dungeon.play(level, Replay([t["move"] for t in game["log"]]), keep_text=True)
        chat = []
        for t in replay["log"]:
            chat += [{"role": "user", "content": t["text"]}, {"role": "assistant", "content": t["move"]}]
        cell = tuple(last["cell"])
        leads = {m: "wall" if dungeon.blocks(level, step(cell, m)) else level[step(cell, m)] for m in MOVES}
        turns.append({"seed": game["seed"], "turn": len(game["log"]), "chat": chat[:-1], "move": last["move"],
                      "p_saved": last["p"], "leads": leads})
    return turns


def think(lm, chats, budget, samples, seed):
    """Thinking for each chat, `samples` times, in one batch: the text generated after the prompt and the prompt."""
    torch, tok = lm.torch, lm.tok
    tok.padding_side = "left"
    prompts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True, enable_thinking=True)
               for c in chats for _ in range(samples)]
    enc = tok(prompts, add_special_tokens=False, return_tensors="pt", padding=True).to(lm.device)
    torch.manual_seed(seed)
    with torch.no_grad():
        out = lm.model.generate(**enc, max_new_tokens=budget, do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
    texts = [tok.decode([i for i in seq.tolist() if i != tok.pad_token_id], skip_special_tokens=False)
             for seq in out[:, enc.input_ids.shape[1]:]]
    return prompts, texts


def run(name, difficulty, budget, samples, seed):
    path = OUT / difficulty / f"{name.replace('/', '__')}.json"
    turns = pit_turns(json.loads(path.read_text()))
    lm = LM(name)
    for t in turns:  # without thinking, from scratch: should match the saved game
        p, _ = lm.moves(lm.feed(lm.template(t["chat"], generation=True), None)[0])
        t["p_now"] = [round(float(x), 4) for x in p]
    start = time.time()
    prompts, texts = think(lm, [t["chat"] for t in turns], budget, samples, seed)
    print(f"{len(texts)} thinking samples in {time.time() - start:.0f} s", flush=True)
    for i, (prompt, text) in enumerate(zip(prompts, texts)):
        t = turns[i // samples]
        finished = "</think>" in text
        thought = text.split("</think>")[0]
        p, mass = lm.moves(lm.feed(prompt + thought + ("</think>\n\n" if finished else CLOSE), None)[0])
        t.setdefault("thinking", []).append({
            "p": [round(float(x), 4) for x in p], "mass": round(mass, 4), "finished": finished,
            "tokens": lm.length(thought), "answer": text.split("</think>")[1][:200] if finished else None,
            "text": thought})
    moves = list(MOVES)
    print(f"{'seed':>4} {'turn':>4} {'pit':>5}  {'P(pit) saved / now':>18}  with thinking: P(pit), chosen (leads to)")
    for t in turns:
        k = moves.index(t["move"])
        chosen = [moves[int(np.argmax(s["p"]))] for s in t["thinking"]]
        print(f"{t['seed']:4d} {t['turn']:4d} {t['move']:>5}  {t['p_saved'][k]:8.2f} / {t['p_now'][k]:.2f}       "
              + "  ".join(f"{s['p'][k]:.2f} {m} ({t['leads'][m]}, {s['tokens']} tok{'' if s['finished'] else ', cut'})"
                          for s, m in zip(t["thinking"], chosen)))
    samples_all = [(t, s) for t in turns for s in t["thinking"]]
    into_pit = np.mean([t["leads"][moves[int(np.argmax(s["p"]))]] == "pit" for t, s in samples_all])
    print(f"without thinking every one of these {len(turns)} turns went into the pit; with thinking {into_pit:.2f} of "
          f"the samples do; mean thinking {np.mean([s['tokens'] for _, s in samples_all]):.0f} tokens, "
          f"{np.mean([s['finished'] for _, s in samples_all]):.2f} finished within {budget}")
    out = OUT / difficulty / "think_pits" / f"{name.replace('/', '__')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": name, "budget": budget, "samples": samples, "seed": seed, "turns": turns}))
    print("saved", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-27B")
    p.add_argument("--difficulty", default="medium")
    p.add_argument("--budget", type=int, default=2000, help="most thinking tokens before the thinking is closed")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args.model, args.difficulty, args.budget, args.samples, args.seed)
