"""The LLM's states while it plays the dungeon, for latent models of its play (the dungeon's counterpart of collect.py).

The model plays as play_dungeon.py's argmax player: one chat turn per move, thinking off, the move read from the
first answer token. At each turn's decision point (the game's message and the generation prompt are in the chat) two
states are kept: the recurrent state S of chosen linear-attention layers, each head projected R1' S R2 to 16 x 16 as
in collect.py, and the last token's hidden state after the same layers. On medium, where play_dungeon.py has saved
the same model's argmax games, the moves are compared with those.

  python dungeon_states.py --model Qwen/Qwen3.5-9B --difficulty easy --games 200
  # data/dungeon_states/<difficulty>/<model>/shard_*.npz and games.json:
  #   memory [G, T, layers, heads*256] and hidden [G, T, layers, d] (float16), p [G, T, 4] (the four moves'
  #   probabilities), turns [G], seeds [G], layers; T is the difficulty's torch, rows past a game's end are zero.
  #   games.json holds the game records (moves, cells, events, outcome) by seed.
  #   truth [G, T, 4]: what each move would do (index into dungeon.CONSEQUENCES), from the engine.
  python dungeon_states.py --model Qwen/Qwen3.5-27B --difficulty medium_hidden --consequences
  #   data/dungeon_states/<difficulty>/consequences/<model>/: also consequence [G, T, 4, 5], the model's answers to
  #   the consequence question per direction (P of safe, damage, die, slide, wall).
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from collect import memory_reader
from dungeon import CONSEQUENCES, MOVES, Dungeon
from play_dungeon import LM, OUT, Model, describe

DATA = Path("data/dungeon_states")


def default_layers(config, n=7):
    """n linear-attention layers spread over the depth (each at or just below k/(n+1) of it)."""
    types = config.layer_types
    picks = []
    for k in range(1, n + 1):
        i = round(k * len(types) / (n + 1))
        while types[i] != "linear_attention":
            i -= 1
        picks.append(i)
    return picks


class Recorder(Model):
    """play_dungeon.Model's argmax player, also keeping the states at each decision point."""

    def __init__(self, lm, layers, consequences=False):
        super().__init__(lm, "argmax", consequences=consequences)
        self.layers, self.read_memory = layers, memory_reader(lm.model, layers)

    def reset(self):
        super().reset()
        self.memory, self.hidden, self.p, self.conseq = [], [], [], []

    def act(self, turn):
        lm, torch = self.lm, self.lm.torch
        self.messages.append({"role": "user", "content": turn["text"]})
        stable = lm.template(self.messages, generation=False)
        prompt = lm.template(self.messages, generation=True)
        assert stable.startswith(self.fed) and prompt.startswith(stable), "the chat template is not incremental"
        _, self.cache = lm.feed(stable[len(self.fed):], self.cache)
        self.fed = stable
        ids = lm.tok(prompt[len(stable):], add_special_tokens=False, return_tensors="pt").input_ids.to(lm.device)
        with torch.no_grad():
            out = lm.model(ids, past_key_values=copy.deepcopy(self.cache), use_cache=True, logits_to_keep=1,
                           output_hidden_states=True)
        options = turn.get("options", list(MOVES))
        p, mass = lm.answer(out.logits[0, -1], options)
        self.memory.append(self.read_memory(out.past_key_values))
        self.hidden.append(torch.stack([out.hidden_states[i + 1][0, -1] for i in self.layers]).half().cpu().numpy())
        self.p.append(p)
        move = self.choose(p, options)
        info = {"p": [round(float(x), 4) for x in p], "mass": round(mass, 4), "tokens": lm.length(prompt)}
        if self.consequences:
            info.update(self.ask_consequences(stable))
            self.conseq.append(np.array(info["consequence"], np.float32))
        self.messages.append({"role": "assistant", "content": move})
        return move, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--difficulty", default="medium")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--seed", type=int, default=0, help="levels are seeds seed, seed+1, ... (as in play_dungeon.py)")
    p.add_argument("--layers", type=int, nargs="*", help="0-based linear-attention layers (default: 7 over the depth)")
    p.add_argument("--shard-size", type=int, default=25)
    p.add_argument("--consequences", action="store_true", help="also ask the consequence question each turn")
    args = p.parse_args()

    dungeon, lm = Dungeon(difficulty=args.difficulty), LM(args.model)
    layers = args.layers or default_layers(lm.model.config)
    T = dungeon.config["player"]["turns"] or dungeon.config["max_turns"]
    name = args.model.replace("/", "__")
    out = DATA / args.difficulty / ("consequences" if args.consequences else "") / name
    out.mkdir(parents=True, exist_ok=True)
    saved = OUT / args.difficulty / f"{name}.json"
    saved = {g["seed"]: g for g in json.loads(saved.read_text())["results"]["argmax"]["games"]} if saved.exists() else {}
    games_path = out / "games.json"
    records = json.loads(games_path.read_text()) if games_path.exists() else {}
    recorder = Recorder(lm, layers, args.consequences)
    print(f"layers {layers}", flush=True)
    for start in range(args.seed, args.seed + args.games, args.shard_size):
        seeds = list(range(start, min(start + args.shard_size, args.seed + args.games)))
        path = out / f"shard_{start:06d}.npz"
        if path.exists():  # resume
            continue
        t0, rows = time.time(), []
        for seed in seeds:
            game = dungeon.play(dungeon.generate(seed), recorder)
            records[str(seed)] = game
            rows.append((np.stack(recorder.memory), np.stack(recorder.hidden), np.stack(recorder.p),
                         np.stack(recorder.conseq) if recorder.conseq else None, game))
        n = len(seeds)
        memory = np.zeros((n, T, *rows[0][0].shape[1:]), np.float16)
        hidden = np.zeros((n, T, *rows[0][1].shape[1:]), np.float16)
        probs = np.zeros((n, T, rows[0][2].shape[1]), np.float32)
        conseq = np.zeros((n, T, len(MOVES), len(CONSEQUENCES)), np.float32)
        truth = np.full((n, T, len(MOVES)), -1, np.int8)  # index into CONSEQUENCES; -1 past the game's end
        for g, (m, h, q, c, game) in enumerate(rows):
            memory[g, :len(m)], hidden[g, :len(h)], probs[g, :len(q)] = m, h, q
            truth[g, :len(q)] = [[CONSEQUENCES.index(x) for x in t["truth"]] for t in game["log"]]
            if c is not None:
                conseq[g, :len(c)] = c
        np.savez(path, memory=memory, hidden=hidden, p=probs, turns=np.array([len(r[2]) for r in rows]),
                 seeds=np.array(seeds), layers=np.array(layers), truth=truth,
                 **({"consequence": conseq} if args.consequences else {}))
        games_path.write_text(json.dumps(records))
        played = [records[str(s)] for s in seeds]
        same = [s for s in seeds if s in saved]
        match = sum([t["move"] for t in records[str(s)]["log"]] == [t["move"] for t in saved[s]["log"]] for s in same)
        print(f"seeds {seeds[0]}-{seeds[-1]}: {describe(played)}  ({time.time() - t0:.0f} s)"
              + (f"\n{'':10s}moves as in {OUT / args.difficulty}: {match} of {len(same)} games" if same else ""),
              flush=True)


if __name__ == "__main__":
    main()
