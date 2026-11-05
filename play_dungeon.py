"""Language models playing whole games of the dungeon (dungeon.py; the rules are in DUNGEON.md).

Each turn is a chat turn: the game's message (the rules on the first turn, then what happened, lives and torch, the
view and the question) and the model's move, read like fov_feasibility.py from the first token of the answer (either
case, with or without a leading space), thinking disabled. The whole game stays in the chat, so the model can
remember what has gone out of view. Two policies on the same levels: argmax (the most probable move) and sample (a
move drawn from the four moves' probabilities, renormalised). The scripted players of dungeon.py are the references:
Random, Greedy (plans on what it sees, remembers only the cell it has just left) and Explorer (knows where it is and
remembers everything it has seen).

The chat is fed incrementally. The cache holds everything up to the last game message as the chat template renders
it (past moves without the empty thinking block), and the generation prompt runs on a copy of the cache. --check
compares this with running the whole prompt from scratch.

  python play_dungeon.py --model Qwen/Qwen3.5-2B --check     # one game, each turn also run from scratch
  python play_dungeon.py --model Qwen/Qwen3.5-2B             # writes results/dungeon/medium/<model>.json
  python play_dungeon.py --plot                              # results/dungeon_medium.png from every run so far
  python play_dungeon.py --transcript Qwen/Qwen3.5-2B --policy argmax --game 3   # one saved game as text
--difficulty picks an entry of dungeon.json's difficulties (default medium).
"""
import argparse
import copy
import glob
import json
import re
import time
from pathlib import Path

import numpy as np

from dungeon import CONSEQUENCES, MIRROR, MOVES, PLAYERS, Dungeon, consequence_question, merge, summary

POLICIES = ("argmax", "sample")
OUT = Path("results/dungeon")


class Model:
    """A language model as a player. reset() starts a new chat; act() adds the game's message and reads the move."""

    def __init__(self, lm, policy, seed=0, check=False, consequences=False, forced=None):
        """consequences: also ask the consequence question per direction, on side branches the chat never keeps.
        forced: play these moves instead of the policy's (the probabilities are still read and logged)."""
        self.lm, self.policy, self.rng, self.check = lm, policy, np.random.default_rng(seed), check
        self.consequences, self.forced = consequences, forced

    def reset(self):
        self.messages, self.cache, self.fed = [], None, ""

    def act(self, turn):
        lm = self.lm
        self.messages.append({"role": "user", "content": turn["text"]})
        stable = lm.template(self.messages, generation=False)
        prompt = lm.template(self.messages, generation=True)
        assert stable.startswith(self.fed) and prompt.startswith(stable), "the chat template is not incremental"
        _, self.cache = lm.feed(stable[len(self.fed):], self.cache)
        self.fed = stable
        logits, _ = lm.feed(prompt[len(stable):], copy.deepcopy(self.cache))
        options = turn.get("options", list(MOVES))
        p, mass = lm.answer(logits, options)
        move = self.choose(p, options)
        info = {"p": [round(float(x), 4) for x in p], "mass": round(mass, 4), "tokens": lm.length(prompt)}
        if self.check:  # the same prompt from scratch, without the cache
            fresh, _ = lm.answer(lm.feed(prompt, None)[0], options)
            info["check"] = float(np.abs(fresh - p).max())
        if self.consequences:
            info.update(self.ask_consequences(stable))
        self.messages.append({"role": "assistant", "content": move})
        return move, info

    def choose(self, p, options):
        if self.forced is not None:
            return self.forced[len(self.messages) // 2]
        return options[int(p.argmax()) if self.policy == "argmax" else int(self.rng.choice(len(options), p=p))]

    def ask_consequences(self, stable):
        """The consequence question for each direction, as a further user message after the game's, each on its own
        branch of the cache, with the answers listed in both orders: P(safe, damage, die, slide, wall) per direction
        (renormalised per order, then averaged), the share of the mass they hold, and the total variation between
        the two orders."""
        lm = self.lm
        branches = [lm.template(self.messages + [{"role": "user", "content": consequence_question(m, reverse)}],
                                generation=True)[len(stable):] for reverse in (False, True) for m in MOVES]
        out = [lm.answer(logits, CONSEQUENCES) for logits in lm.branches(branches, self.cache)]
        q = np.array([a for a, _ in out]).reshape(2, len(MOVES), len(CONSEQUENCES))
        mass = np.array([m for _, m in out]).reshape(2, len(MOVES))
        return {"consequence": q.mean(0).round(4).tolist(), "consequence_mass": mass.mean(0).round(4).tolist(),
                "consequence_order_tv": (0.5 * np.abs(q[0] - q[1]).sum(-1)).round(4).tolist()}


class LM:
    def __init__(self, name):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.device = torch, "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(name)
        # Straight onto the device: on unified memory (GB10) a CPU copy first would hold the weights twice.
        self.model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.bfloat16 if self.device == "cuda" else torch.float32, device_map=self.device).eval()
        self.ids = {}
        self.option_ids(list(MOVES))

    def template(self, messages, generation):
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation,
                                            enable_thinking=False)

    def length(self, text):
        return len(self.tok(text, add_special_tokens=False).input_ids)

    def feed(self, text, cache):
        """Run text after the cache (None: from scratch); the last position's logits and the updated cache."""
        ids = self.tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        with self.torch.no_grad():
            out = self.model(ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        return out.logits[0, -1], out.past_key_values

    def option_ids(self, words):
        """Per answer word, the first tokens of its four spellings (with or without a leading space, lower or
        capitalised); the words must not share one."""
        key = tuple(words)
        if key not in self.ids:
            ids = [sorted({self.tok(s + w, add_special_tokens=False).input_ids[0] for s in ("", " ")
                           for w in (word, word.capitalize())}) for word in words]
            flat = [i for vs in ids for i in vs]
            assert len(set(flat)) == len(flat), f"answer words share a first token: {words}"
            self.ids[key] = ids
        return self.ids[key]

    def answer(self, logits, words):
        """The answer words' probabilities, renormalised, and the share of the next-token mass they hold."""
        p = self.torch.softmax(logits.float(), 0)
        q = self.torch.stack([p[v].sum() for v in self.option_ids(words)]).cpu().numpy()
        return q / q.sum(), float(q.sum())

    def moves(self, logits):
        """The four moves' probabilities, renormalised, and the share of the next-token mass they hold."""
        return self.answer(logits, list(MOVES))

    def branches(self, texts, cache):
        """The last position's logits of each text run after the cache, in one batch on a copy of the cache. Texts are
        right-padded: the model is causal, so padding after a text cannot change the logits at its last token."""
        torch = self.torch
        encoded = [self.tok(t, add_special_tokens=False).input_ids for t in texts]
        batch = torch.zeros(len(texts), max(map(len, encoded)), dtype=torch.long)
        for i, ids in enumerate(encoded):
            batch[i, :len(ids)] = torch.tensor(ids)
        branch = copy.deepcopy(cache)
        branch.reorder_cache(torch.zeros(len(texts), dtype=torch.long, device=self.device))
        with torch.no_grad():
            logits = self.model(batch.to(self.device), past_key_values=branch, use_cache=True).logits
        return [logits[i, len(ids) - 1] for i, ids in enumerate(encoded)]


class Replay:
    """Plays the given moves: with the same level and game seed, a saved game again, with its text."""

    def __init__(self, moves):
        self.moves = moves

    def reset(self):
        self.turn = 0

    def act(self, turn):
        self.turn += 1
        return self.moves[self.turn - 1], {}


def stats(games):
    """Per-turn behaviour: moves into walls, damage taken, turns on cells already visited."""
    turns = [t for g in games for t in g["log"]]
    revisits = [sum(t["cell"] in {u["cell"] for u in g["log"][:i]} for i, t in enumerate(g["log"])) / len(g["log"])
                for g in games]
    return {"bump_rate": float(np.mean(["bump" in t["events"] for t in turns])),
            "damage_per_game": float(np.mean([sum("damage" in t["events"] for t in g["log"]) for g in games])),
            "revisit_rate": float(np.mean(revisits)),
            "explored": float(np.mean([g["explored"] for g in games]))}


def describe(games):
    s = stats(games)
    return (f"{summary(games)}\n{'':10s}bumps {s['bump_rate']:.2f}/turn  damage {s['damage_per_game']:.2f}/game  "
            f"revisits {s['revisit_rate']:.2f}  explored {s['explored']:.2f}")


def run(name, difficulty, n_games, seed, check, stated=False, consequences=False, policies=POLICIES):
    dungeon, lm = Dungeon(difficulty=difficulty), LM(name)
    dungeon.config["level"]["stated"] = stated
    levels = [dungeon.generate(seed + i) for i in range(n_games)]
    if check:
        game = dungeon.play(levels[0], Model(lm, "argmax", check=True))
        for t in game["log"]:
            print(f"turn tokens {t['tokens']:5d}  max |p(cache) - p(fresh)| {t['check']:.4f}  p {t['p']}")
        diffs = [t["check"] for t in game["log"]]
        print(f"{game['outcome']}: {game['end']}  mean difference {np.mean(diffs):.4f}, max {max(diffs):.4f}")
        if np.mean(diffs) > 0.03:
            raise SystemExit("the cached chat does not match a fresh run")
        return
    results = {}
    for policy in policies:
        start, games = time.time(), []
        for level in levels:
            games.append(dungeon.play(level, Model(lm, policy, seed=level.seed, consequences=consequences)))
        results[policy] = {"games": games, **stats(games)}
        print(f"{policy:8s}  {describe(games)}  ({time.time() - start:.0f} s)", flush=True)
    out = OUT / difficulty / ("stated_level" if stated else "") / ("consequences" if consequences else "")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name.replace('/', '__')}.json"
    path.write_text(json.dumps({"model": name, "difficulty": difficulty, "seed": seed, "config": dungeon.config,
                                "results": results}))
    print("saved", path)


def mirror(name, difficulty, n_games):
    """The mirror test. The generator is symmetric under left-right reflection, so a player's move probabilities on a
    level's mirror image should be its probabilities on the level with left and right swapped. Each saved argmax game
    is replayed move for move on its level ("original": the noise floor against the saved probabilities), and with the
    moves mirrored on the mirror image, whose text lists the rays and the tied cells reflected ("mirror": the same text
    with left and right swapped) or in the usual order ("mirror_listing": the listing order moves too). The game seed
    is the same, so the slides are."""
    saved = json.loads((OUT / difficulty / f"{name.replace('/', '__')}.json").read_text())["results"]["argmax"]["games"]
    plain, lm = Dungeon(difficulty=difficulty), LM(name)
    reflected = Dungeon(merge(plain.config, {"view": {"mirrored": True}}))
    results = {}
    for label, d, flip in (("original", plain, False), ("mirror", reflected, True), ("mirror_listing", plain, True)):
        start, games = time.time(), []
        for g in saved[:n_games]:
            level, moves = d.generate(g["seed"]), [t["move"] for t in g["log"]]
            if flip:
                level, moves = d.mirror(level), [MIRROR[m] for m in moves]
            games.append(d.play(level, Model(lm, "argmax", forced=moves)))
        results[label] = games
        print(f"{label:15s} {time.time() - start:.0f} s", flush=True)
    path = OUT / difficulty / "mirror" / f"{name.replace('/', '__')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": name, "difficulty": difficulty, "results": results}))
    print("saved", path)


def runs(difficulty):
    found = [json.loads(Path(f).read_text()) for f in glob.glob(f"{OUT}/{difficulty}/*.json")]
    return sorted(found, key=lambda r: (r["model"].split("-")[0], float(re.search(r"-(\d+(?:\.\d+)?)B", r["model"]).group(1))))


def scripted(config, n_games, seed):
    dungeon = Dungeon(config)
    return {name: [dungeon.play(dungeon.generate(seed + i), cls(dungeon, seed + i)) for i in range(n_games)]
            for name, cls in PLAYERS.items()}


def plot(difficulty):
    import matplotlib.pyplot as plt

    import figstyle

    found = runs(difficulty)
    n = len(found[0]["results"]["argmax"]["games"])
    labels = {"random": "Random", "greedy": "Greedy, no memory", "explorer": "Explorer, memory"}
    rows = [(labels[name], games) for name, games in scripted(found[0]["config"], n, found[0]["seed"]).items()]
    rows += [(f"{r['model'].split('/')[-1]} {policy}", r["results"][policy]["games"]) for r in found for policy in POLICIES]
    kinds = [("exit", "Reached the exit", figstyle.GREEN), ("lethal", "Fell into a pit", figstyle.VERMILLION),
             ("lives", "Burned (no lives)", figstyle.ORANGE), ("time", "Torch burned out", figstyle.GREY),
             ("cap", "Stopped at max_turns", figstyle.PURPLE)]
    kinds = [k for k in kinds if any(g["outcome"] == k[0] for _, games in rows for g in games)]
    figstyle.apply()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(figstyle.DOUBLE, 0.25 * len(rows) + 1.2), sharey=True,
                                 width_ratios=(2, 1))
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    for kind, label, colour in kinds:
        share = np.array([np.mean([g["outcome"] == kind for g in games]) for _, games in rows])
        ax.barh(y, share, 0.75, left=left, color=colour, label=label)
        left += share
    for a in (ax, bx):  # scripted players above, models below
        a.axhline(y[len(labels) - 1] - 0.5, color="black", lw=0.5)
    ax.set_yticks(y, [label for label, _ in rows])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of games")
    ax.set_title("Outcome", loc="left")
    explored = [np.mean([g["explored"] for g in games]) for _, games in rows]
    bx.barh(y, explored, 0.75, color=figstyle.BLUE)
    bx.set_xlim(0, 1)
    bx.set_xlabel("Share of the level seen")
    bx.set_title("Exploration", loc="left")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=len(kinds))
    figstyle.panel_labels([ax, bx])
    figstyle.save(fig, f"results/dungeon_{difficulty}")


def transcript(name, difficulty, policy, index):
    """Replay a saved game with its moves and print it as the model saw it, with the map and the move probabilities."""
    r = json.loads((OUT / difficulty / f"{name.replace('/', '__')}.json").read_text())
    game = r["results"][policy]["games"][index]
    dungeon = Dungeon(r["config"])
    level = dungeon.generate(game["seed"])
    replay = dungeon.play(level, Replay([t["move"] for t in game["log"]]), keep_text=True)
    for t, logged in zip(replay["log"], game["log"]):
        p = "  ".join(f"{m} {x:.2f}" for m, x in zip(MOVES, logged["p"]))
        print(dungeon.render(level, t["cell"]), "\n", t["text"], "\n> ", t["move"], f"   ({p})\n", sep="")
    print(replay["end"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--difficulty", default="medium")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--seed", type=int, default=0, help="levels are seeds seed, seed+1, ...")
    p.add_argument("--check", action="store_true", help="play one game comparing the cache with a fresh run")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--transcript", help="print a saved game of this model")
    p.add_argument("--policy", default="argmax", choices=POLICIES)
    p.add_argument("--game", type=int, default=0)
    p.add_argument("--stated-level", action="store_true", help="the rules also state the level's hazard densities and "
                   "where the exit is (dungeon.json level.stated); results in results/dungeon/<difficulty>/stated_level/")
    p.add_argument("--consequences", action="store_true", help="also ask the consequence question each turn; results in "
                   "results/dungeon/<difficulty>/consequences/")
    p.add_argument("--policies", nargs="+", default=list(POLICIES), choices=POLICIES)
    p.add_argument("--mirror", action="store_true", help="the mirror test on the saved argmax games of --model")
    args = p.parse_args()
    if args.plot:
        plot(args.difficulty)
    elif args.transcript:
        transcript(args.transcript, args.difficulty, args.policy, args.game)
    elif args.mirror:
        mirror(args.model, args.difficulty, args.games)
    else:
        run(args.model, args.difficulty, args.games, args.seed, args.check, args.stated_level, args.consequences,
            args.policies)
