"""Run a frozen LLM on random clue sequences; cache its states and its beliefs per turn.

Each shard holds, for E episodes of T turns (index 0 is the state before any clue):
  acts         [E, T+1, n_layers+1, d]  last-token hidden states, float16 (index 0: embeddings; last: after the final norm)
  pool         [E, T+1, n_layers+1, d]  hidden states averaged over the tokens of that turn's text, float16
  llm_belief   [E, T+1, A]              P(value 0) per attribute (A: --attributes, 4 by default), read from the LLM
                                        (mean of both option orders)
  value_mass   [E, T+1, A]              probability of the value tokens among all tokens, before normalising
  bayes_belief [E, T+1, A]              the Bayesian reference
  actions      [E, T]                   action ids
  templates    [E, T]                   phrasing of each clue (game.TEMPLATES), only with --paraphrase
  memory       [E, T+1, m, 4096]        recurrent state S of chosen linear-attention layers after each turn, each
                                        head projected R1' S R2 to 16 x 16 (Qwen3.5), only with --memory-layers
  llm_decision [E, T+1]                 P(first answer, e.g. proceed) of the theme's decision question, from the LLM
  decision_mass[E, T+1]                 probability of the two answer tokens among all tokens
  bayes_decision [E, T+1]               1 where the Bayes-optimal decision is the first answer
  theme        ()                       the theme (game.THEMES); the decision keys only with a theme that has one
  prior        [E, A]                   P(value 0) per attribute stated in the intro, only with --prior or
                                        --stated-prior; the Bayesian reference starts from it
  reliability_known ()                  False with --unknown-reliability: the intro hides the sources' reliabilities
                                        and the Bayesian reference infers them (game.use_reliability)
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from games import game
FORMATS = ("chat", "plain", "letters", "filler", "bold")
FILLER = "\nThe weather is calm today."
RANDOM_PRIORS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)  # --prior random draws each attribute's from these


def readout(tok, fmt, extra=(), prior=None):
    """Intro text and, per attribute, the two question branches (both option orders) with the ids of the answer
    tokens for (value 0, value 1); each value can have several ids, whose probabilities are summed. A theme with a
    decision (game.DECISION) adds its question as a last group, with its own phrasing and answers.

    chat: the value word after "The object is". plain: the same without the chat template. letters: answer A or B,
    so the answer token is not a word the clues mention. filler: chat with a neutral sentence before the question.
    bold: chat ending in "The object is **", for models that put the answer in bold (Qwen3.5); lower and upper case.
    extra: more (value 0, value 1) pairs to ask about after the four attributes, e.g. colours outside the game.
    prior: P(value 0) per attribute to state in the intro (game.intro).
    """
    subject = game.SUBJECT
    is_tail = subject[0].upper() + subject[1:] + " is"
    if fmt == "plain":
        intro_text, head = game.intro(prior), ""
    else:
        head, chat_tail = tok.apply_chat_template([{"role": "user", "content": "XXX"}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False).split("XXX")
        intro_text = head + game.intro(prior)
    enc = lambda text: tok(text, add_special_tokens=False, return_tensors="pt").input_ids
    first = lambda word: [tok(" " + word, add_special_tokens=False).input_ids[0]]
    asked = [(f"Is {subject} {{a}} or {{b}}?", is_tail, pair) for pair in list(game.ATTRIBUTES.values()) + list(extra)]
    if game.DECISION:
        asked.append((game.DECISION["question"], game.DECISION["tail"], tuple(game.DECISION["answers"])))
    questions = []
    for question, answer_tail, (v0, v1) in asked:
        if fmt == "plain":
            tail = ""
        else:
            tail = chat_tail + {"letters": "Answer:", "bold": answer_tail + " **"}.get(fmt, answer_tail)
        branches = []
        for a, b in ((v0, v1), (v1, v0)):
            text = (FILLER if fmt == "filler" else "") + "\nQuestion: " + question.format(a=a, b=b)
            if fmt == "plain":
                text += "\nAnswer: " + answer_tail
            if fmt == "letters":
                text += f" Reply A for {a} or B for {b}."
                ids = [first("A"), first("B")] if a == v0 else [first("B"), first("A")]
            elif fmt == "bold":
                ids = [[tok(w, add_special_tokens=False).input_ids[0] for w in (v, v.capitalize())] for v in (v0, v1)]
            else:
                ids = [first(v0), first(v1)]
            assert not set(ids[0]) & set(ids[1]), f"{v0}/{v1} share a first token"
            branches.append((enc(text + tail), ids))
        questions.append(branches)
    return intro_text, questions


def question_batch(questions, device):
    """All question branches of a turn in one batch, right-padded: the models are causal, so padding after a question
    cannot change the logits read at its last token. Returns the branches, the batch and each branch's last position."""
    branches = [b for per_attr in questions for b in per_attr]
    lengths = torch.tensor([q.shape[1] for q, _ in branches])
    batch = torch.zeros(len(branches), int(lengths.max()), dtype=torch.long)
    for i, (q, _) in enumerate(branches):
        batch[i, :q.shape[1]] = q[0]
    return branches, batch.to(device), (lengths - 1).to(device)


@torch.no_grad()
def read_beliefs(model, past, questions, branches, batch, last):
    """P(value 0) per attribute and the probability of the answer tokens, averaged over the two option orders."""
    # Branch on a copy repeated along the batch: caches are extended in place, and the main context must only ever
    # hold the clues.
    n = len(branches)
    branch_cache = copy.deepcopy(past)
    branch_cache.reorder_cache(torch.zeros(n, dtype=torch.long, device=batch.device))
    logits = model(batch, past_key_values=branch_cache).logits[torch.arange(n, device=batch.device), last]
    probs = torch.softmax(logits.float(), dim=-1)
    p0 = torch.stack([probs[i, ids[0]].sum() for i, (_, ids) in enumerate(branches)])
    p1 = torch.stack([probs[i, ids[1]].sum() for i, (_, ids) in enumerate(branches)])
    return (p0 / (p0 + p1)).view(len(questions), -1).mean(1).tolist(), (p0 + p1).view(len(questions), -1).mean(1).tolist()


@torch.no_grad()
def run_episode(model, tok, clues, intro_text, questions, steer=None, memory=None):
    """steer: optional {piece index: (layer, vector, token positions or None for all)}; while that piece (0 = intro,
    i = clue i) runs, the vector is added to the hidden state output by that layer (hidden_states[layer]) at those
    tokens of the piece. Questions run unsteered. memory: optional function of the cache, called after each piece;
    its outputs are returned as a fifth value (None without it)."""
    device = model.device
    bos = tok.bos_token  # chat templates that already start with BOS (Gemma) must not get a second one
    pieces = [tok(intro_text, add_special_tokens=not (bos and intro_text.startswith(bos)), return_tensors="pt").input_ids]
    pieces += [tok(clue, add_special_tokens=False, return_tensors="pt").input_ids for clue in clues]

    batch = question_batch(questions, device)

    acts, pools, beliefs, masses, memories, past = [], [], [], [], [], None
    for i, piece in enumerate(pieces):
        handle = None
        if steer and i in steer:
            layer, vector, positions = steer[i]
            shift = torch.zeros(piece.shape[1], vector.shape[0], dtype=vector.dtype, device=device)
            shift[slice(None) if positions is None else positions] = vector
            handle = model.model.layers[layer - 1].register_forward_hook(lambda mod, inp, out, v=shift: out + v)
        out = model(piece.to(device), past_key_values=past, use_cache=True, output_hidden_states=True)
        hidden = list(out.hidden_states)
        if handle is not None:
            handle.remove()
            hidden[layer] = hidden[layer] + shift  # transformers records that layer's output before the hook
        past = out.past_key_values
        if memory is not None:
            memories.append(memory(past))
        acts.append(torch.stack([h[0, -1] for h in hidden]).half().cpu())
        pools.append(torch.stack([h[0].float().mean(0) for h in hidden]).half().cpu())
        belief, mass = read_beliefs(model, past, questions, *batch)
        beliefs.append(belief)
        masses.append(mass)
    return (torch.stack(acts).numpy(), torch.stack(pools).numpy(), beliefs, masses,
            np.stack(memories) if memory is not None else None)


def memory_projection(model, dim=16, seed=0):
    """Fixed Gaussian R1 (key x dim) and R2 (value x dim) that project each head's recurrent state S as R1' S R2."""
    cfg = model.config
    g = torch.Generator().manual_seed(seed)
    r1 = (torch.randn(cfg.linear_key_head_dim, dim, generator=g) / dim ** 0.5).to(model.device)
    r2 = (torch.randn(cfg.linear_value_head_dim, dim, generator=g) / dim ** 0.5).to(model.device)
    return r1, r2


def memory_reader(model, layers, dim=16, seed=0):
    """Function of the cache returning, for each chosen linear-attention layer, its recurrent state S (heads x key x
    value) projected per head as R1' S R2 with fixed Gaussian R1, R2 (key x dim, value x dim): [layers, heads*dim^2]."""
    r1, r2 = memory_projection(model, dim, seed)

    def read(past):
        out = []
        for i in layers:
            S = past.layers[i].recurrent_states[0][0].float()  # [heads, key, value]
            out.append((r1.T @ S @ r2).reshape(-1))
        return torch.stack(out).half().cpu().numpy()
    return read


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--turns", type=int, default=8)
    p.add_argument("--shard-size", type=int, default=500)
    p.add_argument("--out", default="data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--format", choices=FORMATS, default="chat")
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")  # Gemma needs bfloat16
    p.add_argument("--paraphrase", action="store_true", help="phrase each clue with a random template")
    p.add_argument("--memory-layers", type=int, nargs="*", default=[],
                   help="0-based linear-attention layers whose recurrent state to keep (Qwen3.5)")
    p.add_argument("--theme", choices=list(game.THEMES), default="clue", help="the words of the game (game.THEMES)")
    p.add_argument("--attributes", type=int, default=len(game.ALL_ATTRIBUTES),
                   help="play with the first n attributes (game.use_attributes)")
    p.add_argument("--stated-prior", action="store_true", help="the intro states the uniform prior")
    p.add_argument("--prior", nargs="+", help="P(value 0) per attribute to state in the intro, or 'random': each "
                   "episode draws each attribute's from RANDOM_PRIORS")
    p.add_argument("--unknown-reliability", action="store_true",
                   help="the intro does not state how reliable the sources are; the Bayesian reference infers it")
    args = p.parse_args()
    game.use(args.theme)
    game.use_attributes(args.attributes)
    game.use_reliability(not args.unknown_reliability)
    A = len(game.ATTR_NAMES)
    fixed_prior = [0.5] * A if args.stated_prior else None
    if args.prior and args.prior != ["random"]:
        fixed_prior = [float(x) for x in args.prior]
        assert len(fixed_prior) == A, f"--prior needs {A} values"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    # Straight onto the device: on unified memory (GB10) a CPU copy first would hold the weights twice.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype) if device == "cuda" else torch.float32, device_map=device
    ).eval()
    intro_text, questions = readout(tok, args.format)
    memory = memory_reader(model, args.memory_layers) if args.memory_layers else None

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    # Templates and priors come from their own generators, so the clue sequences match a run without them.
    template_rng = np.random.default_rng(args.seed + 1)
    prior_rng = np.random.default_rng(args.seed + 2)
    by_level = {level: [i for i, (l, _) in enumerate(game.TEMPLATES) if l == level] for level in game.LEVELS}
    for start in range(0, args.episodes, args.shard_size):
        n = min(args.shard_size, args.episodes - start)
        actions = rng.integers(game.N_ACTIONS, size=(n, args.turns))
        levels = template_rng.choice(game.LEVELS, size=(n, args.turns))  # level first, so each is equally likely
        templates = np.array([template_rng.choice(by_level[l]) for l in levels.ravel()]).reshape(n, args.turns)
        if not args.paraphrase:
            templates[:] = 0
        priors = (prior_rng.choice(RANDOM_PRIORS, size=(n, A)) if args.prior == ["random"]
                  else None if fixed_prior is None else np.array([fixed_prior] * n))
        path = out / f"shard_{start:06d}.npz"
        if path.exists():  # resume; actions are still drawn so later shards match an uninterrupted run
            continue
        given = [None] * n if priors is None else priors.tolist()
        intros = [intro_text if pr is None else readout(tok, args.format, prior=pr)[0] for pr in given]
        acts, pools, llm, mass, mem = zip(*(run_episode(model, tok, [game.clue_text(x, t) for x, t in zip(a, ts)],
                                                        intro, questions, memory=memory)
                                            for a, ts, intro in zip(actions, templates, intros)))
        llm, mass = np.array(llm, dtype=np.float32), np.array(mass, dtype=np.float32)
        bayes = [[game.bayes_belief(a[:t].tolist(), pr) for t in range(args.turns + 1)] for a, pr in zip(actions, given)]
        decision = {}
        if game.DECISION:  # the last question group is the decision
            decision = {"llm_decision": llm[:, :, -1], "decision_mass": mass[:, :, -1],
                        "bayes_decision": np.array([[game.bayes_decision(a[:t].tolist(), pr) for t in range(args.turns + 1)]
                                                    for a, pr in zip(actions, given)], dtype=np.int8)}
            llm, mass = llm[:, :, :-1], mass[:, :, :-1]
        np.savez(
            path,
            acts=np.stack(acts),
            pool=np.stack(pools),
            llm_belief=llm,
            value_mass=mass,
            bayes_belief=np.array(bayes, dtype=np.float32),
            actions=actions,
            theme=args.theme,
            reliability_known=not args.unknown_reliability,
            **decision,
            **({"templates": templates} if args.paraphrase else {}),
            **({"memory": np.stack(mem), "memory_layers": np.array(args.memory_layers)} if memory else {}),
            **({"prior": priors.astype(np.float32)} if priors is not None else {}),
        )
        print(f"{start + n}/{args.episodes} episodes", flush=True)


if __name__ == "__main__":
    main()
