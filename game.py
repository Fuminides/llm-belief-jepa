"""Clue game: a hidden thing with binary attributes, noisy sources, Bayesian reference.

The mechanics are fixed (attributes x 2 values x 2 phrasings x 2 sources; with four attributes, 32 actions; an
additive Bayesian update from a prior that the intro may state); the words are a theme. "clue" is the original
secret object. "trust" is the same game about a request an agent has received: the sources are an audit tool and
the user, and a decision (accept or reject) with stated costs gives a Bayes-optimal action besides the posterior.
"trust_reversed" swaps the sources' reliabilities. Select a theme with use(name), or with the CLUE_THEME environment
variable for scripts that only import this module; use_attributes(n) plays with the first n attributes.
use_reliability(False) hides the reliabilities: the intro says only that each source is right more often than not,
and the Bayesian reference marginalises each source's reliability over a uniform prior on [0.5, 1], which couples
the attributes through what the clues teach about the sources.
"""
import math
import os
from itertools import product

THEMES = {
    "clue": {
        "subject": "the object",
        "attributes": {"colour": ("red", "blue"), "shape": ("round", "square"), "size": ("big", "small"),
                       "material": ("wooden", "metal")},
        "sources": {"Alice": 0.9, "Bob": 0.6},  # probability that a statement is correct
        "intro": "A secret object has {number} properties: {props}.{stated} {rel}",
        "prop": "its {attr} is {v0} or {v1}",
        "decision": None,
    },
    # Every value and answer word is one token in Qwen3.5's vocabulary, in lower and upper case, as the bold readout
    # needs (collect.readout); "authorised", "forbidden", "reversible", "unverified", "proceed" and "refuse" start with
    # fragment tokens ("for", "pro", "re", "un") that other words share.
    "trust": {
        "subject": "the request",
        "attributes": {"legality": ("legal", "illegal"), "effect": ("temporary", "permanent"),
                       "origin": ("internal", "external"), "signature": ("signed", "unsigned")},
        "sources": {"the audit tool": 0.9, "the user": 0.6},
        "intro": "An agent has received a request to run an operation. The request has {number} properties: {props}."
                 "{stated} {rel} The agent must accept the request only if it is legal: accepting an illegal "
                 "request costs {cost_wrong}, and rejecting a legal request costs {cost_miss}.",
        "prop": "it is {v0} or {v1}",
        # accept iff P(attribute 0 has value 0) > cost_wrong / (cost_wrong + cost_miss)
        "decision": {"question": "Should the agent {a} or {b} the request?", "tail": "The agent should",
                     "answers": ("accept", "reject"), "attribute": 0, "cost_wrong": 10, "cost_miss": 1},
    },
}
THEMES["trust_reversed"] = {**THEMES["trust"], "sources": {"the audit tool": 0.6, "the user": 0.9}}
NUMBERS = {1: "one", 2: "two", 3: "three", 4: "four"}

# Ways to phrase a clue, by designed convolutedness (0-3). All assert the same thing: {claim} is what the source
# asserts ("is red", "is not blue") and {anticlaim} its negation ("is not red", "is blue"), so level 3 wraps the
# negated claim in a denial. Template 0 is the original phrasing.
TEMPLATES = [
    (0, "{source} says {subject} {claim}."),
    (1, "According to {source}, {subject} {claim}."),
    (1, "{source} claims that {subject} {claim}."),
    (1, "{source} told us {subject} {claim}."),
    (2, "{source}, who had a good look at it earlier, told us that {subject} {claim}."),
    (2, "If you ask {source}, the answer you will get is that {subject} {claim}."),
    (2, "When we spoke to {source} this morning, the reply was that {subject} {claim}."),
    (3, "{source} says it is false that {subject} {anticlaim}."),
    (3, "{source} denies that {subject} {anticlaim}."),
    (3, "Asked whether {subject} {anticlaim}, {source} said that this is false."),
]
LEVELS = sorted({level for level, _ in TEMPLATES})


def use_attributes(n):
    """Play with the first n attributes of the theme (all by default). The action ids of those attributes do not
    change, so a game with fewer attributes is the full game with the clues on the others left out."""
    global ATTRIBUTES, ATTR_NAMES, ACTIONS, N_ACTIONS
    ATTRIBUTES = dict(list(ALL_ATTRIBUTES.items())[:n])
    ATTR_NAMES = list(ATTRIBUTES)
    # An action is (attribute index, asserted value index, negated phrasing, source index).
    ACTIONS = list(product(range(len(ATTR_NAMES)), (0, 1), (False, True), range(len(SOURCE_NAMES))))
    N_ACTIONS = len(ACTIONS)


def use(name):
    """Select a theme: sets the module's ATTRIBUTES, SOURCES, ACTIONS and the rest, with all its attributes."""
    global THEME, SUBJECT, ALL_ATTRIBUTES, SOURCES, SOURCE_NAMES, DECISION
    t = THEMES[name]
    THEME, SUBJECT, ALL_ATTRIBUTES, SOURCES, DECISION = name, t["subject"], t["attributes"], t["sources"], t["decision"]
    SOURCE_NAMES = list(SOURCES)
    use_attributes(len(ALL_ATTRIBUTES))


use(os.environ.get("CLUE_THEME", "clue"))

RELIABILITY_KNOWN = True
RELIABILITY_GRID = [0.5 + 0.5 * (i + 0.5) / 400 for i in range(400)]  # midpoints of a uniform prior on [0.5, 1]


def use_reliability(known):
    """Whether the intro states each source's reliability and the Bayesian reference uses it (the default)."""
    global RELIABILITY_KNOWN
    RELIABILITY_KNOWN = known


def _sentence(text):
    return text[0].upper() + text[1:]


def intro(prior=None):
    """prior: P(value 0) per attribute, stated after the properties (whole percents); None states none."""
    t = THEMES[THEME]
    props = "; ".join(t["prop"].format(attr=a, v0=v0, v1=v1) for a, (v0, v1) in ATTRIBUTES.items())
    if RELIABILITY_KNOWN:
        rel = " ".join(_sentence(f"{s}'s statements are correct {round(100 * r)}% of the time.") for s, r in SOURCES.items())
    else:
        rel = ("How often each source's statements are correct is not known, only that each source is right more "
               "often than not.")
    if prior is None:
        stated = ""
    elif all(p == 0.5 for p in prior):
        stated = " Before any statement, each property is equally likely to have either value (50% each)."
    else:
        chances = "; ".join(f"{v0} {round(100 * p)}%, {v1} {round(100 * (1 - p))}%"
                            for (v0, v1), p in zip(ATTRIBUTES.values(), prior, strict=True))
        stated = f" Before any statement, the chances are: {chances}."
    return t["intro"].format(number=NUMBERS[len(ATTRIBUTES)], props=props, stated=stated, rel=rel, **(DECISION or {}))


def clue_text(action_id, template=0):
    attr, value, negated, source = ACTIONS[action_id]
    values = ATTRIBUTES[ATTR_NAMES[attr]]
    claim = f"is not {values[1 - value]}" if negated else f"is {values[value]}"
    anticlaim = f"is {values[1 - value]}" if negated else f"is not {values[value]}"
    return "\n" + _sentence(TEMPLATES[template][1].format(source=SOURCE_NAMES[source], subject=SUBJECT, claim=claim,
                                                         anticlaim=anticlaim))


def question_text(attr):
    v0, v1 = ATTRIBUTES[ATTR_NAMES[attr]]
    return f"\nQuestion: Is {SUBJECT} {v0} or {v1}?\nAnswer: {_sentence(SUBJECT)} is"


def bayes_belief(action_ids, prior=None):
    """P(value 0) for each attribute after the given clues, from the prior (P(value 0) per attribute; uniform if None).
    With the reliabilities known the update is additive in log-odds per attribute; with them unknown
    (use_reliability(False)) it is _bayes_unknown_reliability."""
    if not RELIABILITY_KNOWN:
        return _bayes_unknown_reliability(action_ids, prior)
    log_odds = [math.log(p / (1 - p)) for p in prior] if prior is not None else [0.0] * len(ATTR_NAMES)
    for action_id in action_ids:
        attr, value, _, source = ACTIONS[action_id]
        r = SOURCES[SOURCE_NAMES[source]]
        log_odds[attr] += (1 if value == 0 else -1) * math.log(r / (1 - r))
    return [1 / (1 + math.exp(-x)) for x in log_odds]


def _bayes_unknown_reliability(action_ids, prior=None):
    """Exact posterior marginals when each source's reliability is unknown: the object (2^A hypotheses, prior per
    attribute) and the reliabilities (uniform on [0.5, 1], integrated on RELIABILITY_GRID) are inferred together, so
    a source's record on one attribute changes the weight of its clues on the others."""
    A = len(ATTR_NAMES)
    p0 = prior if prior is not None else [0.5] * A
    marginal = [0.0] * A
    total = 0.0
    for obj in product((0, 1), repeat=A):
        weight = 1.0
        for a, v in enumerate(obj):
            weight *= p0[a] if v == 0 else 1 - p0[a]
        for s in range(len(SOURCE_NAMES)):
            right = wrong = 0
            for action_id in action_ids:
                attr, value, _, source = ACTIONS[action_id]
                if source == s:
                    if obj[attr] == value:
                        right += 1
                    else:
                        wrong += 1
            if right or wrong:
                weight *= sum(r ** right * (1 - r) ** wrong for r in RELIABILITY_GRID) / len(RELIABILITY_GRID)
        total += weight
        for a, v in enumerate(obj):
            if v == 0:
                marginal[a] += weight
    return [m / total for m in marginal]


def decision_threshold():
    """P(value 0 of the decision attribute) above which the first answer (proceed) has the lower expected cost."""
    d = DECISION
    return d["cost_wrong"] / (d["cost_wrong"] + d["cost_miss"])


def bayes_decision(action_ids, prior=None):
    """1 if the Bayes-optimal action after these clues is the first answer (proceed), else 0. Themes without a
    decision have none."""
    if DECISION is None:
        return None
    return int(bayes_belief(action_ids, prior)[DECISION["attribute"]] > decision_threshold())
