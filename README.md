# What does the model think it knows?

A latent model of how an LLM updates its beliefs. Code, saved results and the project page of the paper.

- **Project page**, with turn-by-turn replays of both games: https://fuminides.github.io/llm-belief-jepa/
- **Paper**: [`docs/paper.pdf`](docs/paper.pdf)

An LLM playing a game holds a belief about the hidden state of the game. We read that belief in the recurrent
memory of Qwen3.5 (2B, 9B and 27B), compare it with the exact Bayesian posterior, and train an *anchored JEPA*, a
small latent model whose latent is tied to the belief the LLM states, on that memory to learn how a clue or a move
changes it. All three sizes diverge from Bayes in the same ways: a negated clue weighs about half an affirmative one,
the sources' stated reliabilities are not properly taken into account, the latest clue weighs most, and a clue moves
the belief less the surer the model already is. The information to compute the true posterior is still in the state
on the turns the answer misses it. The anchored JEPA forecasts the stated belief better than a recurrent model of the
clue ids when the wording varies or the model is larger, reproduces the LLM's update rule on clues it never read, and
reacts to unusual input by its surprise. In the dungeon only the 27B plays correctly; it almost never says a forbidden
word, yet takes a shortcut that hurts others in a third of the games.

## The two games

**The clue game** (`games/game.py`). A secret object has four binary properties (colour, shape, size, material). Two
sources, Alice (right 90% of the time) and Bob (60%), give clues, affirmative or negated, drawn at random, so they may
contradict each other. After each clue the LLM is asked about every property and its belief is read from the
next-token probabilities of the answer. The posterior per property is exact and additive in log-odds, so every answer
has a Bayesian reference and a least-squares fit gives the reliabilities the LLM acts as if it had. The `trust` theme
is the same game about a request an agent must accept or reject, with asymmetric costs; `paraphrase` varies the
wording of each clue over ten templates.

**The dungeon** (`games/dungeon.py`, rules in [`games/DUNGEON.md`](games/DUNGEON.md)). A text maze in the first
person: each turn the LLM reads what it sees along its line of sight, its lives and the torch left, and answers with
one move. Fire burns, pits kill, ice slides, the exit starts out of view. There is no tractable optimal policy, but the
one-step risk of each move is exact, and scripted players (Random, Greedy, Explorer) give the references. The
`medium_voice` condition adds an unreliable bystander; the stress presets add a forbidden word that disarms every
hazard and a shortcut that harms others.

## Layout

The code is six packages, run from the repository root as modules (`python -m package.script ...`). Every script's
docstring says what it does, its arguments and what it writes. Results go to `results/<run>/`; the LLM states that
`collect` caches go to `data/<run>/` (not in the repository: tens of GB per run).

| Package | What is in it |
| --- | --- |
| `games/` | The clue game (`game.py`) and the dungeon (`dungeon.py`, `dungeon.json`), LLMs playing it (`play_dungeon.py`; with thinking through vLLM, `think_dungeon.py`), pixel-art replays (`pixelart.py`), and the earlier one-move maze tests (`maze_feasibility.py`, `fov_feasibility.py`, `terrain.json`). |
| `collect/` | Running a frozen LLM on the games and caching its states and beliefs per turn: `collect.py` (clue game), `dungeon_states.py` (dungeon), `readout_check.py` (how to read the belief), `diagnose.py`. |
| `latent/` | The latent models: `jepa.py` (the first latent model and the forecasting test), `anchored.py` (the anchored JEPA: belief plus free numbers), `heads.py` (several beliefs), `capacity.py`, `generalize.py`, `dungeon_latent.py`, and what the latent holds: `explain.py`, `free_numbers.py`, `free_clusters.py`, `plan_latent.py`, `latent_map.py`. |
| `analysis/` | Where the belief lives (`probe_layers.py`, `memory_analysis.py`, `cache_patch.py`, `bayes_probe.py`), the Bayes gap (`fit_bayes.py`, `bayes_gap.py`, `transfer_test.py`, `update_deviations.py`, `paraphrase_analysis.py`), interventions (`intervene.py`, `latent_edit.py`, `residual_edit.py`), the belief gap (`observers.py`, `surprise_gap.py`), and the dungeon analyses (`dungeon_learning.py`, `dungeon_stress.py`, `dungeon_dominance.py`, `dungeon_mirror.py`, `voice_analysis.py`, `think_pits.py`). |
| `figures/` | The shared style (`figstyle.py`) and the paper's figures, one script each; `project_page.py` builds the project page's demo data. |
| `tests/` | The rules of the two games (`pytest`). |

## Where each hypothesis is tested

| | Hypothesis | Scripts | Results |
| --- | --- | --- | --- |
| H1 | The activations hold dynamics the answers do not reveal | `latent/jepa.py --seeds`, `latent/anchored_paraphrase.py`, `latent/dungeon_latent.py`, `figures/forecast_figure.py` | `results/<run>/seeds/`, `results/dungeon/latent/` |
| H2 | One belief or several | `analysis/observers.py`, `latent/heads.py` | `results/qwen3.5-2b-memory-mid/` |
| H3 | The departure from Bayes is structured; the decoder is faithful under intervention | `analysis/fit_bayes.py`, `analysis/bayes_gap.py`, `analysis/transfer_test.py`, `analysis/intervene.py`, `analysis/latent_edit.py`, `analysis/residual_edit.py` | `results/bayes_gap/`, `results/<run>/implied_bayes.*` |
| H4 | The departure is in the use of the state, not its content | `analysis/bayes_probe.py`, `latent/anchored.py`, `analysis/update_deviations.py`, `latent/free_numbers.py`, `latent/free_clusters.py` | `results/qwen3.5-2b-memory-mid/` |
| H5 | The gap between internal and stated belief is structured | `analysis/surprise_gap.py` | `results/qwen3.5-2b-memory-mid/surprise_gap.*` |
| H6 | Rule-breaking under pressure | `analysis/dungeon_stress.py` | `results/dungeon/stress/` |
| | Playing and learning the dungeon | `games/play_dungeon.py`, `games/think_dungeon.py`, `analysis/dungeon_learning.py`, `analysis/think_pits.py`, `analysis/voice_analysis.py` | `results/dungeon/medium*/`, `results/dungeon/learning/` |

`results/` is named after the data runs: the model, then `-memory` when the state is the recurrent memory *S* of the
Gated DeltaNet layers (`-mid`, `-all`: which layers), `-paraphrase` for paraphrased clues, `-trust` and
`-trust_reversed` for the trust theme, `-randprior` for a stated random prior, `-2attr`/`-3attr` for fewer
properties. Each folder holds the figures (`.pdf`, `.png`) and the numbers behind them (`.json`).

## Running

Python 3.12 with `pip install -r requirements.txt`; the models are downloaded from the Hugging Face hub on first use.
A clue-game collection with Qwen3.5-2B fits on an 8 GB GPU; the 9B and 27B runs and the dungeon games with thinking
were played on a 121 GB machine. The pipeline for one model:

```
pytest                                                       # the rules of the two games
python -m collect.collect --help                             # 1. cache states and beliefs on the clue game
python -m analysis.probe_layers --help                       # 2. which layer holds the belief
python -m latent.jepa --help                                 # 3. the anchored JEPA, forecasting (H1)
python -m analysis.fit_bayes --help                          # 4. the implied observer and the Bayes gap (H3)
python -m games.play_dungeon --model Qwen/Qwen3.5-27B        # whole dungeon games
python -m games.pixelart results/dungeon/medium/Qwen__Qwen3.5-27B.json --game 4   # a saved game as PNG and GIF
python -m figures.project_page                               # the project page's demo data
```

## Citation and licence

The code is under the MIT licence (`LICENSE`).

```bibtex
@unpublished{fumanalidocin2026belief,
  title  = {What does the model think it knows? A latent model of how an {LLM} updates its beliefs},
  author = {Fumanal-Idocin, Javier},
  year   = {2026},
  note   = {Preprint}
}
```
