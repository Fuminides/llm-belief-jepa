# What does the model think it knows?

A latent model of how an LLM updates its beliefs. Code, results and the project page of the paper.

- Project page, with replays of the two games: https://fuminides.github.io/llm-belief-jepa/
- Paper: `docs/paper.pdf`

An LLM playing a game holds a belief about the hidden state of the game. We read that belief in the recurrent memory of
Qwen3.5 (2B, 9B, 27B), compare it with the exact Bayesian posterior, and train an anchored JEPA on the memory to learn
how a clue or a move changes it. The departure from Bayes is structured and the same at every size: a negation counts
half, the sources weigh nearer each other than stated, the latest clue weighs most. The information to be right is in
the state; the update misuses it. The belief is one and held in every layer. The JEPA's prediction error flags unusual
input, not wrong answers. In the dungeon only the 27B plays the game, and it breaks a forbidden rule once in 800 games.

## The two games

**The clue game** (`game.py`). A secret object has four binary properties. Two sources, Alice (right 90% of the time)
and Bob (60%), give clues, affirmative or negated, that may contradict each other. After each clue the LLM is asked
about each property and its belief is read from the next-token probabilities. The posterior per property is exact, so
every answer has a Bayesian reference. The `trust` theme is the same game about a request an agent must accept or
reject.

**The dungeon** (`dungeon.py`, rules in `DUNGEON.md`). A text maze in the first person: the LLM reads what it sees
along its line of sight, its lives and the torch left, and answers with one move per turn. Fire burns, pits kill, ice
slides, and the exit starts out of view. The stress presets add a forbidden word and a harmful shortcut.

## Layout

Every script is a command: its docstring states what it does, its arguments and what it writes. Results go to
`results/<run>/`, collected LLM states to `data/<run>/` (not in the repository: they are tens of GB).

| Area | Scripts |
| --- | --- |
| Games | `game.py`, `dungeon.py`, `play_dungeon.py`, `think_dungeon.py` (vLLM), `pixelart.py`, `test_game.py`, `test_dungeon.py` |
| Collecting states and beliefs | `collect.py`, `dungeon_states.py`, `readout_check.py`, `diagnose.py` |
| Reading the belief | `probe_layers.py`, `memory_analysis.py`, `cache_patch.py`, `bayes_probe.py` |
| The anchored JEPA | `jepa.py`, `anchored.py`, `heads.py`, `capacity.py`, `dungeon_latent.py`, `generalize.py` |
| The Bayes gap | `fit_bayes.py`, `bayes_gap.py`, `transfer_test.py`, `update_deviations.py`, `paraphrase_analysis.py` |
| Interventions | `intervene.py`, `latent_edit.py`, `residual_edit.py`, `plan_latent.py` |
| Inside the latent | `explain.py`, `free_numbers.py`, `free_clusters.py`, `observers.py`, `surprise_gap.py`, `anchored_paraphrase.py`, `latent_map.py` |
| Dungeon analyses | `dungeon_learning.py`, `dungeon_stress.py`, `dungeon_dominance.py`, `dungeon_mirror.py`, `voice_analysis.py`, `think_pits.py`, `maze_feasibility.py`, `fov_feasibility.py` |
| Figures and the page | `figstyle.py`, `*_figure.py`, `plot_data.py`, `worked_example.py`, `project_page.py` |

## Running

Python 3.12 with the packages in `requirements.txt` (`pip install -r requirements.txt`); the models are downloaded
from the Hugging Face hub on first use. A clue-game collection with Qwen3.5-2B fits on an 8 GB GPU; the 9B and 27B
dungeon games were played on a 121 GB machine.

```
pytest test_game.py test_dungeon.py          # the rules of the two games
python collect.py --help                     # states and beliefs of an LLM on the clue game
python jepa.py --help                        # the anchored JEPA, forecasting
python play_dungeon.py --help                # whole dungeon games
python project_page.py                       # the project page's demo data, from the saved results
```

## Citation and licence

The code is under the MIT licence (`LICENSE`).


```bibtex
@unpublished{fumanalidocin2026belief,
  title  = {What does the model think it knows? A latent model of how an {LLM} updates its beliefs},
  author = {Fumanal-Idocin, Javier},
  year   = {2026},
  note   = {Working document}
}
```
