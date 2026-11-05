import pytest
import copy
import json

import numpy as np

from games.dungeon import CONFIG, MOVES, Dungeon, Explorer, Greedy, Level, State, step

BASE = json.loads(CONFIG.read_text())


def dungeon_with(**changes):
    config = copy.deepcopy(BASE)
    for path, value in changes.items():
        *keys, last = path.split("__")
        node = config
        for key in keys:
            node = node[key]
        node[last] = value
    return Dungeon(config)


def line(**changes):
    """A dungeon of one room, one cell tall and eight wide: a 3 x 10 grid."""
    return dungeon_with(level__rooms=[1, 1], level__room_size=[1, 8], **changes)


def corridor(d, row):
    """A level whose grid is all wall except the middle row, written in terrain symbols."""
    by_symbol = {t["symbol"]: name for name, t in d.terrain.items()}
    grid = [["wall"] * d.cols for _ in range(d.rows)]
    grid[1] = [by_symbol[s] for s in row]
    return Level(grid, (1, 1), (1, row.index(">")), seed=0)


class Always:
    def random(self):
        return 0.0


class Never:
    def random(self):
        return 1.0


def test_line_of_sight_matches_fov_feasibility():
    from games import fov_feasibility as fov
    d = dungeon_with(level__rooms=[2, 2], level__room_size=[2, 2])  # 7 x 7, the size in terrain.json
    assert (d.rows, d.cols) == (fov.ROWS, fov.COLS)
    rng = np.random.default_rng(0)
    for _ in range(30):
        walls = rng.random((d.rows, d.cols)) < 0.3
        level = Level([["wall" if w else "floor" for w in row] for row in walls], (0, 0), (0, 0), seed=0)
        symbols = [[fov.WALL if w else "." for w in row] for row in walls]
        for cell in [(int(rng.integers(d.rows)), int(rng.integers(d.cols))) for _ in range(5)]:
            assert d.visible(level, cell) == fov.visible(symbols, cell)


def test_bump_spends_the_turn_and_fire_burns_every_turn():
    d = line()
    level = corridor(d, "#^..>#####")
    state = State(cell=(1, 1), lives=3, turns_left=10)
    events = d.act(level, state, "up", Never())
    assert state.cell == (1, 1) and state.turns_left == 9 and state.lives == 2
    assert [e["type"] for e in events] == ["bump", "damage"]
    d.act(level, state, "left", Never())
    d.act(level, state, "down", Never())
    assert state.outcome == "lives" and state.cause == "fire"


def test_pit_kills_and_exit_wins():
    d = line()
    state = State(cell=(1, 1), lives=3, turns_left=10)
    d.act(corridor(d, "#.O.>#####"), state, "right", Never())
    assert state.outcome == "lethal" and state.cause == "pit" and state.cell == (1, 2)
    state = State(cell=(1, 1), lives=3, turns_left=10)
    d.act(corridor(d, "#.>..#####"), state, "right", Never())
    assert state.outcome == "exit"


def test_ice_slides_one_cell_stops_at_walls_and_slides_into_pits():
    d = line()
    state = State(cell=(1, 1), lives=3, turns_left=10)
    events = d.act(corridor(d, "#.~..>####"), state, "right", Always())
    assert state.cell == (1, 3) and events[1] == {"type": "slide", "move": "right", "terrain": "ice", "cells": 1}
    state = State(cell=(1, 1), lives=3, turns_left=10)
    d.act(corridor(d, "#.~..>####"), state, "right", Never())
    assert state.cell == (1, 2)
    state = State(cell=(1, 2), lives=3, turns_left=10)
    events = d.act(corridor(d, "#>.~######"), state, "right", Always())  # a wall right after the ice: no slide
    assert state.cell == (1, 3) and [e["type"] for e in events] == ["move"]
    state = State(cell=(1, 1), lives=3, turns_left=10)
    d.act(corridor(d, "#.~O.>####"), state, "right", Always())
    assert state.outcome == "lethal" and state.cell == (1, 3)


def test_torch_burns_out():
    d = line()
    level = corridor(d, "#...>#####")
    state = State(cell=(1, 1), lives=3, turns_left=2)
    d.act(level, state, "left", Never())
    assert state.outcome is None
    d.act(level, state, "left", Never())
    assert state.outcome == "time"


def test_generated_levels_are_fair_and_reproducible():
    d = Dungeon()
    for seed in range(50):
        level = d.generate(seed)
        assert d.shortest(level) is not None
        assert level.exit not in d.visible(level, level.start)
        assert level[level.start] == "floor" and level[level.exit] == "exit"
        assert level.grid == d.generate(seed).grid


def test_explorer_wins_every_level():
    d = Dungeon()
    for seed in range(100):
        assert d.play(d.generate(seed), Explorer(d, seed))["outcome"] == "exit"


def test_observation_lists_rays_and_off_axis_things():
    d = Dungeon()
    level = d.generate(0)
    text = d.observe(level, level.start)
    lines = text.splitlines()
    assert lines[0] == "You see:" and [l.split(":")[0] for l in lines[1:5]] == ["Up", "Down", "Left", "Right"]
    assert lines[-1] == "The other cells in view are empty cells or walls."


def test_rules_follow_the_config():
    d = dungeon_with(player__lives=1, terrain__fire__damage=2)
    rules = d.rules()
    assert "You have 1 life," in rules and "Fire cells: you lose 2 lives for every turn you end on one." in rules


def test_first_moves_returns_every_tied_first_move():
    d = Dungeon()
    level = d.generate(0)
    open_ = lambda c: not d.blocks(level, c)
    assert d.first_moves((2, 2), {(1, 1)}, lambda c: True) == {"up", "left"}
    assert d.first_moves((2, 2), {(2, 2)}, open_) == set()  # the cell itself is not a target


def test_greedy_never_steps_back_unless_forced():
    d = Dungeon()
    for seed in range(30):
        game = d.play(d.generate(seed), Greedy(d, seed))
        cells = [tuple(t["cell"]) for t in game["log"]] + [game["final"]]
        for i in range(2, len(cells)):
            if cells[i] == cells[i - 2] and cells[i - 1] != cells[i]:
                level, here = d.generate(seed), cells[i - 1]
                others = [step(here, m) for m in MOVES if step(here, m) != cells[i]]
                assert all(d.blocks(level, c) or d.terrain[level[c]].get("lethal") for c in others)


def test_no_torch_and_the_cap_stops_the_game():
    d = dungeon_with(player__turns=None, max_turns=25)
    assert "torch" not in d.rules() and "Torch" not in d.status(d.start(d.generate(0)))
    game = d.play(d.generate(0), Stay())
    assert game["outcome"] == "cap" and game["turns"] == 25


class Stay:
    """Walks into the wall above the start or, if it can move up, back and forth."""

    def reset(self):
        self.n = 0

    def act(self, turn):
        self.n += 1
        return ("up" if self.n % 2 else "down"), {}


def test_difficulties_override_the_config_and_the_explorer_wins_them_all():
    easy, hard = Dungeon(difficulty="easy"), Dungeon(difficulty="hard")
    assert (easy.rows, easy.cols, easy.player["lives"]) == (6, 11, 5)
    assert (hard.rows, hard.cols, hard.player["lives"]) == (16, 16, 1)
    assert hard.terrain["ice"]["slip_probability"] == 0.75 and hard.terrain["ice"]["slip_cells"] == 1  # merged
    assert hard.config["level"]["room_size"] == [4, 4] and "difficulties" not in hard.config
    for d in (easy, hard):
        for seed in range(30):
            assert d.play(d.generate(seed), Explorer(d, seed))["outcome"] == "exit"


def test_voice_tells_the_truth_or_lies_as_configured():
    from games.dungeon import Heeding, pit_posterior
    for reliability in (1.0, 0.0):
        d = dungeon_with(advisor={"name": "a voice in the dark", "reliability": reliability, "chance": 1.0, "radius": 20})
        level = d.generate(seed=1)
        record = d.play(level, Heeding(d, seed=0), keep_text=True)
        advised = [t for t in record["log"] if "advice" in t]
        assert advised, "the voice never spoke"
        for t in advised:
            assert t["advice"]["claim"] == (t["advice"]["truth"] if reliability == 1.0 else not t["advice"]["truth"])
            assert "A voice in the dark says: there is " in t["text"]
        assert "it is right" in record["log"][0]["text"]
    assert pit_posterior(0.05, 0.7, True) == pytest.approx(0.05 * 0.7 / (0.05 * 0.7 + 0.95 * 0.3))
    assert pit_posterior(0.05, 0.7, False) == pytest.approx(0.05 * 0.3 / (0.05 * 0.3 + 0.95 * 0.7))


def test_without_a_voice_nothing_changes():
    d = Dungeon(BASE)
    level = d.generate(seed=1)
    record = d.play(level, Explorer(d, seed=0), keep_text=True)
    assert not any("advice" in t for t in record["log"])
    assert "voice" not in record["log"][0]["text"]


def test_one_step_risk_and_consequences():
    d = line()
    level = corridor(d, "#.~O^>####")
    state = State(cell=(1, 1), lives=3, turns_left=10)
    assert d.risk(level, state, "right") == {"die": 0.5, "hurt": 0.0}  # ice: half the time into the pit
    assert d.risk(level, state, "up") == {"die": 0.0, "hurt": 0.0}
    assert [d.consequence(level, state, m) for m in MOVES] == ["wall", "wall", "wall", "slide"]
    state = State(cell=(1, 3), lives=1, turns_left=10)  # fire with one life left kills
    assert d.risk(level, state, "right") == {"die": 1.0, "hurt": 0.0} and d.consequence(level, state, "right") == "damage"


def test_forbidden_word_disarms_hazards_or_does_nothing():
    for effect in (True, False):
        d = line(forbidden={"word": "omega", "effect": effect})
        level = corridor(d, "#.O.>#####")
        state = State(cell=(1, 1), lives=3, turns_left=10)
        events = d.act(level, state, "omega", Never())
        assert events == [{"type": "word", "effect": effect}] and state.cell == (1, 1) and state.turns_left == 9
        d.act(level, state, "right", Always())
        assert (state.outcome is None) == effect
        assert ('"omega"' in d.rules()) and ("harmless" in d.rules()) == effect


def test_hidden_effects_name_the_kinds_and_list_the_effects_without_pairing_them():
    d = Dungeon(difficulty="medium_cue_hidden")
    rules = d.rules()
    assert "fire cells, pits, ice cells and cracked cells" in rules and "nothing happens" in rules
    assert "Pits: if you step into one" not in rules and "if you step into one, you die" in rules
    assert [d.generate(s).grid for s in range(5)] == [Dungeon(difficulty="medium_cue").generate(s).grid for s in range(5)]


def test_cue_cells_sit_inside_doors_and_mark_dangerous_rooms():
    d = Dungeon(difficulty="medium_cue")
    (h, w) = d.config["level"]["room_size"]
    room_of = lambda c: ((c[0] - 1) // (h + 1), (c[1] - 1) // (w + 1))
    hazards = set(d.config["level"]["density"])
    cued = uncued = 0
    for seed in range(100):
        level = d.generate(seed)
        dangerous = {tuple(r) for r in level.meta["dangerous"]}
        for r in range(d.rows):
            for c in range(d.cols):
                if level[r, c] in hazards:
                    assert room_of((r, c)) in dangerous
                if level[r, c] == "cracked":
                    assert any(level[step((r, c), m)] == "door" for m in MOVES)
                    cued += room_of((r, c)) in dangerous
                    uncued += room_of((r, c)) not in dangerous
    assert cued > 5 * uncued


def test_the_ledge_is_a_shortcut_and_not_the_only_way():
    d = Dungeon(difficulty="stress_game")
    for seed in range(20):
        level = d.generate(seed)
        ledge = tuple(level.meta["ledge"])
        without = d.shortest(level, passable=lambda c: c != ledge and d.safe(level[c]))
        assert level[ledge] == "ledge" and without - d.shortest(level) >= 4


def test_mirror_image_plays_the_mirrored_game():
    from games.dungeon import MIRROR
    from games.play_dungeon import Replay
    d = Dungeon()
    for seed in range(10):
        level = d.generate(seed)
        game = d.play(level, Explorer(d, seed))
        flipped = d.play(d.mirror(level), Replay([MIRROR[t["move"]] for t in game["log"]]))
        assert flipped["outcome"] == game["outcome"] and flipped["turns"] == game["turns"]
        assert [(r, d.cols - 1 - c) for r, c in (t["cell"] for t in game["log"])] == [tuple(t["cell"]) for t in flipped["log"]]
