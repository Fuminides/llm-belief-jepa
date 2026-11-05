"""Pixel-art pictures of dungeon levels and replays of saved games, for the paper and the web (no model needed).

A level is redrawn from its config and seed, and a saved game is played again from its moves (the same slides:
see Dungeon.game), so any game in results/ can be shown. Tiles are 16 x 16 sprites drawn below as text in the
PICO-8 palette (one hex digit per pixel, "." transparent over the floor); the map is drawn at 1x and scaled up by
a whole factor, so pixels stay sharp.

  python -m games.pixelart results/dungeon/medium/Qwen__Qwen3.5-27B.json --game 4
  # writes results/pixel/<model>_<difficulty>_<policy>_<seed>.{png,gif}: the level with the path taken (PNG) and
  # the game turn by turn (GIF): what the player sees lit, what it has seen dimmed, the rest dark
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from games.dungeon import CONFIG, MOVES, PLAYERS, Dungeon, step

T = 16
PALETTE = ["000000", "1d2b53", "7e2553", "008751", "ab5236", "5f574f", "c2c3c7", "fff1e8",
           "ff004d", "ffa300", "ffec27", "00e436", "29adff", "83769c", "ff77a8", "ffccaa"]
SPRITES = {
    "floor": ["1111111101111111", "1111111101111111", "1151111101111111", "1111111101111511",
              "1111111101111111", "1111111101111111", "1111111101111111", "0000000000000000",
              "1111011111111101", "1111011111111101", "1111011111511101", "1111011111111101",
              "1511011111111101", "1111011111111101", "1111011111111101", "0000000000000000"],
    "wall": ["6666666066666660", "5555555055555550", "5555555055555550", "0000000000000000",
             "6660666666606666", "5550555555505555", "5550555555505555", "0000000000000000",
             "6666666066666660", "5555555055555550", "5555555055555550", "0000000000000000",
             "6660666666606666", "5550555555505555", "5550555555505555", "0000000000000000"],
    "column": ["................", "..666666666666..", "..777777777777..", "..555555555555..",
               "...6666666666...", "....76666655....", "....76666655....", "....76666655....",
               "....76666655....", "....76666655....", "....76666655....", "....76666655....",
               "...6666666666...", "..666666666666..", "..555555555555..", "...00000000000.."],
    "fire": ["................", ".......8........", "......88....8...", "......898..88...",
             ".....8998..898..", "..8..89a98.8998.", "..88.89a9989a98.", ".8988899a989a98.",
             ".89a989aa99aa98.", ".89aa99aaa9aa98.", ".89aaaa7aaaaa98.", "..89aa777aaa98..",
             "..899a777aa998..", "...8899a99988...", "....88888888....", "................"],
    "pit": ["................", ".....666666.....", "...6655555566...", "..655111111556..",
            "..651000000156..", ".65100000000156.", ".65000000000056.", ".65000000000056.",
            ".65000000000056.", ".65000000000056.", ".55000000000055.", "..550000000055..",
            "..555000000555..", "...5555555555...", ".....555555.....", "................"],
    "ice": ["cccccccccccccccc", "c77ccccccccc7ccc", "c7cccccccccc77cc", "ccccccc6cccccccc",
            "cccccc6ccccccccc", "ccccc6cccccccc7c", "cccccccccccccc7c", "ccc7cccccccccccc",
            "cc77cccccc6ccccc", "cccccccccc6ccccc", "ccccccccc6cccccc", "cccccccccccc7ccc",
            "c6ccccccccc77ccc", "6ccccccccccccccc", "cccccccccccccc6c", "cccccccccccccccc"],
    "exit": ["................", ".3bbbbbbbbbbbb3.", ".b666666666666b.", ".b666666666666b.",
             ".b000000000000b.", ".bddddddddddddb.", ".bddddddddddddb.", ".b000000000000b.",
             ".b555555555555b.", ".b555555555555b.", ".b000000000000b.", ".b111111111111b.",
             ".b111111111111b.", ".b000000000000b.", ".3bbbbbbbbbbbb3.", "................"],
    "door": ["0000000000000000", "0444244424442440", "0444244424442440", "0444244424442440",
             "0444244424442440", "0444244424442440", "0444244424442440", "0444244424449440",
             "0444244424449440", "0444244424442440", "0444244424442440", "0444244424442440",
             "0444244424442440", "0444244424442440", "0444244424442440", "0000000000000000"],
    "player": ["................", "......4444......", ".....444444..a..", ".....4ffff4.a9a.",
               ".....f0ff0f..9..", ".....ffffff..4..", "......ffff...4..", "....cccccccc.f..",
               "...cccccccccff..", "...fcccccccc....", "...f.cccccc.....", ".....cccccc.....",
               ".....11..11.....", ".....11..11.....", "....444..444....", "................"],
    "heart": ["................", "................", "................", "................",
              "....88...88.....", "...8878.8888....", "...8788888888...", "...8888888888...",
              "....88888888....", ".....888888.....", "......8888......", ".......88.......",
              "................", "................", "................", "................"],
}


def sprite(name):
    """The sprite as an RGBA image ("." transparent)."""
    rows = SPRITES[name]
    assert len(rows) == T and all(len(r) == T for r in rows), name
    px = np.zeros((T, T, 4), np.uint8)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch != ".":
                px[y, x] = [*bytes.fromhex(PALETTE[int(ch, 16)]), 255]
    return Image.fromarray(px, "RGBA")


TILES = {name: sprite(name) for name in SPRITES}


def tile(name):
    """A terrain's tile: its sprite over the floor, or its map symbol on magenta if it has no sprite."""
    out = TILES["floor"].copy()
    if name in TILES:
        out.alpha_composite(TILES[name])
    return out


def level_image(d, level, light=None, seen=None, path=None):
    """The level at 1x: cells in light (a set) at full brightness, seen cells dimmed, the rest dark (None: all lit);
    path, a list of cells, drawn as a line through their centres."""
    img = Image.new("RGBA", (d.cols * T, d.rows * T), (0, 0, 0, 255))
    for r in range(d.rows):
        for c in range(d.cols):
            t = tile(level[r, c]) if level[r, c] in TILES else Image.new("RGBA", (T, T), (255, 0, 255, 255))
            if light is not None and (r, c) not in light:
                if seen is None or (r, c) not in seen:
                    continue
                px = np.asarray(t).astype(float)
                px[..., :3] = px[..., :3] * 0.3 + px[..., :3].mean(-1, keepdims=True) * 0.1
                t = Image.fromarray(px.astype(np.uint8), "RGBA")
            img.paste(t, (c * T, r * T))
    if path:
        draw = ImageDraw.Draw(img)
        xy = [(c * T + T // 2, r * T + T // 2) for r, c in path]
        draw.line(xy, fill=(255, 236, 39, 255), width=2)
        for (x, y), fill in ((xy[0], (0, 228, 54, 255)), (xy[-1], (255, 0, 77, 255))):
            draw.rectangle([x - 2, y - 2, x + 2, y + 2], fill=fill)
    return img


def replay_states(d, level, moves):
    """Play the moves again as Dungeon.game does (same slides): per turn, the state before the move and the
    events of the move."""
    import copy
    rng = np.random.default_rng([level.seed, 0])
    state, out = d.start(level), []
    for move in moves:
        before = copy.copy(state)
        out.append((before, d.act(level, state, move, rng)))
    return out, state


FONT = ImageFont.load_default_imagefont()
ARROWS = {"up": "^", "down": "v", "left": "<", "right": ">"}


INK, DIM, NAVY, YELLOW = (255, 241, 232), (131, 118, 156), (29, 43, 83), (255, 236, 39)


def hud(d, state, p, move, width):
    """The bar above the map: lives as hearts, the turn and the torch left; below, the probability of each move,
    the move made in yellow."""
    img = Image.new("RGBA", (width, 2 * T), NAVY + (255,))
    draw = ImageDraw.Draw(img)
    for i in range(d.player["lives"]):
        heart = TILES["heart"] if i < state.lives else Image.fromarray(
            (np.asarray(TILES["heart"]) * [0.3, 0.3, 0.3, 1]).astype(np.uint8), "RGBA")
        img.alpha_composite(heart, (i * 11 - 2, -4))
    right = width - 3
    if state.turns_left is not None:
        draw.rectangle([right - 40, 5, right, 10], outline=(95, 87, 79))
        draw.rectangle([right - 40, 5, right - 40 + round(40 * state.turns_left / d.player["turns"]), 10], fill=(255, 163, 0))
        right -= 44
    turn = f"turn {state.turn}"
    draw.text((right - draw.textlength(turn, font=FONT), 1), turn, font=FONT, fill=INK)
    if p is not None:
        slot = width // 4
        for k, (m, q) in enumerate(zip(MOVES, p)):
            x, colour = k * slot + 3, YELLOW if m == move else DIM
            draw.text((x, 17), ARROWS[m], font=FONT, fill=colour)
            draw.rectangle([x + 8, 21, x + slot - 6, 26], outline=(95, 87, 79))
            draw.rectangle([x + 8, 21, x + 8 + round((slot - 14) * q), 26], fill=colour)
    return img


def caption(text, width):
    img = Image.new("RGBA", (width, 13), NAVY + (255,))
    ImageDraw.Draw(img).text((3, 0), text, font=FONT, fill=DIM)
    return img


def banner(img, text):
    """The text in a box over the middle of the image."""
    draw = ImageDraw.Draw(img)
    w = draw.textlength(text, font=FONT)
    x, y = (img.width - w) // 2, img.height // 2 - 6
    draw.rectangle([x - 5, y - 4, x + w + 4, y + 14], fill=NAVY + (255,), outline=YELLOW)
    draw.text((x, y), text, font=FONT, fill=YELLOW)


def ending(d, game):
    if game["outcome"] == "exit":
        return "escaped!"
    cause = d.terrain[game["cause"]]["name"] if game["cause"] else None
    return {"lethal": f"fell into a {cause}", "lives": f"no lives left ({cause}s)", "time": "the torch burned out",
            "cap": "out of turns"}[game["outcome"]]


def stack(*parts):
    out = Image.new("RGBA", (parts[0].width, sum(p.height for p in parts)))
    y = 0
    for part in parts:
        out.paste(part, (0, y))
        y += part.height
    return out


def replay_frames(d, level, game, label, tween=4, hold=12):
    """The game as frames at 1x: per turn the map lit by what the player sees, then the move in tween steps; at
    the end, how the game ended, held for hold frames."""
    moves = [t["move"] for t in game["log"]]
    turns, final = replay_states(d, level, moves)
    assert [tuple(s.cell) for s, _ in turns] == [tuple(t["cell"]) for t in game["log"]], "the replay left the saved game"
    seen, frames = set(), []
    bottom = caption(label, d.cols * T)
    for (state, events), saved in zip(turns, game["log"]):
        light = d.visible(level, state.cell)
        seen |= light
        base = level_image(d, level, light, seen)
        # where the player is drawn along the move: the cells it passes, or a nudge toward what it bumped into
        cells = [state.cell]
        for e in events:
            if e["type"] in ("move", "slide"):
                for _ in range(e.get("cells", 1)):
                    cells.append(step(cells[-1], e["move"]))
        bump = next((e for e in events if e["type"] == "bump"), None)
        positions = [(cells[0][1] * T, cells[0][0] * T)] * 2
        for a, b in zip(cells, cells[1:]):
            positions += [(a[1] * T + (b[1] - a[1]) * T * k // tween, a[0] * T + (b[0] - a[0]) * T * k // tween)
                          for k in range(1, tween + 1)]
        if bump:
            dr, dc = MOVES[bump["move"]]
            positions += [(cells[0][1] * T + dc * 4, cells[0][0] * T + dr * 4), (cells[0][1] * T, cells[0][0] * T)]
        for i, (x, y) in enumerate(positions):
            frame = base.copy()
            player = TILES["player"]
            last = i == len(positions) - 1
            if last and any(e["type"] == "damage" for e in events):
                player = Image.fromarray((np.asarray(player) * [1, 0.35, 0.35, 1]).astype(np.uint8), "RGBA")
            if last and any(e["type"] == "lethal" for e in events):
                player = player.resize((T // 2, T // 2), Image.NEAREST)
                x, y = x + T // 4, y + T // 4
            frame.alpha_composite(player, (x, y))
            frames.append(stack(hud(d, state, saved.get("p"), saved["move"], frame.width), frame, bottom))
    if final.outcome == "exit" or final.outcome in ("time", "cap"):  # alive: lit from where it ended
        seen |= d.visible(level, final.cell)
        frame = level_image(d, level, d.visible(level, final.cell), seen)
        frame.alpha_composite(TILES["player"], (final.cell[1] * T, final.cell[0] * T))
    else:
        frame = frames[-1].crop((0, 2 * T, d.cols * T, 2 * T + d.rows * T))
    banner(frame, ending(d, game))
    frames += [stack(hud(d, final, None, None, frame.width), frame, bottom)] * hold
    return frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="?", help="a saved games file (play_dungeon.py or think_dungeon.py)")
    p.add_argument("--policy", default=None, help="which games in the file (default: the first)")
    p.add_argument("--game", type=int, default=0, help="index of the game in the file")
    p.add_argument("--config", default=None, help="without results: a scripted player's game on this config")
    p.add_argument("--difficulty", default=None)
    p.add_argument("--player", default="explorer", choices=PLAYERS)
    p.add_argument("--seed", type=int, default=0, help="the level of the scripted player's game")
    p.add_argument("--scale", type=int, default=3)
    p.add_argument("--out", default="results/pixel")
    args = p.parse_args()
    if args.results:
        r = json.loads(Path(args.results).read_text())
        policy = args.policy or next(iter(r["results"]))
        game = r["results"][policy]["games"][args.game]
        d = Dungeon(r["config"])
        level = d.generate(game["seed"])
        label, name = f"{r['model'].split('/')[-1]}, {policy}", f"{r['model'].split('/')[-1]}_{r.get('difficulty')}_{policy}"
    else:
        d = Dungeon(args.config or CONFIG, args.difficulty)
        level = d.generate(args.seed)
        game = d.play(level, PLAYERS[args.player](d, args.seed))
        label, name = args.player, f"{Path(args.config or CONFIG).stem}_{args.difficulty}_{args.player}"
    name += f"_{game['seed']}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    big = lambda im: im.resize((im.width * args.scale, im.height * args.scale), Image.NEAREST)
    path = [tuple(t["cell"]) for t in game["log"]] + [tuple(game["final"])]
    big(level_image(d, level, path=path)).save(out / f"{name}.png")
    frames = [big(f).convert("RGB") for f in replay_frames(d, level, game, f"{label}, level {game['seed']}")]
    frames[0].save(out / f"{name}.gif", save_all=True, append_images=frames[1:], duration=70, loop=0)
    print(f"{game['outcome']} in {game['turns']} turns; wrote {out / name}.png and .gif ({len(frames)} frames)")


if __name__ == "__main__":
    main()
