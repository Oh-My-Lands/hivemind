#!/usr/bin/env python3
"""Split a tournament PGN from `hivemind eval` into per-game BPGN files.

The engine already writes BPGN move text -- `1A. e4 {39.1} 1a. e6 {39.1}`, both
boards interleaved in the order the moves actually happened, with each brace
holding that player's clock in seconds *after* the move. So this is not a
conversion so much as three small repairs, plus splitting a 2400-game file into
things a viewer will accept one at a time.

Aimed at github.com/bmacho/bughouse-viewer, whose `bug.js` is a real BPGN
parser. Its page takes a chess.com link rather than pasted BPGN, so load a file
produced here from the browser console:

    v1.reloadgame(`<paste file contents>`, "", "")

What gets repaired:

1. `[TimeControl "40"]` -> `"40+0"`. bug.js does `timecontrol.split('+')` and
   `parseInt(tmp[1])`, so a bare number leaves the increment NaN.
2. `[Time]` and the four *Elo tags are added, because the viewer's own
   generator emits them and some code paths read them.
3. Games are emitted one per file.

Not repaired, because it is real: clocks can go slightly negative ({-0.5}). A
flag is checked at the start of a turn, so a player holding 0.1s still gets a
full move and overruns. See engine/CLOCK_AB_BENCHMARK.md, "Caveats".

    python3 scripts/pgn_to_bpgn.py gate.pgn out/ --only-flagged
"""
import argparse
import re
from pathlib import Path

HEADER = re.compile(r'^\[(\w+)\s+"(.*)"\]$')


def split_games(text):
    """Yields (headers, movetext) per game, in file order."""
    games, headers, moves = [], {}, []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        m = HEADER.match(line)
        if m:
            # A header line after movetext means the previous game is complete.
            if moves:
                games.append((headers, moves))
                headers, moves = {}, []
            headers[m.group(1)] = m.group(2)
        else:
            moves.append(line)
    if moves:
        games.append((headers, moves))
    return games


def to_bpgn(headers, moves):
    h = dict(headers)

    tc = h.get("TimeControl", "")
    if tc and "+" not in tc:
        h["TimeControl"] = f"{tc}+0"

    h.setdefault("Time", "00:00:00")
    for tag in ("WhiteA", "BlackA", "WhiteB", "BlackB"):
        h.setdefault(tag, "?")
        h.setdefault(f"{tag}Elo", "0")

    order = ["Event", "Site", "Date", "Time", "Round", "Variant",
             "WhiteA", "WhiteAElo", "BlackA", "BlackAElo",
             "WhiteB", "WhiteBElo", "BlackB", "BlackBElo",
             "TimeControl", "Result", "Termination", "WhiteTeam", "BlackTeam"]
    lines = [f'[{k} "{h[k]}"]' for k in order if k in h]
    lines += [f'[{k} "{v}"]' for k, v in h.items() if k not in order]
    return "\n".join(lines) + "\n\n" + " ".join(moves) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pgn", type=Path)
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--only-flagged", action="store_true",
                    help="keep only games decided on the clock -- the ones worth watching")
    ap.add_argument("--limit", type=int, default=0, help="stop after N games")
    args = ap.parse_args()

    games = split_games(args.pgn.read_text())
    args.outdir.mkdir(parents=True, exist_ok=True)

    written = 0
    for headers, moves in games:
        if args.only_flagged and "on time" not in headers.get("Termination", ""):
            continue
        rnd = headers.get("Round", str(written + 1))
        (args.outdir / f"game-{int(rnd):04d}.bpgn").write_text(to_bpgn(headers, moves))
        written += 1
        if args.limit and written >= args.limit:
            break

    print(f"{len(games)} games in {args.pgn}, wrote {written} to {args.outdir}/")


if __name__ == "__main__":
    main()
