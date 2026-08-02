#!/usr/bin/env python3
"""Convert fetched games.jsonl into the parquet that generate_planes.py reads.

Run this wherever polars is available -- it does not need to happen on the
fetching host. One pass, no incremental rewriting.

The output columns are exactly what `process_parquet_file` self-joins on and
what `TrainingGameReader` reads. Rows are per *board*, two per game, joined on
partner_game_id downstream.
"""

import argparse
import json
import os
import sys

import polars as pl

# Written by fetch_games.board_row; declared explicitly so a schema drift shows
# up here rather than as a confusing failure inside the plane generator.
SCHEMA = {
    "game_id": pl.Utf8,
    "partner_game_id": pl.Utf8,
    "tcn": pl.Utf8,
    "timestamps": pl.Utf8,
    "winner": pl.Utf8,
    "white_user": pl.Utf8,
    "white_rating": pl.Int64,
    "black_user": pl.Utf8,
    "black_rating": pl.Int64,
    "time_control": pl.Utf8,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jsonl", default="data/games.jsonl")
    parser.add_argument("--out", default="data/games.parquet")
    parser.add_argument("--drop-increment", action="store_true", default=True,
                        help="drop time controls with an increment; int() on "
                             "them fails in TrainingGameReader and the game is "
                             "silently skipped anyway (default: on)")
    args = parser.parse_args()

    if not os.path.exists(args.jsonl):
        sys.exit(f"no such file: {args.jsonl}")

    rows = []
    malformed = 0
    with open(args.jsonl) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                malformed += 1

    if not rows:
        sys.exit(f"{args.jsonl} contained no usable rows")

    df = pl.DataFrame(rows, schema=SCHEMA)
    before = len(df)

    # A torn write or a duplicate fetch can leave the same board twice.
    df = df.unique(subset="game_id", keep="first")

    if args.drop_increment:
        df = df.filter(~pl.col("time_control").str.contains(r"\+"))

    # Both boards must be present or the self-join drops the game silently.
    ids = set(df.get_column("game_id").to_list())
    df = df.filter(pl.col("partner_game_id").is_in(ids))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    df.write_parquet(args.out, compression="zstd")

    print(f"read      {before} board rows ({malformed} malformed lines skipped)")
    print(f"wrote     {len(df)} board rows (~{len(df) // 2} games) to {args.out}")
    if len(df) < before:
        print(f"dropped   {before - len(df)} (duplicate, increment, or unpaired)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
