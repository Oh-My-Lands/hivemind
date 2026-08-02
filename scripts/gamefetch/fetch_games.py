#!/usr/bin/env python3
"""Fetch per-move clocks for bughouse games from chess.com.

The explorer's games.db has 1.49M discovered bughouse games but stores only
`tcn` -- no `moveTimestamps` and no `partnerGameId`. Both are required here:
without timestamps there are no clocks to derive a sit margin from, and without
the partner id the two boards cannot be paired into a game at all. Only the
per-game callback endpoint carries them, so every game needs a refetch.

The DB is therefore used as a *worklist*, not a corpus -- the archive crawl
that found these 1.49M ids is the one expensive thing that does not have to be
repeated.

Two modes:

    seed    games.db  ->  candidates.txt   (run once, needs the DB)
    fetch   candidates.txt -> games.jsonl  (long-running, resumable)

Output is append-only JSONL rather than parquet on purpose. The existing
`download_games()` in src/scrape_cc.py re-reads and rewrites the whole parquet
every 100 games, which is O(n^2) and loses everything if it dies mid-write.
Appending one line per board is crash-safe, needs no dependencies beyond the
stdlib plus cloudscraper, and converts to parquet in one pass afterwards with
jsonl_to_parquet.py.

Rate limiting is deliberate. chess.com's callback endpoint sits behind
Cloudflare and is reached through cloudscraper; getting blocked costs far more
than the hours saved by hammering it. One request per second, single threaded,
matching what src/scrape_cc.py already does.
"""

import argparse
import json
import os
import random
import sqlite3
import sys
import time
from typing import Dict, Iterable, List, Optional, Set

CALLBACK_URL = "https://www.chess.com/callback/live/game/{game_id}"

# Time controls written as a bare integer of seconds. Anything with an
# increment ("180+2") fails `int(row["time_control"])` in TrainingGameReader
# and the game is silently dropped downstream, so there is no point spending
# two seconds of rate limit fetching it.
DEFAULT_TIME_CONTROLS = "180,120"


# ---------------------------------------------------------------------------
# seed
# ---------------------------------------------------------------------------

def seed_from_db(db_path: str, out_path: str, min_rating: int,
                 time_controls: Set[str], order: str, seed: int) -> int:
    """Extract candidate game ids from the explorer DB into a flat text file.

    Kept separate from the fetch so the long-running service never touches the
    DB: it runs under a strict systemd sandbox, and sqlite wants to create
    -wal/-shm siblings next to the database even for reads.
    """
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)

    placeholders = ",".join("?" for _ in time_controls)
    query = f"""
        SELECT url, end_time
        FROM games
        WHERE white_rating >= ? AND black_rating >= ?
          AND time_control IN ({placeholders})
    """
    params = [min_rating, min_rating, *sorted(time_controls)]

    rows = conn.execute(query, params).fetchall()
    conn.close()

    # The id is the last path segment of the game url.
    candidates = []
    for url, end_time in rows:
        if not url:
            continue
        game_id = str(url).rstrip("/").split("/")[-1]
        if game_id.isdigit():
            candidates.append((game_id, end_time or 0))

    if order == "recent":
        candidates.sort(key=lambda pair: pair[1], reverse=True)
    elif order == "random":
        random.Random(seed).shuffle(candidates)
    else:
        raise ValueError(f"unknown order: {order}")

    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as handle:
        for game_id, _ in candidates:
            handle.write(game_id + "\n")
    os.replace(tmp_path, out_path)

    return len(candidates)


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

def load_ids(path: str, field: Optional[str] = None) -> Set[str]:
    """Ids already accounted for, so a restart resumes instead of refetching."""
    seen: Set[str] = set()
    if not os.path.exists(path):
        return seen
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if field is None:
                seen.add(line)
                continue
            try:
                seen.add(str(json.loads(line)[field]))
            except (json.JSONDecodeError, KeyError):
                # A torn final line from a kill mid-write. Skipping it means at
                # worst one game is fetched twice.
                continue
    return seen


def iter_candidates(path: str) -> Iterable[str]:
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield line


class Fetcher:
    def __init__(self, sleep_seconds: float, max_retries: int, timeout: int):
        try:
            import cloudscraper
        except ImportError:
            sys.exit(
                "cloudscraper is required (the callback endpoint is behind "
                "Cloudflare).\n  pip install cloudscraper"
            )
        self.scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "desktop": True}
        )
        self.sleep_seconds = sleep_seconds
        self.max_retries = max_retries
        self.timeout = timeout
        self.requests = 0

    def get_game(self, game_id: str) -> Optional[dict]:
        """One callback fetch, with backoff. None means give up on this id."""
        for attempt in range(self.max_retries):
            try:
                response = self.scraper.get(
                    CALLBACK_URL.format(game_id=game_id), timeout=self.timeout
                )
            except Exception as exc:  # network, TLS, Cloudflare challenge
                wait = self.sleep_seconds * (2 ** attempt)
                print(f"  {game_id}: {type(exc).__name__}, retry in {wait:.0f}s",
                      flush=True)
                time.sleep(wait)
                continue

            self.requests += 1

            if response.status_code == 200:
                time.sleep(self.sleep_seconds)
                try:
                    return response.json()
                except ValueError:
                    return None

            if response.status_code == 404:
                return None  # deleted or never existed; not worth retrying

            # 429 and 403 both mean back off hard rather than press on.
            wait = max(30.0, self.sleep_seconds * (4 ** (attempt + 1)))
            print(f"  {game_id}: HTTP {response.status_code}, "
                  f"backing off {wait:.0f}s", flush=True)
            time.sleep(wait)

        return None


def board_row(wrapper: dict) -> Optional[Dict]:
    """Flatten one board's callback payload into the training parquet schema.

    Column names and the time_control format match what
    src/utils/game_reader.TrainingGameReader and process_parquet_file expect.
    """
    game = wrapper.get("game") or {}
    players = wrapper.get("players") or {}

    top = players.get("top") or {}
    bottom = players.get("bottom") or {}
    if top.get("color") == "white":
        white_p, black_p = top, bottom
    else:
        white_p, black_p = bottom, top

    move_list = game.get("moveList")
    timestamps = game.get("moveTimestamps")
    if not move_list or not timestamps:
        return None
    if white_p.get("rating") is None or black_p.get("rating") is None:
        return None

    try:
        base = int(game["baseTime1"] // 10)
        increment = int(game["timeIncrement1"] // 10)
    except (KeyError, TypeError):
        return None

    return {
        "game_id": str(game.get("id")),
        "partner_game_id": str(game.get("partnerGameId")),
        "tcn": move_list,
        "timestamps": timestamps,
        "winner": game.get("colorOfWinner", ""),
        "white_user": white_p.get("username"),
        "white_rating": int(white_p["rating"]),
        "black_user": black_p.get("username"),
        "black_rating": int(black_p["rating"]),
        "time_control": str(base) + (f"+{increment}" if increment > 0 else ""),
    }


def run_fetch(args: argparse.Namespace) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    done = load_ids(args.out, field="game_id")
    failed = load_ids(args.failed)
    pairs_before = len(done) // 2

    print(f"resuming: {len(done)} boards on disk (~{pairs_before} games), "
          f"{len(failed)} ids previously abandoned", flush=True)

    if pairs_before >= args.target_games:
        print(f"target of {args.target_games} games already met; nothing to do",
              flush=True)
        return 0

    fetcher = Fetcher(args.sleep, args.max_retries, args.timeout)
    pairs_written = 0
    started = time.time()

    out_handle = open(args.out, "a", buffering=1)       # line buffered
    failed_handle = open(args.failed, "a", buffering=1)

    def abandon(game_id: str, why: str) -> None:
        failed_handle.write(game_id + "\n")
        print(f"  {game_id}: skipped ({why})", flush=True)

    try:
        for game_id in iter_candidates(args.candidates):
            if pairs_before + pairs_written >= args.target_games:
                break
            if game_id in done or game_id in failed:
                continue

            payload = fetcher.get_game(game_id)
            if payload is None:
                abandon(game_id, "no payload")
                continue

            partner_id = str((payload.get("game") or {}).get("partnerGameId") or "")
            if not partner_id or partner_id == "None":
                abandon(game_id, "no partner id")
                continue

            partner_payload = fetcher.get_game(partner_id)
            if partner_payload is None:
                abandon(game_id, "partner unavailable")
                continue

            rows = [board_row(payload), board_row(partner_payload)]
            if any(row is None for row in rows):
                abandon(game_id, "incomplete board data")
                continue

            # process_parquet_file requires >= min_rating on all four players,
            # so a pair failing it would be dropped later anyway.
            ratings = [r for row in rows for r in (row["white_rating"], row["black_rating"])]
            if min(ratings) < args.min_rating:
                abandon(game_id, f"partner board below {args.min_rating}")
                continue

            # Point each board at its counterpart's game_id. The callback's
            # partnerGameId is a UUID while game_id is the numeric live-game id,
            # so as returned the two never reconcile -- and both consumers need
            # them to. jsonl_to_parquet filters partner_game_id.is_in(game_ids)
            # and process_parquet_file self-joins on it, so leaving the UUID in
            # place drops every row at conversion time while still exiting 0.
            # The pairing is unambiguous here: these two payloads were fetched
            # as a pair.
            rows[0]["partner_game_id"] = rows[1]["game_id"]
            rows[1]["partner_game_id"] = rows[0]["game_id"]

            for row in rows:
                out_handle.write(json.dumps(row) + "\n")
                done.add(row["game_id"])

            pairs_written += 1

            if pairs_written % args.log_every == 0:
                elapsed = time.time() - started
                rate = pairs_written / elapsed if elapsed else 0.0
                total = pairs_before + pairs_written
                remaining = args.target_games - total
                eta_h = (remaining / rate / 3600) if rate else float("inf")
                print(f"{total}/{args.target_games} games "
                      f"({rate * 3600:.0f}/h, ~{eta_h:.1f}h left, "
                      f"{fetcher.requests} requests)", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted; progress is on disk, rerun to resume", flush=True)
        return 130
    finally:
        out_handle.close()
        failed_handle.close()

    total = pairs_before + pairs_written
    print(f"done: {total} games in {args.out} "
          f"(+{pairs_written} this run, {fetcher.requests} requests)", flush=True)
    return 0 if total >= args.target_games else 1


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    seed = sub.add_parser("seed", help="extract candidate ids from games.db")
    seed.add_argument("--db", default="/opt/bughouse/data/games.db")
    seed.add_argument("--out", default="data/candidates.txt")
    seed.add_argument("--min-rating", type=int, default=2200)
    seed.add_argument("--time-controls", default=DEFAULT_TIME_CONTROLS,
                      help=f"comma separated, bare seconds (default {DEFAULT_TIME_CONTROLS})")
    seed.add_argument("--order", choices=["recent", "random"], default="recent")
    seed.add_argument("--seed", type=int, default=0, help="rng seed for --order random")

    fetch = sub.add_parser("fetch", help="fetch callbacks for candidate ids")
    fetch.add_argument("--candidates", default="data/candidates.txt")
    fetch.add_argument("--out", default="data/games.jsonl")
    fetch.add_argument("--failed", default="data/failed.txt")
    fetch.add_argument("--target-games", type=int, default=10000,
                       help="stop once this many paired games are on disk")
    fetch.add_argument("--min-rating", type=int, default=2200)
    fetch.add_argument("--sleep", type=float, default=1.0,
                       help="seconds between requests; do not lower casually")
    fetch.add_argument("--max-retries", type=int, default=4)
    fetch.add_argument("--timeout", type=int, default=20)
    fetch.add_argument("--log-every", type=int, default=25)

    args = parser.parse_args()

    if args.mode == "seed":
        controls = {c.strip() for c in args.time_controls.split(",") if c.strip()}
        count = seed_from_db(args.db, args.out, args.min_rating, controls,
                             args.order, args.seed)
        print(f"wrote {count} candidate ids to {args.out}")
        return 0

    return run_fetch(args)


if __name__ == "__main__":
    sys.exit(main())
