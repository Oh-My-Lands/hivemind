"""
Parsing for the Hivemind engine's UCI output.

Shared by the serverless handler and the dev server so the two cannot drift.
Deliberately dependency-free: it is copied flat into /app by the Dockerfile.

Everything here is written against output captured from the built engine, not
from the UCI spec -- this engine's dialect is non-standard in ways that matter:

    info depth 12 multipv 2 score cp -6 q -0.0219 prior 0.4569 visits 21179 \
        nodes 47264 nps 13864 hashfull 10 tbhits 10512 time 3409 \
        pv (e2e4,pass) (e7e5,e2e4) (b1c3,e7e5)
    info string Early stopping: saved 593ms
    bestmove (d2d4,pass)

Three things to know:

  - Moves are joint actions over both boards. "pass" is a real action (sit),
    not a missing move, so it maps to None rather than being dropped.
  - `info string` diagnostics are interleaved with PV lines.
  - `q`, `prior` and `visits` are additions this fork makes. `score cp` comes
    from a tangent transform that saturates hard, so `q` is the honest number
    and `visits` is what says whether to trust it.
"""

import re
from typing import Any, Dict, List, Optional

_INT_FIELDS = {
    "depth", "multipv", "visits", "nodes", "nps", "hashfull", "tbhits", "time",
}
_FLOAT_FIELDS = {"q", "prior"}

# "(d2d4,pass)" or "(g5e7,P@e6)" -- drops use Fairy-Stockfish's piece@square form.
_JOINT_ACTION = re.compile(r"\(([^,()]*),([^,()]*)\)")

_PASS = "pass"


def move_or_none(token: str) -> Optional[str]:
    """Maps a move token to None when it means "sit"/absent."""
    token = token.strip()
    return None if token in ("", _PASS, "none", "(none)") else token


def parse_pv(text: str) -> List[Dict[str, Optional[str]]]:
    """Parses a PV of joint actions into per-ply {a, b} pairs."""
    return [
        {"a": move_or_none(a), "b": move_or_none(b)}
        for a, b in _JOINT_ACTION.findall(text)
    ]


def parse_info_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Parses one `info` line, or returns None if it is not a usable PV line.

    Unknown tokens are skipped rather than raising, so this keeps working when
    the engine grows a field it has not been taught.
    """
    if not line.startswith("info"):
        return None

    if line.startswith("info string"):
        return {"kind": "string", "message": line[len("info string"):].strip()}

    tokens = line.split()
    out: Dict[str, Any] = {"kind": "pv"}
    i = 1
    while i < len(tokens):
        token = tokens[i]

        if token == "pv":
            # `pv` is always last and consumes the rest of the line.
            out["pv"] = parse_pv(" ".join(tokens[i + 1:]))
            break

        if token == "score" and i + 2 < len(tokens):
            try:
                out["score"] = {"kind": tokens[i + 1], "value": int(tokens[i + 2])}
            except ValueError:
                pass
            i += 3
            continue

        if i + 1 < len(tokens) and token in _INT_FIELDS:
            try:
                out[token] = int(tokens[i + 1])
            except ValueError:
                pass
            i += 2
            continue

        if i + 1 < len(tokens) and token in _FLOAT_FIELDS:
            try:
                out[token] = float(tokens[i + 1])
            except ValueError:
                pass
            i += 2
            continue

        i += 1

    if "depth" not in out:
        return None
    out.setdefault("multipv", 1)
    return out


def parse_bestmove_line(line: str) -> Optional[Dict[str, Optional[str]]]:
    """Parses `bestmove (moveA,moveB)` into its two halves."""
    if not line.startswith("bestmove"):
        return None
    match = _JOINT_ACTION.search(line)
    if not match:
        return {"moveA": None, "moveB": None}
    return {
        "moveA": move_or_none(match.group(1)),
        "moveB": move_or_none(match.group(2)),
    }


def collect_final_lines(info_lines: List[str], analysis_board: int = 1) -> List[Dict[str, Any]]:
    """
    Reduces a whole search's worth of info lines to the settled result.

    The engine re-emits every PV slot as the search deepens, so the same
    `multipv` index appears many times. Only the last emission of each slot
    describes the finished search; earlier ones are superseded. Keyed by slot,
    last-wins, then ordered by rank.

    Each line gains a `move`: the candidate on the board being analysed, which
    is the half of the first PV entry belonging to that board. That is the move
    the lines are grouped by, and the one a UI should show.
    """
    by_slot: Dict[int, Dict[str, Any]] = {}

    for raw in info_lines:
        parsed = parse_info_line(raw)
        if parsed is None or parsed["kind"] != "pv":
            continue
        by_slot[parsed["multipv"]] = parsed

    lines: List[Dict[str, Any]] = []
    for slot in sorted(by_slot):
        entry = dict(by_slot[slot])
        entry.pop("kind", None)

        pv = entry.get("pv") or []
        first = pv[0] if pv else {}
        # analysis_board is 1-based: 1 = board A, 2 = board B.
        entry["move"] = first.get("a" if analysis_board == 1 else "b")
        entry["partnerMove"] = first.get("b" if analysis_board == 1 else "a")

        lines.append(entry)

    return lines
