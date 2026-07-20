"""
Tests for the serverless handler's parsing and result assembly.

Parser tests use lines captured verbatim from the built engine. The end-to-end
tests need a GPU and the built binary, so they skip unless HIVEMIND_ENGINE is
set:

    pytest test_handler.py                                    # parsing only

    ENGINE_PATH=/workspace/hivemind/engine/build/hivemind \
    HIVEMIND_ENGINE=/workspace/hivemind/engine/build/hivemind \
      pytest test_handler.py                                  # everything
"""

import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uci_parse import (  # noqa: E402
    collect_final_lines,
    parse_bestmove_line,
    parse_info_line,
    parse_pv,
)

# Two depths of the same three-slot search, captured from the built engine.
DEPTH_11 = [
    "info depth 11 multipv 1 score cp -3 q -0.0137 prior 0.3265 visits 13218 "
    "nodes 28688 nps 13634 hashfull 6 tbhits 5960 time 2104 "
    "pv (d2d4,pass) (d7d5,e2e4)",
    "info depth 11 multipv 2 score cp -5 q -0.0204 prior 0.4569 visits 12588 "
    "nodes 28688 nps 13634 hashfull 6 tbhits 5960 time 2104 "
    "pv (e2e4,pass) (e7e5,e2e4)",
    "info depth 11 multipv 3 score cp -3 q -0.0119 prior 0.0368 visits 1718 "
    "nodes 28688 nps 13634 hashfull 6 tbhits 5960 time 2104 "
    "pv (b1c3,pass) (d7d5,e2e4)",
]
DEPTH_12 = [
    "info string Early stopping: saved 593ms",
    "info depth 12 multipv 1 score cp -4 q -0.0169 prior 0.3265 visits 21238 "
    "nodes 47264 nps 13864 hashfull 10 tbhits 10512 time 3409 "
    "pv (d2d4,pass) (d7d5,pass)",
    "info depth 12 multipv 2 score cp -6 q -0.0219 prior 0.4569 visits 21179 "
    "nodes 47264 nps 13864 hashfull 10 tbhits 10512 time 3409 "
    "pv (e2e4,pass) (e7e5,e2e4)",
    "info depth 12 multipv 3 score cp -4 q -0.0143 prior 0.0368 visits 3013 "
    "nodes 47264 nps 13864 hashfull 10 tbhits 10512 time 3409 "
    "pv (b1c3,pass) (d7d5,e2e4)",
]
ALL_LINES = DEPTH_11 + DEPTH_12


class TestCollectFinalLines:
    def test_keeps_only_the_last_emission_per_slot(self):
        lines = collect_final_lines(ALL_LINES, analysis_board=1)
        assert len(lines) == 3
        assert all(line["depth"] == 12 for line in lines), (
            "superseded depth-11 lines leaked into the settled result"
        )

    def test_orders_by_rank(self):
        lines = collect_final_lines(ALL_LINES, analysis_board=1)
        assert [line["multipv"] for line in lines] == [1, 2, 3]

    def test_extracts_the_move_on_the_analysed_board(self):
        lines = collect_final_lines(ALL_LINES, analysis_board=1)
        assert [line["move"] for line in lines] == ["d2d4", "e2e4", "b1c3"]
        # Board A is to move in all of these, so the partner is sitting.
        assert all(line["partnerMove"] is None for line in lines)

    def test_board_two_reads_the_other_half(self):
        lines = collect_final_lines(ALL_LINES, analysis_board=2)
        # Every PV opens with the partner passing, so board B has no move yet.
        assert all(line["move"] is None for line in lines)
        assert [line["partnerMove"] for line in lines] == ["d2d4", "e2e4", "b1c3"]

    def test_carries_the_numbers_a_ui_needs(self):
        best = collect_final_lines(ALL_LINES, analysis_board=1)[0]
        assert best["q"] == pytest.approx(-0.0169)
        assert best["visits"] == 21238
        assert best["prior"] == pytest.approx(0.3265)
        assert best["score"] == {"kind": "cp", "value": -4}

    def test_ignores_info_strings(self):
        assert len(collect_final_lines(["info string hello"], 1)) == 0

    def test_empty_input(self):
        assert collect_final_lines([], 1) == []

    def test_survives_a_truncated_trailing_line(self):
        # The engine can be killed mid-write; a partial line must not poison
        # the whole result.
        lines = collect_final_lines(ALL_LINES + ["info depth 13 multi"], 1)
        assert len(lines) == 3


class TestParsing:
    def test_pass_is_an_action_not_a_missing_move(self):
        assert parse_pv("(d2d4,pass)") == [{"a": "d2d4", "b": None}]

    def test_drop_moves(self):
        assert parse_pv("(g5e7,P@e6)") == [{"a": "g5e7", "b": "P@e6"}]

    def test_mate_score(self):
        line = ("info depth 5 multipv 1 score mate 3 q 0.99 prior 0.5 "
                "visits 90 time 7 pv (h5f7,pass)")
        assert parse_info_line(line)["score"] == {"kind": "mate", "value": 3}

    def test_bestmove(self):
        assert parse_bestmove_line("bestmove (d2d4,pass)") == {
            "moveA": "d2d4", "moveB": None,
        }

    def test_unknown_fields_are_skipped(self):
        line = DEPTH_12[1].replace("time 3409", "wibble 7 time 3409")
        assert parse_info_line(line)["time"] == 3409


# --------------------------------------------------------------------- e2e

ENGINE = os.environ.get("HIVEMIND_ENGINE")
requires_engine = pytest.mark.skipif(
    not ENGINE or not os.path.isfile(ENGINE),
    reason="set HIVEMIND_ENGINE to the built binary for end-to-end tests",
)

START = ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1|"
         "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1")

HERE = os.path.dirname(os.path.abspath(__file__))


def _run_handler(payload: dict) -> dict:
    """
    Runs one job in a fresh subprocess.

    Out of process on purpose: the handler holds a module-level engine, and a
    stale one leaking between tests would hide exactly the kind of state bug
    worth catching.
    """
    script = (
        "import json,sys; sys.path.insert(0, %r);"
        "import handler;"
        "print('@@@'+json.dumps(handler.handler({'input': %s})))"
        % (HERE, json.dumps(payload))
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=600,
        cwd=os.environ.get("HIVEMIND_ENGINE_CWD", os.getcwd()),
    )
    for line in proc.stdout.splitlines():
        if line.startswith("@@@"):
            return json.loads(line[3:])
    raise AssertionError(f"no result\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")


@requires_engine
def test_returns_ranked_lines():
    out = _run_handler({
        "fen": START, "movetime": 4000, "multipv": 4, "analysisBoard": 1,
    })
    assert "error" not in out, out
    lines = out["lines"]
    assert lines, "no ranked lines returned"

    # The whole point of grouping: one entry per distinct move on our board.
    moves = [line["move"] for line in lines]
    assert len(set(moves)) == len(moves), f"duplicate candidate moves: {moves}"

    # PV 1 must agree with bestmove, or the UI contradicts itself.
    assert lines[0]["move"] == out["moveA"]

    for line in lines:
        assert -1.0 <= line["q"] <= 1.0
        assert line["visits"] > 0
        assert 0.0 <= line["prior"] <= 1.0
        assert line["pv"]


@requires_engine
def test_node_budget_is_honoured():
    out = _run_handler({"fen": START, "nodes": 20000, "multipv": 2})
    assert "error" not in out, out
    # Overshoot is batch granularity across parallel workers, not drift.
    assert 20000 <= out["nodes"] <= 20000 * 1.2


@requires_engine
def test_reports_when_fewer_lines_than_requested():
    # A tiny budget cannot widen the root to 50 distinct moves.
    out = _run_handler({"fen": START, "nodes": 500, "multipv": 50})
    assert "error" not in out, out
    assert len(out["lines"]) < 50
    assert "note" in out, "should say why it returned fewer lines"


@requires_engine
def test_rejects_bad_analysis_board():
    assert "error" in _run_handler({"fen": START, "analysisBoard": 3, "nodes": 100})
