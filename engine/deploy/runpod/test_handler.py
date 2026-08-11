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


class TestSearchTimeout:
    """
    The bestmove backstop, which had no test and was wrong for every node
    search: it took MAX_MOVE_TIME_MS as the budget, so 20k and 4M alike waited
    60s and then raised TimeoutError. That put an invisible ceiling at ~430-490k
    nodes on the serving GPU and turned a reachable budget into an HTTP 502.
    """

    def test_scales_with_the_node_budget(self):
        import handler as handler_module

        # The number that mattered: 1M nodes used to get 60s and could not
        # finish in it. At the 6,000 nps floor it now gets ~197s.
        assert handler_module.search_timeout_seconds(nodes=1_000_000) == pytest.approx(196.7, abs=0.5)
        # And the budget that silently never worked.
        assert handler_module.search_timeout_seconds(nodes=500_000) > 60

    def test_covers_the_measured_worst_case_throughput(self):
        import handler as handler_module

        # 7,221 nps is the slowest search measured on the serving GPU. Every
        # budget the caller's proxy permits has to fit inside its own timeout at
        # that rate, or the backstop is again the thing that fails the search.
        worst_case_nps = 7_221
        for nodes in (20_000, 200_000, 500_000, 1_000_000, 4_000_000):
            assert handler_module.search_timeout_seconds(nodes=nodes) > nodes / worst_case_nps

    def test_a_movetime_search_is_unchanged(self):
        import handler as handler_module

        # Only the node path was wrong; movetime already knew its own duration.
        assert handler_module.search_timeout_seconds(movetime=30_000) == 60
        assert handler_module.search_timeout_seconds() == pytest.approx(31.0)

    def test_movetime_wins_when_both_are_given(self):
        import handler as handler_module

        # `go` sends movetime in that case, so the timeout has to follow the
        # command actually issued rather than the larger of the two.
        assert handler_module.search_timeout_seconds(movetime=1_000, nodes=4_000_000) == 31


class TestParseClocks:
    """
    The `clocks` quartet's validation.

    This is the only input the engine reads *semantically* rather than
    structurally: sit permission and flag-terminality come off these four
    numbers, so a quartet that is silently truncated, reordered, or coerced
    produces a confident evaluation of a position nobody is in. Every case here
    must fall back to Mode and say so, never guess.
    """

    @staticmethod
    def parse(raw):
        # Imported here rather than at module scope, matching TestSearchTimeout:
        # handler.py reaches for the runpod SDK, which is absent locally.
        import handler as handler_module

        return handler_module._parse_clocks(raw)

    def test_four_deciseconds_pass_through(self):
        assert self.parse([1200, 1180, 1190, 1205]) == [1200, 1180, 1190, 1205]

    def test_a_tuple_is_accepted(self):
        assert self.parse((1, 2, 3, 4)) == [1, 2, 3, 4]

    def test_a_dict_is_read_in_engine_order(self):
        # A-White A-Black B-White B-Black, which is the Clocks option's order
        # and not the order the keys happen to appear in.
        got = self.parse(
            {"bBlack": 4, "aWhite": 1, "bWhite": 3, "aBlack": 2}
        )
        assert got == [1, 2, 3, 4]

    def test_a_dict_missing_a_key_is_refused(self):
        assert self.parse({"aWhite": 1, "aBlack": 2, "bWhite": 3}) is None

    def test_absent_means_no_clock_model(self):
        assert self.parse(None) is None

    def test_a_short_quartet_is_refused_rather_than_padded(self):
        assert self.parse([1, 2, 3]) is None

    def test_a_long_quartet_is_refused_rather_than_truncated(self):
        assert self.parse([1, 2, 3, 4, 5]) is None

    def test_true_is_refused_rather_than_read_as_one_decisecond(self):
        # bool subclasses int, so a caller still sending the old boolean mode
        # would otherwise land on a 1-decisecond clock -- a flagged position.
        assert self.parse([True, 1180, 1190, 1205]) is None

    def test_a_negative_clock_is_refused(self):
        assert self.parse([1200, -1, 1190, 1205]) is None

    def test_a_non_numeric_entry_is_refused(self):
        assert self.parse([1200, "1180", 1190, 1205]) is None

    def test_a_bare_scalar_is_refused(self):
        assert self.parse(1200) is None

    def test_zero_is_a_flagged_clock_not_a_missing_one(self):
        # Zero is legal input: a flagged clock is a real position the engine
        # must be able to evaluate, and is not the same as "no clocks given".
        assert self.parse([0, 1180, 1190, 1205]) == [0, 1180, 1190, 1205]

    def test_floats_are_truncated_to_deciseconds(self):
        # Documents the coercion that does happen: JSON has no int/float
        # distinction, so a float is taken rather than refused, and truncated.
        assert self.parse([119.7, 1180, 1190, 1205]) == [119, 1180, 1190, 1205]


class TestVerifyReady:
    """
    The engine answers `uciok` before it knows whether any GPU engine loaded,
    so a worker with an unusable TensorRT plan completes the UCI handshake,
    exits 0, and fails every search. Only a real search proves readiness.
    """

    def _engine_returning(self, lines):
        import handler as handler_module

        eng = handler_module.HivemindEngine()
        eng._send = lambda cmd: None
        eng.set_position = lambda *a, **k: None
        pending = list(lines)
        eng._read_line = lambda timeout=1: pending.pop(0) if pending else None
        return eng

    def test_passes_when_a_search_returns(self):
        self._engine_returning(["info depth 1", "bestmove (e2e4,pass)"]).verify_ready()

    def test_raises_when_no_gpu_engine_loaded(self):
        eng = self._engine_returning(["Error: No engines have been initialized!"])
        with pytest.raises(RuntimeError, match="no GPU engine loaded"):
            eng.verify_ready()

    def test_raises_when_the_engine_says_nothing(self):
        with pytest.raises(TimeoutError):
            self._engine_returning([]).verify_ready(timeout=0.3)


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
        "print('@@@'+json.dumps(handler.handler_sync({'input': %s})))"
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
