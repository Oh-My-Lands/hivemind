"""
Tests for the WebSocket bridge.

The parser tests use lines captured verbatim from the built engine, not
hand-written approximations -- the point is to track what the engine actually
emits.

The end-to-end test needs a GPU and the built engine, so it is skipped unless
HIVEMIND_ENGINE points at one.

    pytest test_ws_bridge.py                                  # parser only
    HIVEMIND_ENGINE=../../build/hivemind \
    HIVEMIND_ENGINE_CWD=../..  pytest test_ws_bridge.py       # everything
"""

import asyncio
import json
import os

import pytest
import websockets

from ws_bridge import (
    EngineSession,
    OutboundBuffer,
    parse_bestmove_line,
    parse_info_line,
    parse_pv,
    serve_connection,
)

# Captured from `./build/hivemind` with MultiPV 3 on the start position.
REAL_INFO = (
    "info depth 12 multipv 2 score cp -6 q -0.0219 prior 0.4569 visits 21179 "
    "nodes 47264 nps 13864 hashfull 10 tbhits 10512 time 3409 "
    "pv (e2e4,pass) (e7e5,e2e4) (b1c3,e7e5) (b8c6,pass)"
)
REAL_INFO_WITH_DROP = (
    "info depth 12 multipv 1 score cp -4 q -0.0169 prior 0.3265 visits 21238 "
    "nodes 47264 nps 13864 hashfull 10 tbhits 10512 time 3409 "
    "pv (d2d4,pass) (g5e7,P@e6)"
)


class TestParseInfoLine:
    def test_extracts_every_field(self):
        got = parse_info_line(REAL_INFO)
        assert got["kind"] == "pv"
        assert got["depth"] == 12
        assert got["multipv"] == 2
        assert got["score"] == {"kind": "cp", "value": -6}
        assert got["q"] == pytest.approx(-0.0219)
        assert got["prior"] == pytest.approx(0.4569)
        assert got["visits"] == 21179
        assert got["nodes"] == 47264
        assert got["nps"] == 13864
        assert got["hashfull"] == 10
        assert got["tbhits"] == 10512
        assert got["time"] == 3409

    def test_parses_joint_actions_and_pass(self):
        got = parse_info_line(REAL_INFO)
        assert got["pv"][0] == {"a": "e2e4", "b": None}
        assert got["pv"][1] == {"a": "e7e5", "b": "e2e4"}
        assert got["pv"][3] == {"a": "b8c6", "b": None}

    def test_parses_drop_moves(self):
        got = parse_info_line(REAL_INFO_WITH_DROP)
        assert got["pv"][1] == {"a": "g5e7", "b": "P@e6"}

    def test_recognises_info_string_separately(self):
        # These are interleaved with PV lines; a parser assuming PV structure
        # would break on them.
        got = parse_info_line("info string Early stopping: saved 593ms")
        assert got == {"kind": "string", "message": "Early stopping: saved 593ms"}

    def test_defaults_multipv_when_absent(self):
        line = "info depth 3 score cp 12 q 0.05 prior 0.1 visits 10 time 5 pv (e2e4,pass)"
        assert parse_info_line(line)["multipv"] == 1

    def test_parses_mate_scores(self):
        line = "info depth 5 multipv 1 score mate 3 q 0.99 prior 0.5 visits 90 time 7 pv (h5f7,pass)"
        assert parse_info_line(line)["score"] == {"kind": "mate", "value": 3}

    def test_ignores_unknown_fields(self):
        # Must survive the engine growing a field the bridge does not know.
        line = REAL_INFO.replace("time 3409", "wibble 7 time 3409")
        assert parse_info_line(line)["time"] == 3409

    def test_returns_none_for_non_info(self):
        assert parse_info_line("bestmove (d2d4,pass)") is None
        assert parse_info_line("readyok") is None

    def test_returns_none_without_depth(self):
        assert parse_info_line("info nodes 5") is None

    def test_survives_truncated_line(self):
        # A partially flushed line must not raise.
        assert parse_info_line("info depth 12 multipv 2 score cp") is not None
        assert parse_info_line("info depth") is None


class TestParsePv:
    def test_empty_pv(self):
        assert parse_pv("") == []

    def test_both_sides_passing(self):
        assert parse_pv("(pass,pass)") == [{"a": None, "b": None}]


class TestParseBestmove:
    def test_parses_joint_bestmove(self):
        assert parse_bestmove_line("bestmove (d2d4,pass)") == {
            "moveA": "d2d4",
            "moveB": None,
        }

    def test_handles_none(self):
        assert parse_bestmove_line("bestmove (none)") == {
            "moveA": None,
            "moveB": None,
        }

    def test_returns_none_for_other_lines(self):
        assert parse_bestmove_line("info depth 1") is None


class TestOutboundBuffer:
    @pytest.mark.asyncio
    async def test_supersedes_same_key(self):
        buf = OutboundBuffer()
        buf.put(("info", 1, 1), {"depth": 1})
        buf.put(("info", 1, 1), {"depth": 2})
        batch = await buf.drain()
        assert batch == [{"depth": 2}], "stale frame should have been replaced"
        assert buf.dropped == 1

    @pytest.mark.asyncio
    async def test_keeps_distinct_keys_in_order(self):
        buf = OutboundBuffer()
        buf.put(("info", 1, 1), {"multipv": 1})
        buf.put(("info", 1, 2), {"multipv": 2})
        assert await buf.drain() == [{"multipv": 1}, {"multipv": 2}]

    @pytest.mark.asyncio
    async def test_never_drops_terminal_messages(self):
        buf = OutboundBuffer()
        buf.put(("bestmove", 1, 0.1), {"type": "bestmove"})
        buf.put(("bestmove", 2, 0.2), {"type": "bestmove"})
        assert len(await buf.drain()) == 2

    @pytest.mark.asyncio
    async def test_drain_blocks_until_a_message_arrives(self):
        buf = OutboundBuffer()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(buf.drain(), timeout=0.05)


class TestSearchAttribution:
    """
    `go` and `ucinewgame` both implicitly stop a running search, so a bestmove
    can arrive when the client did not ask for one. Lines must still be
    attributed to the search they belong to.
    """

    def _session(self):
        return EngineSession(ws=None, engine_path="/nonexistent", engine_cwd=".")

    def test_info_belongs_to_oldest_inflight_search(self):
        s = self._session()
        s._inflight.extend([7, 8])
        s._handle_engine_line(REAL_INFO)
        sent = list(s.out._pending.values())[0]
        assert sent["searchId"] == 7

    def test_bestmove_closes_the_oldest_search(self):
        s = self._session()
        s._inflight.extend([7, 8])
        s._handle_engine_line("bestmove (d2d4,pass)")
        sent = list(s.out._pending.values())[0]
        assert sent["searchId"] == 7
        assert list(s._inflight) == [8]

        # Now the next search owns incoming info.
        s._handle_engine_line(REAL_INFO)
        assert [m for m in s.out._pending.values() if m["type"] == "info"][0][
            "searchId"
        ] == 8

    def test_unsolicited_bestmove_does_not_crash(self):
        s = self._session()
        s._handle_engine_line("bestmove (d2d4,pass)")
        assert list(s.out._pending.values())[0]["searchId"] is None


# --------------------------------------------------------------------- e2e

ENGINE = os.environ.get("HIVEMIND_ENGINE")
ENGINE_CWD = os.environ.get("HIVEMIND_ENGINE_CWD", ".")

requires_engine = pytest.mark.skipif(
    not ENGINE or not os.path.isfile(ENGINE),
    reason="set HIVEMIND_ENGINE to the built binary to run end-to-end tests",
)


@pytest.fixture
async def bridge():
    """Runs the bridge on an ephemeral port against the real engine."""
    async with websockets.serve(
        lambda ws: serve_connection(ws, ENGINE, ENGINE_CWD, idle_timeout=600),
        "127.0.0.1",
        0,
    ) as server:
        port = server.sockets[0].getsockname()[1]
        yield f"ws://127.0.0.1:{port}"


async def _recv_until(ws, predicate, timeout=180):
    """Collects messages until one satisfies predicate; returns all of them."""
    seen = []
    async def loop():
        while True:
            message = json.loads(await ws.recv())
            seen.append(message)
            if predicate(message):
                return seen
    return await asyncio.wait_for(loop(), timeout=timeout)


START = ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1|"
         "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1")


@requires_engine
@pytest.mark.asyncio
async def test_streams_info_then_bestmove(bridge):
    async with websockets.connect(bridge) as ws:
        await _recv_until(ws, lambda m: m["type"] == "ready")

        await ws.send(json.dumps({
            "type": "analyze", "fen": START,
            "multipv": 3, "analysisBoard": 1, "movetime": 4000,
        }))
        seen = await _recv_until(ws, lambda m: m["type"] == "bestmove")

        infos = [m for m in seen if m["type"] == "info"]
        assert infos, "no info lines streamed"
        # Streaming, not batched at the end: lines must predate the bestmove.
        assert seen.index(infos[0]) < seen.index(seen[-1])
        assert {m["multipv"] for m in infos} <= {1, 2, 3}
        assert all("q" in m and "visits" in m for m in infos)
        assert seen[-1]["moveA"] is not None


@requires_engine
@pytest.mark.asyncio
async def test_infinite_search_stops_on_request(bridge):
    async with websockets.connect(bridge) as ws:
        await _recv_until(ws, lambda m: m["type"] == "ready")

        await ws.send(json.dumps({
            "type": "analyze", "fen": START, "multipv": 2, "infinite": True,
        }))
        # Let it run, then stop it. An infinite search must not end on its own.
        await asyncio.sleep(5)
        await ws.send(json.dumps({"type": "stop"}))

        seen = await _recv_until(ws, lambda m: m["type"] == "bestmove", timeout=30)
        infos = [m for m in seen if m["type"] == "info"]
        assert len(infos) >= 2
        assert max(m["time"] for m in infos) >= 3000, "search ended early"


@requires_engine
@pytest.mark.asyncio
async def test_search_ids_distinguish_consecutive_searches(bridge):
    async with websockets.connect(bridge) as ws:
        await _recv_until(ws, lambda m: m["type"] == "ready")

        bestmoves = []
        for _ in range(2):
            await ws.send(json.dumps({
                "type": "analyze", "fen": START, "movetime": 2000,
            }))
            seen = await _recv_until(ws, lambda m: m["type"] == "bestmove")
            bestmoves.append(seen[-1])

        ids = [m["searchId"] for m in bestmoves]
        assert ids[0] is not None and ids[1] is not None
        assert ids[0] != ids[1], "consecutive searches must be distinguishable"

        # Every info line must belong to a search that was actually started.
        assert all(m["searchId"] in ids for m in seen if m["type"] == "info")


@requires_engine
@pytest.mark.asyncio
async def test_interrupting_a_search_attributes_both_bestmoves(bridge):
    """
    The client starts a second search without waiting for the first to finish.
    `go` implicitly stops the running search, so two bestmoves come back -- and
    each must be attributed to the search it actually belongs to, rather than
    both landing on whichever search happened to be current when they arrived.
    """
    async with websockets.connect(bridge) as ws:
        await _recv_until(ws, lambda m: m["type"] == "ready")

        await ws.send(json.dumps({
            "type": "analyze", "fen": START, "infinite": True,
        }))
        first = await _recv_until(ws, lambda m: m["type"] == "searchStarted")
        first_id = first[-1]["searchId"]

        # Let it get going, then pre-empt it.
        await asyncio.sleep(3)
        await ws.send(json.dumps({
            "type": "analyze", "fen": START, "movetime": 2000,
        }))

        seen = await _recv_until(
            ws,
            lambda m: m["type"] == "bestmove" and m["searchId"] != first_id,
            timeout=60,
        )

        bestmoves = [m for m in seen if m["type"] == "bestmove"]
        ids = [m["searchId"] for m in bestmoves]
        assert first_id in ids, "the pre-empted search never reported a bestmove"
        assert len(set(ids)) == len(ids), f"bestmoves misattributed: {ids}"


@requires_engine
@pytest.mark.asyncio
async def test_bad_frame_reports_error_without_closing(bridge):
    async with websockets.connect(bridge) as ws:
        await _recv_until(ws, lambda m: m["type"] == "ready")

        await ws.send("not json")
        seen = await _recv_until(ws, lambda m: m["type"] == "error", timeout=10)
        assert seen[-1]["type"] == "error"

        # The session must survive a bad frame.
        await ws.send(json.dumps({"type": "ping"}))
        await _recv_until(ws, lambda m: m["type"] == "pong", timeout=10)


@requires_engine
@pytest.mark.asyncio
async def test_rejects_single_board_fen(bridge):
    async with websockets.connect(bridge) as ws:
        await _recv_until(ws, lambda m: m["type"] == "ready")
        await ws.send(json.dumps({
            "type": "position",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1",
        }))
        seen = await _recv_until(ws, lambda m: m["type"] == "error", timeout=10)
        assert "both boards" in seen[-1]["message"]
