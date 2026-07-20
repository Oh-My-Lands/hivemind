#!/usr/bin/env python3
"""
WebSocket bridge for the Hivemind bughouse engine.

SECONDARY PATH. The primary deployment is deploy/runpod/, which returns one
settled result per request and needs no long-lived server. Prefer
deploy/runpod/dev_server.py for local development, since it exercises the same
handler the deployed endpoint runs.

This exists for the case that one does not cover: watching lines refine live
while a search deepens, and stopping it partway. It needs a persistent process,
so it only makes sense on a pod you are already running.

Protocol, client -> server (JSON text frames):

    {"type": "position", "fen": "<fenA>|<fenB>", "moves": ["1e2e4", ...]}
    {"type": "analyze", "multipv": 5, "analysisBoard": 1,
                        "team": "white", "mode": "go",
                        "movetime": 5000 | "nodes": 100000 | "infinite": true}
    {"type": "stop"}
    {"type": "ping"}

Server -> client:

    {"type": "ready"}                       once the engine answers uciok
    {"type": "info", "searchId": N, ...}    one per parsed info line
    {"type": "infoString", "message": ...}  engine diagnostics, passed through
    {"type": "bestmove", "searchId": N, "moveA": ..., "moveB": ...}
    {"type": "error", "message": ...}
    {"type": "pong"}

One engine process per connection. That is deliberately simple rather than
efficient: the process holds GPU memory and a TensorRT plan, so a busy server
would want a pool. For a single-user analysis tool, isolation is worth more.
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import time
from collections import deque
from typing import Any, Dict, List, Optional

import websockets

# Parsing lives with the serverless handler, which is the primary deployment.
# Shared rather than duplicated so the two cannot disagree about the engine's
# output format.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runpod")
)
from uci_parse import (  # noqa: E402
    parse_bestmove_line,
    parse_info_line,
    parse_pv,
)

LOG = logging.getLogger("ws_bridge")

MAX_MULTIPV = 100
MAX_MOVETIME_MS = 600_000
MAX_NODES = 100_000_000


class OutboundBuffer:
    """
    Coalescing sink for server -> client messages.

    Analysis output is a stream of *replacements*, not events: a newer line for
    PV slot 2 makes the previous one worthless. So messages are keyed, and a
    later message with the same key overwrites an undelivered earlier one. A
    slow client then sees fewer, current frames rather than a growing backlog
    of stale ones.

    Terminal messages (bestmove, errors) are given unique keys so they are
    never dropped.
    """

    def __init__(self) -> None:
        self._pending: Dict[Any, Dict[str, Any]] = {}
        self._event = asyncio.Event()
        self._dropped = 0

    def put(self, key: Any, message: Dict[str, Any]) -> None:
        if key in self._pending:
            self._dropped += 1
        self._pending[key] = message
        self._event.set()

    @property
    def dropped(self) -> int:
        return self._dropped

    async def drain(self) -> List[Dict[str, Any]]:
        """Waits for at least one message, then takes everything buffered."""
        await self._event.wait()
        # dict preserves insertion order, so ordering across distinct keys holds
        batch = list(self._pending.values())
        self._pending.clear()
        self._event.clear()
        return batch


class EngineSession:
    """Owns one engine process and pumps it for one WebSocket connection."""

    def __init__(self, ws, engine_path: str, engine_cwd: str) -> None:
        self.ws = ws
        self.engine_path = engine_path
        self.engine_cwd = engine_cwd
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.out = OutboundBuffer()

        # Searches that have been started but not yet produced a bestmove, in
        # order. The engine emits exactly one bestmove per `go`, so this
        # attributes streamed lines to the right search without guessing:
        # info belongs to the oldest in-flight search, and a bestmove closes it.
        #
        # This matters because `go` implicitly stops any running search, and
        # `ucinewgame` does too -- both make the previous search emit its
        # bestmove at a moment the client did not ask for.
        self._inflight: deque = deque()
        self._next_search_id = 1

        # Options are sticky in UCI, so only changes are worth sending. Mode in
        # particular is hashed into the position key, so changing it invalidates
        # the tree and requires a ucinewgame rather than a continuation.
        self._options: Dict[str, str] = {}
        self._position_command: Optional[str] = None
        self._write_lock = asyncio.Lock()
        self.last_activity = time.monotonic()

    # -------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        LOG.info("starting engine: %s (cwd=%s)", self.engine_path, self.engine_cwd)
        self.proc = await asyncio.create_subprocess_exec(
            self.engine_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=self.engine_cwd,
        )
        await self._send("uci")

    async def close(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.returncode is None:
                await self._send("quit")
                try:
                    await asyncio.wait_for(self.proc.wait(), timeout=5)
                except asyncio.TimeoutError:
                    LOG.warning("engine ignored quit; killing")
                    self.proc.kill()
                    await self.proc.wait()
        except (ProcessLookupError, ConnectionResetError, BrokenPipeError):
            pass
        finally:
            self.proc = None

    async def _send(self, command: str) -> None:
        """Writes one command to the engine's stdin."""
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("engine is not running")
        LOG.debug(">>> %s", command)
        async with self._write_lock:
            self.proc.stdin.write((command + "\n").encode())
            await self.proc.stdin.drain()

    # ----------------------------------------------------------------- pumps

    async def pump_engine(self) -> None:
        """Reads engine stdout forever, parsing lines into outbound messages."""
        assert self.proc is not None and self.proc.stdout is not None
        while True:
            raw = await self.proc.stdout.readline()
            if not raw:
                break  # engine exited
            line = raw.decode(errors="replace").strip()
            if not line:
                continue
            LOG.debug("<<< %s", line)
            self._handle_engine_line(line)

        code = self.proc.returncode if self.proc else None
        self.out.put(("error", time.monotonic()), {
            "type": "error",
            "message": f"engine exited (code {code})",
        })

    def _handle_engine_line(self, line: str) -> None:
        if line.startswith("uciok"):
            self.out.put("ready", {"type": "ready"})
            return

        if line.startswith("bestmove"):
            parsed = parse_bestmove_line(line)
            search_id = self._inflight.popleft() if self._inflight else None
            self.out.put(("bestmove", search_id, time.monotonic()), {
                "type": "bestmove",
                "searchId": search_id,
                **(parsed or {}),
            })
            return

        info = parse_info_line(line)
        if info is None:
            return

        if info["kind"] == "string":
            self.out.put(("infoString", info["message"]), {
                "type": "infoString",
                "message": info["message"],
            })
            return

        search_id = self._inflight[0] if self._inflight else None
        payload = {k: v for k, v in info.items() if k != "kind"}
        # Keyed by PV slot, so a refreshed line for slot 2 replaces the stale one
        # rather than queueing behind it.
        self.out.put(("info", search_id, payload["multipv"]), {
            "type": "info",
            "searchId": search_id,
            **payload,
        })

    async def pump_client(self) -> None:
        """Drains buffered messages to the WebSocket."""
        while True:
            for message in await self.out.drain():
                await self.ws.send(json.dumps(message))

    # -------------------------------------------------------------- commands

    async def _set_option(self, name: str, value: Any) -> None:
        value = str(value)
        if self._options.get(name) == value:
            return
        self._options[name] = value
        await self._send(f"setoption name {name} value {value}")

    async def handle_message(self, message: Dict[str, Any]) -> None:
        self.last_activity = time.monotonic()
        kind = message.get("type")

        if kind == "ping":
            self.out.put(("pong", time.monotonic()), {"type": "pong"})
            return

        if kind == "position":
            await self._handle_position(message)
            return

        if kind == "analyze":
            await self._handle_analyze(message)
            return

        if kind == "stop":
            # Non-blocking in the engine: it signals the search and returns, so
            # the bestmove arrives on the reader shortly after.
            await self._send("stop")
            return

        raise ValueError(f"unknown message type: {kind!r}")

    async def _handle_position(self, message: Dict[str, Any]) -> None:
        fen = message.get("fen")
        if not isinstance(fen, str) or not fen.strip():
            raise ValueError("position requires a 'fen' string")
        if "|" not in fen:
            raise ValueError("fen must contain both boards separated by '|'")

        moves = message.get("moves") or []
        if not isinstance(moves, list) or any(not isinstance(m, str) for m in moves):
            raise ValueError("'moves' must be a list of strings")
        # Board-prefixed UCI, '1' for board A and '2' for board B.
        if any(not m[:1] in ("1", "2") for m in moves):
            raise ValueError("each move must be prefixed with its board, 1 or 2")

        self._position_command = "position fen " + fen.strip()
        if moves:
            self._position_command += " moves " + " ".join(moves)
        await self._send(self._position_command)

    async def _handle_analyze(self, message: Dict[str, Any]) -> None:
        multipv = int(message.get("multipv", 1))
        if not 1 <= multipv <= MAX_MULTIPV:
            raise ValueError(f"multipv must be in 1..{MAX_MULTIPV}")

        board = int(message.get("analysisBoard", 1))
        if board not in (1, 2):
            raise ValueError("analysisBoard must be 1 or 2")

        team = message.get("team", "white")
        if team not in ("white", "black"):
            raise ValueError("team must be 'white' or 'black'")

        mode = message.get("mode", "go")
        if mode not in ("go", "sit"):
            raise ValueError("mode must be 'go' or 'sit'")

        # Mode feeds an NN input plane and is hashed into the position key, so a
        # change invalidates any retained tree. Continuing a search across it
        # would mix evaluations from two different rule sets.
        mode_changed = self._options.get("Mode") not in (None, mode)

        await self._set_option("MultiPV", multipv)
        await self._set_option("AnalysisBoard", board)
        await self._set_option("Team", team)
        await self._set_option("Mode", mode)

        if mode_changed:
            await self._send("ucinewgame")

        # Re-send the position: ucinewgame clears tree reuse, and the client may
        # be analysing a different node than the last `position` set.
        position = message.get("fen")
        if position:
            await self._handle_position(message)
        elif self._position_command:
            await self._send(self._position_command)
        else:
            raise ValueError("no position set; send a 'position' message first")

        search_id = self._next_search_id
        self._next_search_id += 1
        # Registered before `go` is written, so a bestmove cannot arrive and be
        # attributed to the wrong search.
        self._inflight.append(search_id)

        if message.get("infinite"):
            await self._send("go infinite")
        elif message.get("nodes"):
            nodes = min(int(message["nodes"]), MAX_NODES)
            await self._send(f"go nodes {nodes}")
        else:
            movetime = min(int(message.get("movetime", 3000)), MAX_MOVETIME_MS)
            await self._send(f"go movetime {movetime}")

        self.out.put(("searchStarted", search_id), {
            "type": "searchStarted",
            "searchId": search_id,
        })


async def _watchdog(session: EngineSession, idle_timeout: float) -> None:
    """Closes a session that has gone quiet, so an idle GPU is not held."""
    while True:
        await asyncio.sleep(min(idle_timeout, 30))
        idle = time.monotonic() - session.last_activity
        if idle >= idle_timeout:
            LOG.info("closing session after %.0fs idle", idle)
            await session.ws.close(code=1000, reason="idle timeout")
            return


async def serve_connection(ws, engine_path: str, engine_cwd: str,
                           idle_timeout: float) -> None:
    session = EngineSession(ws, engine_path, engine_cwd)
    LOG.info("client connected: %s", getattr(ws, "remote_address", "?"))

    try:
        await session.start()
    except Exception as exc:
        LOG.exception("failed to start engine")
        await ws.send(json.dumps({"type": "error", "message": str(exc)}))
        await ws.close()
        return

    tasks = [
        asyncio.create_task(session.pump_engine(), name="pump_engine"),
        asyncio.create_task(session.pump_client(), name="pump_client"),
        asyncio.create_task(_watchdog(session, idle_timeout), name="watchdog"),
    ]

    try:
        async for raw in ws:
            try:
                message = json.loads(raw)
                if not isinstance(message, dict):
                    raise ValueError("expected a JSON object")
                await session.handle_message(message)
            except (ValueError, json.JSONDecodeError) as exc:
                # Client errors are reported, not fatal -- a bad frame should
                # not tear down a session mid-analysis.
                await ws.send(json.dumps({"type": "error", "message": str(exc)}))
    except websockets.ConnectionClosed:
        pass
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await session.close()
        if session.out.dropped:
            LOG.info("session dropped %d superseded frames", session.out.dropped)
        LOG.info("client disconnected")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("BRIDGE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("BRIDGE_PORT", "8765")))
    parser.add_argument("--engine",
                        default=os.environ.get("ENGINE_PATH", "./build/hivemind"))
    parser.add_argument("--engine-cwd",
                        default=os.environ.get("ENGINE_CWD", "."),
                        help="engine working directory; it looks for ./networks here")
    parser.add_argument("--idle-timeout", type=float,
                        default=float(os.environ.get("IDLE_TIMEOUT", "900")),
                        help="seconds of client silence before the session closes")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    engine_path = args.engine
    if not os.path.isfile(engine_path):
        resolved = shutil.which(engine_path)
        if resolved is None:
            raise SystemExit(
                f"engine not found at {engine_path!r}; pass --engine or set ENGINE_PATH"
            )
        engine_path = resolved
    engine_path = os.path.abspath(engine_path)

    networks = os.path.join(args.engine_cwd, "networks")
    if not os.path.isdir(networks):
        # The engine looks for ./networks relative to its cwd and fails opaquely
        # without it, so this is worth catching up front.
        LOG.warning("no networks/ directory under %s -- the engine will likely fail",
                    os.path.abspath(args.engine_cwd))

    async def run() -> None:
        async with websockets.serve(
            lambda ws: serve_connection(ws, engine_path, args.engine_cwd,
                                        args.idle_timeout),
            args.host,
            args.port,
            ping_interval=20,
            ping_timeout=20,
            max_queue=32,
        ):
            LOG.info("listening on ws://%s:%d (engine=%s)",
                     args.host, args.port, engine_path)
            await asyncio.Future()  # run forever

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        LOG.info("shutting down")


if __name__ == "__main__":
    main()
