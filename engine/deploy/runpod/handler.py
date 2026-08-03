#!/usr/bin/env python3
"""
Hivemind Bughouse Engine - RunPod Serverless Handler

Handles inference requests for the bughouse chess engine.
Receives board positions, runs MCTS search, returns best moves.

UCI Protocol Summary for Hivemind:
- FEN format: "fenA|fenB" (two boards separated by |)
- Move format: "1e2e4" (board number prefix: 1=A, 2=B)
- Bestmove format: "(moveA,moveB)" or "(e2e4,pass)"
- Options: Hash, MultiPV, Team (white/black), Mode (sit/go)
"""

import os
import asyncio
import subprocess
import json
import re
import time
import threading
import queue
import signal
from collections import deque
from typing import Dict, Any, Optional, List

from uci_parse import (
    collect_final_lines,
    parse_bestmove_line,
    parse_info_line,
)

try:
    import runpod
except ImportError:  # the dev server and tests do not need the SDK
    runpod = None

# Configuration
ENGINE_PATH = os.environ.get('ENGINE_PATH', '/app/build/hivemind')
NETWORKS_DIR = os.environ.get('NETWORKS_DIR', '/app/networks')
ENGINE_CWD = os.environ.get('ENGINE_CWD', '/app')
DEFAULT_NODES = 800
DEFAULT_MOVE_TIME_MS = 1000
MAX_MOVE_TIME_MS = 30000  # 30 second max
MAX_NODES = 50_000_000
MAX_MULTIPV = 100

# Throughput floor for turning a node budget into a timeout. See `go` for why
# this is a floor rather than an estimate, and why it is under the 7,221 nps
# worst case measured on the serving GPU.
#
# Note what this does NOT bound: RunPod kills the job at the endpoint's own
# `executionTimeoutMs` (600s), so a budget over ~4.3M nodes cannot finish
# whatever this says. The caller's proxy caps at 4M for that reason.
MIN_NODES_PER_SECOND = 6000

# Fixed overhead allowed on top of the search itself: the `go` reaching the
# engine, the first visit on a cold cache, and bestmove being read back out.
SEARCH_TIMEOUT_BUFFER_S = 30


def search_timeout_seconds(movetime: int = None, nodes: int = None) -> float:
    """
    How long to wait for bestmove.

    A backstop against a wedged engine, not an expected search length -- but it
    has to scale with what was actually asked for, which is the bug this
    function exists to have fixed.

    A node search used to take its budget from MAX_MOVE_TIME_MS, so every node
    search -- 20k or 4M -- waited exactly 60s and then raised. The old comment
    was right that this is a backstop rather than a prediction, and then took
    the wrong number for it: the movetime cap says what the *other* budget kind
    may ask for and nothing whatever about how long a node search should run.
    The effect was an invisible ceiling at ~430-490k nodes on the serving GPU.
    The client offered a 500k option for eight days and it never once returned;
    every attempt surfaced as an HTTP 502 carrying "Timeout waiting for
    bestmove", which reads as a broken engine rather than a budget that was
    never reachable in the first place.

    MIN_NODES_PER_SECOND is a floor, not an estimate. The measured worst case on
    the serving RTX 4000 Ada is 7,221 nps, on a middlegame holding 18 pieces in
    hand -- pocket size, not tactics, is what makes a bughouse position slow.
    This sits under that so a position slower than any yet measured still
    completes, because the two errors do not cost the same: too generous and a
    genuinely wedged engine is caught late, having burned GPU seconds nobody is
    waiting for; too tight and a healthy search is killed *after doing all of
    the work*, and billed in full for no answer.
    """
    if movetime:
        search_time_ms = movetime
    elif nodes:
        search_time_ms = (nodes / MIN_NODES_PER_SECOND) * 1000
    else:
        search_time_ms = DEFAULT_MOVE_TIME_MS

    return (search_time_ms / 1000) + SEARCH_TIMEOUT_BUFFER_S


class HivemindEngine:
    """Manages a Hivemind UCI engine session for bughouse."""
    
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.initialized = False
        self.output_queue: queue.Queue = queue.Queue()
        # Everything the engine said, kept alongside the queue because the queue
        # is drained by whoever is waiting. On a startup failure the queue is
        # usually empty by the time anyone asks what went wrong, and the one
        # line that explains it -- ld.so's "error while loading shared
        # libraries", or the engine's own fatal -- is already gone.
        self.recent_output: deque = deque(maxlen=40)
        self.reader_thread: Optional[threading.Thread] = None
        self.running = False
        # Tracked so a Mode change can force a fresh tree; see handler().
        self.current_mode: Optional[str] = None
        
    def _reader_worker(self):
        """Background thread to read engine output."""
        while self.running and self.process and self.process.stdout:
            try:
                line = self.process.stdout.readline()
                if line:
                    line = line.strip()
                    print(f"<<< {line}", flush=True)
                    self.recent_output.append(line)
                    self.output_queue.put(line)
                elif self.process.poll() is not None:
                    # Process exited
                    break
            except Exception as e:
                print(f"Reader error: {e}", flush=True)
                break
        
    def start(self):
        """Start the engine process."""
        if self.process is not None and self.process.poll() is None:
            return  # Already running
            
        print(f"Starting engine: {ENGINE_PATH}")
        self.process = subprocess.Popen(
            [ENGINE_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,
            # The engine resolves ./networks against its working directory.
            # /app is where the image puts it; override for local runs.
            cwd=ENGINE_CWD,
        )
        
        # Start reader thread
        self.running = True
        self.reader_thread = threading.Thread(target=self._reader_worker, daemon=True)
        self.reader_thread.start()
        
        # Wait a moment for process to start
        time.sleep(0.1)
        
        # Initialize UCI
        # 60s was too short. The engine answers uci only after the network is
        # up, and while a cached plan loads in 0.6s, a cache miss rebuilds from
        # ONNX -- measured at 236s. At 60s that rebuild could not even finish
        # before the worker was killed and restarted, so the failure looked like
        # a silent hang rather than a slow build. Allow it to complete and say
        # so; a plan that has to be rebuilt on every cold start is a problem to
        # fix, not to hide behind a timeout.
        self._send("uci")
        response = self._wait_for("uciok", timeout=300)
        print(f"UCI response: {response[:3]}...")

        self._send("isready")
        self._wait_for("readyok", timeout=300)
        
        self.initialized = True
        print("Engine initialized successfully")
        
    def stop(self):
        """Stop the engine process."""
        if self.process:
            try:
                self.running = False
                self._send("quit")
                self.process.wait(timeout=5)
            except:
                self.process.kill()
            self.process = None
            self.initialized = False
            
    def _send(self, cmd: str):
        """Send a command to the engine."""
        if self.process and self.process.stdin:
            print(f">>> {cmd}", flush=True)
            self.process.stdin.write(cmd + "\n")
            self.process.stdin.flush()
            
    def _read_line(self, timeout: float = 10) -> Optional[str]:
        """Read a line from engine output with timeout."""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        
    def _wait_for(self, expected: str, timeout: float = 60) -> List[str]:
        """Wait for a specific response, collecting all lines."""
        lines = []
        start = time.time()
        while time.time() - start < timeout:
            line = self._read_line(timeout=1)
            if line:
                lines.append(line)
                if expected in line:
                    return lines
                continue

            # No line this second. If the process is gone it is never going to
            # produce one, so stop waiting -- a dead engine used to burn the
            # whole timeout before reporting, which turned an instant crash into
            # a five-minute "hang" and made it look like a slow startup.
            #
            # Only checked when the queue is empty: the reader thread can still
            # have buffered output from a process that has already exited, and
            # that output is the diagnostic we care about most.
            if self.process and self.process.poll() is not None:
                break

        # Report what the engine actually said, and whether it is still alive.
        #
        # This used to raise a bare "Timeout waiting for 'uciok'", discarding
        # every line collected above. On a worker that is the only diagnostic
        # available, and without it a failed start is indistinguishable between
        # a crash, a missing library, and a slow TensorRT plan build -- all of
        # which look like sixty seconds of silence followed by exit 1.
        rc = self.process.poll() if self.process else None
        waited = time.time() - start
        tail = lines[-15:] if lines else ["<no output at all>"]

        if rc is None:
            what = f"Timeout waiting for '{expected}' after {waited:.0f}s (engine still running)"
        else:
            # A negative rc is -N for termination by signal N. Naming the signal
            # matters: SIGILL in particular means the binary used an instruction
            # this CPU lacks, which is a build-target bug (-march too new for the
            # host), not anything to do with the engine's own logic.
            if rc < 0:
                try:
                    name = signal.Signals(-rc).name
                except ValueError:
                    name = f"signal {-rc}"
                cause = f"killed by {name} (rc={rc})"
            else:
                cause = f"exited rc={rc}"
            what = f"Engine {cause} after {waited:.0f}s without printing '{expected}'"

        raise TimeoutError(what + ". Last engine output:\n  " + "\n  ".join(tail))
        
    def verify_ready(self, timeout: float = 1800):
        """
        Confirms the engine can actually search, and raises if it cannot.

        A UCI handshake is not evidence of anything: the engine answers `uciok`
        before it knows whether any GPU engine loaded, so a worker whose
        TensorRT plan failed to deserialize completes the handshake, reports no
        error, and then fails every single search. Only a real search proves
        the network is loaded.

        The default timeout is generous because a missing or unusable plan
        triggers a rebuild from ONNX, which takes minutes.
        """
        self.set_position("startpos")
        self._send("go nodes 1")

        start = time.time()
        while time.time() - start < timeout:
            line = self._read_line(timeout=1)
            if line is None:
                continue
            if line.startswith("bestmove"):
                print(f"Engine verified in {time.time() - start:.1f}s", flush=True)
                return
            if "No engines have been initialized" in line:
                raise RuntimeError(
                    "engine started but no GPU engine loaded -- the TensorRT "
                    "plan is missing or unusable and could not be rebuilt"
                )
        raise TimeoutError("engine did not answer a 1-node search")

    def set_option(self, name: str, value: str):
        """Set a UCI option."""
        self._send(f"setoption name {name} value {value}")
        
    def new_game(self):
        """Reset for a new game."""
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok", timeout=30)
        
    def set_position(self, fen: str, moves: List[str] = None):
        """
        Set the board position.
        
        Args:
            fen: Bughouse FEN in format "fenA|fenB"
            moves: Optional list of moves in format "1e2e4" (board prefix + move)
        """
        if fen == "startpos":
            cmd = "position startpos"
        else:
            cmd = f"position fen {fen}"
            
        if moves:
            cmd += " moves " + " ".join(moves)
            
        self._send(cmd)
        
    def _poll_line(self) -> Optional[str]:
        """Take a line if one is waiting. Never blocks, so callers can await."""
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None

    def _drain(self) -> int:
        """Discard anything left in the queue. Returns how many lines went."""
        n = 0
        while self._poll_line() is not None:
            n += 1
        return n

    def cancel_search(self, timeout: float = 10.0) -> bool:
        """Stop the running search and consume the bestmove it emits.

        Consuming it is the part that matters. `stop` does not abort the search
        so much as cut it short: the engine still prints a bestmove, and if
        nobody reads it, it stays in the queue. The next search would then read
        that line before its own output and return the previous position's move
        -- a wrong answer to a different question, which is a worse failure than
        the wasted GPU time this whole path exists to avoid.

        Returns False when no bestmove arrived within the timeout, meaning the
        engine's state is unknown; go() drains before searching so the next
        caller is protected either way.
        """
        self._send("stop")
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                line = self.output_queue.get(timeout=0.5)
            except queue.Empty:
                if self.process and self.process.poll() is not None:
                    return False  # engine died; nothing is coming
                continue
            if line.startswith("bestmove"):
                return True
        print("cancel_search: no bestmove after stop; engine state unknown",
              flush=True)
        return False

    async def go(self, movetime: int = None, nodes: int = None) -> Dict[str, Any]:
        """
        Run search and return results.

        Async so that a cancelled job can actually interrupt it. The RunPod SDK
        stops a job by cancelling its asyncio task (rp_scale.stop_job), which
        can only take effect at an await -- a synchronous handler blocks the
        event loop outright, so the stop signal is not even read until the
        search has finished and the GPU time is already spent.

        Hivemind uses "go movetime <ms>" format.

        Returns:
            Dict with:
            - bestmove: "(moveA,moveB)" format
            - info: List of info strings
            - eval: Centipawn evaluation (if available)
            - nodes: Nodes searched
            - depth: Search depth
            - time_ms: Time taken
        """
        # Anything still queued belongs to a previous search -- normally nothing,
        # but a cancelled search whose bestmove never arrived can leave a line
        # behind, and reading it here would answer this position with that one.
        stale = self._drain()
        if stale:
            print(f"go: discarded {stale} stale line(s) before searching",
                  flush=True)

        if movetime:
            self._send(f"go movetime {movetime}")
        elif nodes:
            # `go nodes` is honoured directly now. This used to convert nodes
            # into an estimated movetime, which was never more than a guess --
            # throughput varies with position and GPU.
            self._send(f"go nodes {nodes}")
        else:
            self._send(f"go movetime {DEFAULT_MOVE_TIME_MS}")

        timeout = search_timeout_seconds(movetime=movetime, nodes=nodes)
        
        # Collect output until bestmove
        info_lines = []
        result = {
            "bestmove": None,
            "info": [],
            "eval": None,
            "nodes": None,
            "depth": None,
            "time_ms": None
        }

        start = time.time()
        try:
            while time.time() - start < timeout:
                line = self._poll_line()
                if not line:
                    # The only await in this loop, so it is the point at which a
                    # cancellation can be delivered. Short enough that a stopped
                    # job gives the GPU back promptly; long enough not to spin.
                    await asyncio.sleep(0.02)
                    continue

                if line.startswith("info"):
                    info_lines.append(line)
                    result["info"].append(line)

                    # Parse info line
                    self._parse_info(line, result)

                elif line.startswith("bestmove"):
                    # Parse bestmove: "bestmove (e2e4,d2d4)"
                    match = re.search(r'bestmove\s+(\([^)]+\)|\S+)', line)
                    if match:
                        result["bestmove"] = match.group(1)

                    result["raw_info"] = info_lines
                    result["time_ms"] = int((time.time() - start) * 1000)
                    return result
        except asyncio.CancelledError:
            # The job was cancelled -- the client disconnected, or it timed out.
            # Stop searching before propagating: the worker is billed for as
            # long as the engine keeps working, and nobody is waiting for it.
            elapsed = time.time() - start
            stopped = self.cancel_search()
            print(f"search cancelled after {elapsed:.1f}s "
                  f"(clean stop: {stopped})", flush=True)
            raise

        raise TimeoutError("Timeout waiting for bestmove")
        
    def _parse_info(self, line: str, result: Dict):
        """Parse UCI info line and update result dict."""
        parts = line.split()
        
        for i, part in enumerate(parts):
            if part == "depth" and i + 1 < len(parts):
                try:
                    result["depth"] = int(parts[i + 1])
                except ValueError:
                    pass
                    
            elif part == "nodes" and i + 1 < len(parts):
                try:
                    result["nodes"] = int(parts[i + 1])
                except ValueError:
                    pass
                    
            elif part == "cp" and i + 1 < len(parts):
                try:
                    result["eval"] = int(parts[i + 1]) / 100.0
                except ValueError:
                    pass
                    
            elif part == "mate" and i + 1 < len(parts):
                try:
                    mate_in = int(parts[i + 1])
                    # Convert mate to large eval
                    result["eval"] = 100.0 if mate_in > 0 else -100.0
                    result["mate"] = mate_in
                except ValueError:
                    pass


# Global engine instance (reused across warm requests)
engine = HivemindEngine()

# Engine startup runs in the background so the worker can register with RunPod
# immediately. _engine_ready fires on success; _engine_error holds the reason on
# failure. Exactly one of the two is set before the event is signalled.
_engine_ready = threading.Event()
_engine_error: Optional[str] = None


def _startup_diagnostic() -> str:
    """Why the engine died, for callers who cannot read the worker log.

    A bare "BrokenPipeError" says only that the process was gone by the time we
    wrote to it, which is true of every startup failure and distinguishes none
    of them. The exit status and the engine's last words do distinguish them:
    127 with "error while loading shared libraries" is a missing .so in the
    image, whereas a TensorRT plan that fails to deserialise says so and takes
    minutes rather than milliseconds.
    """
    parts: List[str] = []

    proc = engine.process
    if proc is None:
        parts.append("engine process was never spawned")
    else:
        # The reader thread may still be draining the pipe of a process that
        # exited microseconds ago; without this the fatal line is often missed.
        if engine.reader_thread is not None:
            engine.reader_thread.join(timeout=2.0)
        code = proc.poll()
        if code is None:
            parts.append("engine process still running")
        elif code < 0:
            parts.append(f"engine killed by signal {-code}")
        else:
            parts.append(f"engine exited {code}")

    if engine.recent_output:
        parts.append("last output: " + " | ".join(engine.recent_output))
    else:
        parts.append("engine produced no output before exiting")

    return " (" + "; ".join(parts) + ")"


def _init_engine_background():
    """Load and verify the engine, then release anything waiting on it."""
    global _engine_error
    try:
        engine.start()
        engine.verify_ready()
        print("Engine initialized successfully; accepting work", flush=True)
    except Exception as exc:
        _engine_error = f"{type(exc).__name__}: {exc}{_startup_diagnostic()}"
        print(f"FATAL: engine failed to become ready: {_engine_error}", flush=True)
    finally:
        # Signalled on both paths: waiters must wake up to see the error, not
        # block until their own timeout on a failure that already happened.
        _engine_ready.set()


def _await_engine(timeout: float = 600) -> Optional[str]:
    """Block until the engine is usable. Returns an error string, or None."""
    if not _engine_ready.wait(timeout=timeout):
        return (f"engine still initializing after {timeout:.0f}s; a TensorRT "
                "plan rebuild can take ~4 minutes on a cold worker")
    return _engine_error


async def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler.

    Async on purpose. The SDK calls handler(job) directly (rp_job.py:257) and
    only awaits the result if it is awaitable, so a synchronous handler holds
    the event loop for the whole search and the worker cannot even notice that
    the job was cancelled until it is over. Being a coroutine is what makes a
    cancelled search stop early instead of running to completion and billing for
    work nobody is waiting for.

    Input format:
    {
        "input": {
            "fen": "rnbqkbnr/...|rnbqkbnr/...",  # Bughouse FEN (boardA|boardB)
            "movetime": 2500,  # Search time in milliseconds
            "moves": "1e2e4 1e7e5",  # Optional: space-separated moves with board prefix
            "mode": "go",  # "go" (normal) or "sit" (time advantage)
            "team": "white",  # Which team we're playing as
            "multipv": false  # Whether to return multiple PVs
        }
    }
    
    Output format:
    {
        "bestmove": "(e2e4,d2d4)",  # Joint action
        "moveA": "e2e4",  # Board A move
        "moveB": "d2d4",  # Board B move  
        "eval": 0.15,  # Evaluation in pawns
        "depth": 12,
        "nodes": 5000,
        "time_ms": 1000,
        "info": [...]  # Raw UCI info strings
    }
    """
    global engine
    
    try:
        job_input = job.get("input", {})
        action = job_input.get("action", "move")

        # Wait for background startup rather than starting the engine here. The
        # old `if not engine.initialized: engine.start()` raced: two concurrent
        # jobs on one worker could both see False and each spawn a process.
        # Waited for in a thread so the event loop stays free: the SDK's
        # stop-signal poller runs on it, and blocking here would make a job
        # uncancellable for as long as a cold engine takes to load.
        init_error = await asyncio.get_running_loop().run_in_executor(
            None, _await_engine)
        if init_error is not None:
            return {"error": "engine unavailable: " + init_error}


        # Handle different actions
        if action == "newgame":
            engine.new_game()
            return {"status": "ok", "message": "New game started"}
            
        # Default action is "move" - no need for explicit action field
        fen = job_input.get("fen")
        if not fen:
            return {"error": "Missing 'fen' parameter"}
            
        # Set team if provided
        team = job_input.get("team", "white")
        engine.set_option("Team", team)

        # Mode feeds an NN input plane and is hashed into the position key, so a
        # change invalidates any retained tree. Reusing one across a change would
        # mix evaluations made under two different rule sets.
        mode = job_input.get("mode", "go")
        if engine.current_mode is not None and engine.current_mode != mode:
            engine.new_game()
        engine.set_option("Mode", mode)
        engine.current_mode = mode

        # How many candidate moves to rank. `multipv: true` is still accepted
        # for callers written against the old boolean flag.
        multipv = job_input.get("multipv", 1)
        if isinstance(multipv, bool):
            multipv = 5 if multipv else 1
        multipv = max(1, min(int(multipv), MAX_MULTIPV))
        engine.set_option("MultiPV", str(multipv))

        # Which board the caller is sitting at. The engine plays both, but the
        # ranked lines are grouped by the move on this one.
        analysis_board = int(job_input.get("analysisBoard", 1))
        if analysis_board not in (1, 2):
            return {"error": "analysisBoard must be 1 or 2"}
        engine.set_option("AnalysisBoard", str(analysis_board))

        # Set position - moves can be string or list
        moves = job_input.get("moves", "")
        if isinstance(moves, str) and moves:
            moves = moves.split()
        engine.set_position(fen, moves if moves else None)

        # Search budget: nodes is reproducible across machines, movetime is not.
        nodes = job_input.get("nodes", 0)
        movetime = job_input.get("movetime", 0)

        if nodes:
            nodes = max(1, min(int(nodes), MAX_NODES))
            result = await engine.go(nodes=nodes)
        else:
            movetime = min(int(movetime), MAX_MOVE_TIME_MS) or DEFAULT_MOVE_TIME_MS
            result = await engine.go(movetime=movetime)

        # Parse joint action into individual moves
        bestmove = result.get("bestmove", "(none)")
        moveA, moveB = parse_joint_action(bestmove)

        # The settled ranking: one entry per candidate move on the analysed
        # board, best first.
        lines = collect_final_lines(result.get("raw_info", []), analysis_board)

        response = {
            "bestmove": bestmove,
            "moveA": moveA,
            "moveB": moveB,
            "lines": lines,
            "analysisBoard": analysis_board,
            "eval": result.get("eval"),
            "depth": result.get("depth"),
            "nodes": result.get("nodes"),
            "time_ms": result.get("time_ms"),
            "mate": result.get("mate"),
        }

        # Progressive widening bounds how many distinct root moves exist, and
        # grouping collapses them further, so a short search can return fewer
        # lines than asked for. Say so rather than leaving it to be guessed.
        if len(lines) < multipv:
            response["note"] = (
                f"requested {multipv} lines, search produced {len(lines)}; "
                "increase the search budget for more"
            )

        if job_input.get("includeRawInfo"):
            response["info"] = result.get("raw_info", [])

        return response
            
    except asyncio.CancelledError:
        # Explicit, though CancelledError is a BaseException and so would not be
        # caught below anyway: this must not go down the error path, which kills
        # the engine process. engine.go() has already stopped the search and
        # consumed its bestmove, and the worker stays healthy for the next job.
        # Re-raising is what lets the SDK mark the job cancelled.
        raise

    except Exception as e:
        import traceback
        traceback.print_exc()

        # Try to restart engine on error
        try:
            engine.stop()
        except:
            pass

        return {"error": str(e)}


def handler_sync(job: Dict[str, Any]) -> Dict[str, Any]:
    """Blocking handler() for callers with no event loop.

    The dev server, the self-test and the tests all call the handler directly.
    They gain nothing from cancellation -- there is no RunPod job to cancel --
    so they get a wrapper rather than an async rewrite each.
    """
    return asyncio.run(handler(job))


def parse_joint_action(bestmove: str) -> tuple:
    """
    Parse joint action format "(moveA,moveB)" into individual moves.
    
    Examples:
        "(e2e4,d2d4)" -> ("e2e4", "d2d4")
        "(e2e4,pass)" -> ("e2e4", None)
        "(pass,d2d4)" -> (None, "d2d4")
        "(none)" -> (None, None)
    """
    if not bestmove or bestmove == "(none)":
        return (None, None)
        
    # Remove parentheses
    inner = bestmove.strip("()")
    
    if "," not in inner:
        return (None, None)
        
    parts = inner.split(",")
    if len(parts) != 2:
        return (None, None)
        
    moveA = parts[0].strip() if parts[0].strip() != "pass" else None
    moveB = parts[1].strip() if parts[1].strip() != "pass" else None
    
    return (moveA, moveB)


def _selftest() -> None:
    """One-off local search, for checking a build by hand."""
    test_job = {
        "input": {
            "fen": ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1|"
                    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1"),
            "movetime": 2000,
            "multipv": 3,
            "analysisBoard": 1,
            "team": "white",
        }
    }
    print(f"Engine path: {ENGINE_PATH}")
    print(f"Networks dir: {NETWORKS_DIR}")
    print(json.dumps(handler_sync(test_job), indent=2))


# The Dockerfile runs this file directly (`python -u handler.py`), so __main__
# has to be the serving path. It previously held the self-test while
# runpod.serverless.start() sat in the else branch, reachable only on import --
# so the container ran one test search, printed it, and exited without ever
# serving. Pass --selftest to get the old behaviour deliberately.
if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        _selftest()
    elif runpod is None:
        raise SystemExit(
            "the runpod SDK is not installed; use --selftest, or run "
            "deploy/runpod/dev_server.py for a local HTTP endpoint"
        )
    else:
        # Register with RunPod FIRST, then load the engine in the background.
        #
        # This used to run engine.start() + verify_ready() before
        # runpod.serverless.start(), so that a worker which cannot search would
        # fail at startup rather than accept traffic it cannot serve. That is
        # the right instinct and the wrong place for it here: RunPod expects a
        # worker to connect to the job queue shortly after boot, and a cold
        # TensorRT plan rebuild takes ~4 minutes (measured). The worker spent
        # that time not registered, the platform reaped it as failed, and it
        # restarted into the same rebuild forever.
        #
        # The failure mode was near-undiagnosable from outside: jobs sat at
        # IN_QUEUE with retried=0 and failed=0 because nothing was ever
        # dispatched, /status stayed empty because nothing ever ran, and the
        # worker died early enough that its logs were often dropped. The only
        # visible symptom was "worker exited with exit code 1" on a loop.
        #
        # Registering first keeps the worker alive while it initializes.
        # handler() blocks on _await_engine(), so the guarantee that mattered --
        # never serving a request the engine cannot answer -- is preserved; it
        # is enforced per-request instead of per-process. A cold first request
        # pays the rebuild, which is the cost the old ordering was trying to
        # avoid, and is much the lesser problem.
        threading.Thread(target=_init_engine_background, daemon=True).start()
        runpod.serverless.start({"handler": handler})
