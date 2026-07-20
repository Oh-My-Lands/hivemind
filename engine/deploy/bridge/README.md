# WebSocket bridge

Streams engine analysis to a browser client over a WebSocket.

This is a **different deployment shape** from `../runpod/`, which is why it
lives in its own directory. The serverless handler is request/response: it
fires `go movetime` and blocks collecting output until `bestmove`. Analysis
wants a long-lived search whose `info` lines arrive as they are produced and
which can be stopped at any moment, so this runs as a persistent server on a
GPU pod rather than as a serverless worker.

## Running

```bash
pip install -r requirements.txt

# The engine looks for ./networks relative to its working directory.
cd /workspace/hivemind/engine
python3 deploy/bridge/ws_bridge.py --engine ./build/hivemind --engine-cwd .
```

Options: `--host`, `--port` (default 8765), `--idle-timeout` (default 900s),
`--verbose`. Each is also readable from an environment variable — `BRIDGE_HOST`,
`BRIDGE_PORT`, `ENGINE_PATH`, `ENGINE_CWD`, `IDLE_TIMEOUT`.

The first run on a new GPU builds the TensorRT plan from the ONNX, which takes
several minutes. It is cached beside the weights, so later starts are quick.

## Protocol

Client to server:

```jsonc
{"type": "position", "fen": "<fenA>|<fenB>", "moves": ["1e2e4", "2d7d5"]}
{"type": "analyze",  "fen": "<fenA>|<fenB>",   // optional; reuses last position
                     "multipv": 5,
                     "analysisBoard": 1,       // 1 = board A, 2 = board B
                     "team": "white",
                     "mode": "go",             // or "sit"
                     "movetime": 5000}         // or "nodes": N, or "infinite": true
{"type": "stop"}
{"type": "ping"}
```

Server to client:

```jsonc
{"type": "ready"}
{"type": "searchStarted", "searchId": 3}
{"type": "info", "searchId": 3, "depth": 12, "multipv": 2,
                 "score": {"kind": "cp", "value": -6},
                 "q": -0.0219, "prior": 0.4569, "visits": 21179,
                 "nodes": 47264, "nps": 13864, "time": 3409,
                 "pv": [{"a": "e2e4", "b": null}, {"a": "e7e5", "b": "e2e4"}]}
{"type": "infoString", "message": "Early stopping: saved 593ms"}
{"type": "bestmove", "searchId": 3, "moveA": "d2d4", "moveB": null}
{"type": "error", "message": "..."}
{"type": "pong"}
```

`null` in a PV entry or bestmove means "sit" — a first-class action in this
engine, not a missing move.

## Things worth knowing

**Moves are board-prefixed.** `1e2e4` is board A, `2e2e4` is board B. Drops use
Fairy-Stockfish's form, so a prefixed drop looks like `1P@e6`.

**FENs must carry both boards**, joined by `|`, with reserves inline in `[...]`.
The frontend's `app/utils/engine/bughouseFen.ts` produces this; note that the
engine does *not* reject a malformed FEN, it silently builds a nonsense
position from whatever parsed, so validate before sending.

**`searchId` matters more than it looks.** Both `go` and `ucinewgame`
implicitly stop a running search, so a `bestmove` can arrive for a search the
client already moved on from. Every streamed message carries the id of the
search it belongs to; a client that renders whatever arrived last will show
results from a stale position. Ignore anything whose `searchId` is not current.

**Frames are coalesced, not queued.** Analysis output is a stream of
replacements — a newer line for PV slot 2 makes the previous one worthless — so
an undelivered frame is overwritten by its successor. A slow client sees fewer,
current frames rather than a growing backlog. `bestmove` and errors are never
dropped.

**One engine process per connection.** Simple and isolated, but each process
holds GPU memory and a TensorRT plan. A multi-user deployment wants a pool.

**Mode changes force a fresh tree.** `Mode` feeds an NN input plane and is
hashed into the position key, so switching between `go` and `sit` triggers a
`ucinewgame` rather than continuing the existing search tree.

## Tests

```bash
pytest test_ws_bridge.py                      # parser and attribution only

HIVEMIND_ENGINE=/workspace/hivemind/engine/build/hivemind \
HIVEMIND_ENGINE_CWD=/workspace/hivemind/engine \
  pytest test_ws_bridge.py                    # adds end-to-end, needs a GPU
```

The parser tests use `info` lines captured verbatim from the built engine
rather than hand-written approximations, so they track what it actually emits.
