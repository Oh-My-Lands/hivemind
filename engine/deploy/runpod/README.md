# Hivemind RunPod Serverless Deployment

Runs the Hivemind bughouse engine as a scale-to-zero GPU endpoint on RunPod.

`handler.py` wraps the compiled `hivemind` binary as a long-lived UCI subprocess
and translates RunPod job JSON into UCI commands. `Dockerfile.slim` is the image
— it builds on NVIDIA's TensorRT development base and *runs* on the much smaller
CUDA base (~0.55 GB compressed against 7.47 GB).

For what has been measured, what is still open, and the failures that produced
the sharper comments in these files, see [`../SERVERLESS_STATUS.md`](../SERVERLESS_STATUS.md).

## Prerequisites

- `podman` (or `ENGINE_CMD=docker`)
- Push access to `ghcr.io/oh-my-lands`
- A RunPod account

## Build and deploy

```bash
cd engine
./deploy/fetch_networks.sh        # networks/ is gitignored; see below
./deploy/runpod/build_and_push.sh # tag defaults to the git SHA
```

The script builds, verifies the image, pushes, and prints a digest. **Deploy the
digest, not the tag.** RunPod console → Serverless → your endpoint → Edit
Endpoint → Container Image; saving recycles the workers, which is what makes
them pull.

```
ghcr.io/oh-my-lands/hivemind-engine@sha256:...
```

Tags are mutable and workers cache them. A `:v1` was once pushed over in place
and the workers went on serving the old copy — the endpoint kept failing with an
error the new image could not produce, which reads as "the fix didn't work"
rather than "you are not running the fix". The script never reuses a tag, and a
digest makes the question unaskable.

After it comes up, **run a real search**. A worker reporting `ready` says
nothing about whether the engine loaded — see Troubleshooting.

### networks/

`engine/networks/` is gitignored (~94 MB of binaries) and holds two files, both
required:

| File | Role |
|---|---|
| `model-….onnx` | the model — portable, same bytes everywhere |
| `model-…_fp16_b16_sm89_trt10_v1.engine` | a TensorRT **plan**: the model compiled to GPU-specific kernels |

Baking the plan is close to mandatory. Loading one takes **0.6s**; building one
from the ONNX takes **236s**, which blows the job timeout on a cold worker — and
does so on *every* cold start, because nothing persists. `build_and_push.sh`
refuses to build without a plan present.

Shipping the plan alone also fails, confusingly: `getEnginePath()` derives the
plan's filename from the ONNX stem, so a missing ONNX surfaces as "doesn't
contain a file ending with .onnx" rather than anything about the plan.

A plan is bound to one GPU architecture, encoded in the filename. `sm89` covers
the 4090 / L40S / L4 / RTX 2000 Ada. On any other GPU the plan misses the cache
and is rebuilt — which is the 236s path. **Pin the endpoint's GPU types to match
the plan you shipped.**

## Endpoint configuration

| Setting | Value | Why |
|---|---|---|
| Container image | the digest printed above | not the tag |
| GPU types | sm89 only, matching the plan | anything else rebuilds the plan |
| Min workers | 0 | scale to zero |
| Max workers | 5 | adjust to traffic |
| Idle timeout | 30–60s | keeps a worker warm between moves in a game |

No network volume. The model ships inside the image; `NETWORKS_DIR` defaults to
`/app/networks` and does not need setting.

## Calling the endpoint

**Submit with `/run`, not `/runsync`.** An HTTP disconnect is invisible to
RunPod, so stopping a search needs an explicit
`POST /v2/{endpoint}/cancel/{job_id}` — and only `/run` returns a job id.
`/runsync` gives you no way to stop a search the client has walked away from,
and the GPU time is billed regardless. Poll `/status/{job_id}`, and cancel both
when the client aborts and when your own deadline expires.

Measured: a 25s search cancelled 3s in reached `CANCELLED` 0.3s later, with
~21.7s of GPU time not spent.

`/runsync` is fine for a one-off check by hand:

```bash
curl -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1|rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1",
      "nodes": 20000,
      "multipv": 3,
      "analysisBoard": 1
    }
  }'
```

### Request

All fields live under `input`.

| Field | Type | Default | Description |
|---|---|---|---|
| `fen` | string | **required** | Both boards, separated by `\|`. Crazyhouse-style pockets: `…RNBQKBNR[] w KQkq - 0 1` |
| `nodes` | int | — | MCTS nodes. Capped at 50,000,000 |
| `movetime` | int | 1000 | Milliseconds. Capped at 30,000. Ignored when `nodes` is set |
| `moves` | string \| array | — | Moves from the given position, board-prefixed: `"1e2e4 2d7d5"` |
| `multipv` | int | 1 | Candidate moves to rank, capped at 100. `true`/`false` still accepted, meaning 5 and 1 |
| `analysisBoard` | 1 \| 2 | 1 | Which board you sit at. The engine plays both; lines are grouped by the move on this one |
| `team` | `"white"` \| `"black"` | `"white"` | Which team you play |
| `mode` | `"go"` \| `"sit"` | `"go"` | `sit` is the time-advantage ruleset. Changing it resets the tree — it feeds an NN input plane and is hashed into the position key |
| `action` | string | `"move"` | Only `"newgame"` is distinct; it resets engine state and returns immediately. Any other value takes the search path |
| `includeRawInfo` | bool | false | Adds the raw UCI `info` lines as `info` |

`nodes` beats `movetime` for analysis: it fixes the work and lets the clock
vary, rather than the reverse. Note that it does **not** fix the *answer* —
measured across seven 50k-node runs on one position, `nodes` was identical every
time while `bestmove` came back `e2e4` twice and `d2d4` five times. The search
is nondeterministic run to run. Do not build caching or test assertions on the
assumption that a repeated search repeats.

### Response

```json
{
  "bestmove": "(e2e4,d2d4)",
  "moveA": "e2e4",
  "moveB": "d2d4",
  "analysisBoard": 1,
  "eval": 0.15,
  "depth": 12,
  "nodes": 20016,
  "time_ms": 2272,
  "mate": null,
  "lines": [ ... ]
}
```

`moveA`/`moveB` are `bestmove` split, with `pass` mapped to `null`.

Each entry in `lines` is one candidate move on the analysed board, best first:

| Field | Description |
|---|---|
| `move` | the candidate on `analysisBoard` — what a UI should show |
| `partnerMove` | its half on the other board |
| `score` | `{"kind": "cp"\|"mate", "value": int}` |
| `pv` | principal variation, one `{"a", "b"}` pair per ply |
| `multipv` | rank, 1-based |
| `depth`, `visits`, `nodes`, `nps`, `time`, `q`, `prior` | as emitted, when present |

A `note` field appears when fewer lines came back than `multipv` asked for.
Progressive widening bounds how many distinct root moves exist and grouping
collapses them further, so a short search legitimately returns fewer.

Engine and validation errors come back as `{"error": "..."}` *inside* a
successful response — check the field, not the HTTP status.

## Local development

`dev_server.py` wraps the *same* handler function behind HTTP, so the frontend
hits the same request and response shape it will get on RunPod and the two
cannot drift. Run it on a GPU pod; deploying then changes only the URL.

```bash
cd /workspace/hivemind/engine
python3 deploy/runpod/dev_server.py --port 8080

curl -s localhost:8080/run -H 'content-type: application/json' -d '{
  "input": {"fen": "<fenA>|<fenB>", "movetime": 3000, "multipv": 5}
}' | python3 -m json.tool
```

It serves `POST /run` and `/runsync` interchangeably (both are synchronous
locally) and wraps the result as `{"output": …, "status": "COMPLETED"}`, matching
RunPod. `GET /health` for liveness. No auth, no TLS — do not put it on a public
port.

One search and exit, for checking a build by hand:

```bash
python3 deploy/runpod/handler.py --selftest
```

Off-pod, set `ENGINE_PATH`, `NETWORKS_DIR` and `ENGINE_CWD` to your local build.

## Troubleshooting

**`engine unavailable: BrokenPipeError`** — this is not diagnostic. It is
`_send("uci")` writing to an already-dead process, true of every startup failure.
The real message is in the worker's **Container** log; the System log shows only
the image pull.

**A worker is `ready` but every job fails.** RunPod will report a worker healthy
with an unusable GPU: its own GPU test binary failing gets demoted to a warning,
it falls back to merely counting GPUs, and it logs "All fitness checks passed".
Never take `ready` as evidence the engine loaded — run a search.

**CUDA error 803 / `cudaErrorSystemDriverMismatch`.** Something has put
`/usr/local/cuda/compat` ahead of the host driver on `LD_LIBRARY_PATH`. CUDA
forward compatibility is data-center-GPU-only; anywhere else the bundled
`libcuda` and the host kernel module are an invalid pair and the first CUDA call
dies in ~250 ms, before TensorRT is reached. `build_and_push.sh` checks for this
and refuses to push. Cost a day the first time.

**"CUDA driver version is insufficient"** — the honest opposite failure, on a
host whose driver predates 12.8. Pin the endpoint's GPU types rather than
reordering `LD_LIBRARY_PATH`.

**Jobs sit at `IN_QUEUE` with `retried=0`, `/status` empty, "worker exited with
exit code 1" on a loop.** The worker is dying before it registers. Historically
this was the engine loading *before* `serverless.start()`; it now registers
first and loads in a thread, so a recurrence means the process is failing
earlier than that.

**First search takes ~4 minutes, then is fast.** The plan missed the cache and
was rebuilt — wrong GPU architecture for the baked plan. Check the endpoint's
GPU pinning against the `_sm89_` in the plan filename.

**Engine fails at init on the slim image.** Possibly cuDNN tactics baked into
the plan — the build stage has cuDNN and the runtime stage does not. See the
caveat in `Dockerfile.slim`.
