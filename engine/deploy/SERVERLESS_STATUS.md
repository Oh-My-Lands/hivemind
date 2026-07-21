# Serverless deployment status

Last updated 2026-07-21. Target: RunPod serverless, scaling to zero between
requests. Earlier numbers were measured on an RTX 4090 dev pod; anything dated
2026-07-21 comes from the live serverless endpoint on an RTX 2000 Ada.

## Done

- **The container actually serves.** `runpod.serverless.start()` was in an
  `else` branch reachable only on import, so `CMD ["python","-u","handler.py"]`
  ran the self-test and exited. Serving is now `__main__`; `--selftest` is opt-in.
- **The engine loads in the background, and no request is served before it is
  ready.** This inverts an earlier ordering. `engine.start()` + `verify_ready()`
  used to run *before* `serverless.start()`, but a worker that failed there
  exited 1 before ever registering: jobs sat `IN_QUEUE` with `retried=0`,
  `/status` stayed empty, and the logs were usually dropped — the only symptom
  was "worker exited with exit code 1" on a loop. `handler.py` now registers
  first and initializes in a thread; `handler()` blocks on `_await_engine()`, so
  the guarantee that mattered is enforced per-request instead of per-process.
  `verify_ready()` still runs a real 1-node search — `uciok` proves nothing, it
  answers before the network is loaded.

  Consequence to keep in mind: **a worker reporting `ready` says nothing about
  the engine.** See the driver-mismatch entry below.
- **A bad TensorRT plan is no longer fatal and silent.** It used to exit 0 with
  "No engines have been initialized". It now falls back to rebuilding.
- **Plans are named by hardware and TRT version**, e.g.
  `model-…_fp16_b16_sm89_trt10_v1.engine`. A plan for the wrong GPU misses the
  cache and rebuilds rather than being loaded and misbehaving.
- **`ENGINE_CWD`** replaced a hardcoded `/app`, which broke every test off-pod.

- **The slim image serves.** Endpoint `94cvfk6ted0njo` returns real analysis on
  `ghcr.io/oh-my-lands/hivemind-engine:v1` (2026-07-21).

- **A cancelled search now stops.** `handler` is `async def`, and `go()` awaits
  between polls. That is the whole mechanism: the SDK stops a job by cancelling
  its asyncio task (`rp_scale.stop_job` → `task.cancel()`), which can only land
  at an `await`. The old synchronous handler held the event loop for the entire
  search, so the worker could not even read the stop signal until the search was
  over and the GPU time already spent — which is why a `bestmove` was once seen
  arriving 25s after a disconnect.

  The subtle half is `cancel_search()`. UCI `stop` does not abort a search, it
  cuts it short: the engine still prints a `bestmove`. Leave that line in the
  queue and the *next* search reads it first and answers the new position with
  the old position's move. That is a wrong answer, which is worse than the
  wasted GPU time this exists to avoid, so cancellation consumes the bestmove,
  and `go()` drains before searching as a backstop. Both paths are covered by a
  test with a negative control (without the drain, the stale move does leak).

- **The client asks for the cancellation.** The worker honouring a stop signal
  is only half of it: an HTTP disconnect is invisible to RunPod, and stopping a
  job needs an explicit `POST /v2/{endpoint}/cancel/{job_id}`. Holding that id
  means submitting with `/run` rather than `/runsync`, which does not give an id
  up until it answers — exactly the information missing when a client leaves
  mid-search. The proxy (`app/api/engine/run/route.ts` in the frontend repo) now
  submits with `/run`, polls, and cancels both when the client aborts and when
  its own deadline expires.

  Measured against the live endpoint: a 25s search cancelled 3s in reached
  `CANCELLED` **0.3s** later, `executionTime` 2731ms, ~21.7s of GPU time not
  spent — against a `bestmove` that used to arrive 25s after the client had
  gone. The following search returned a correct, uncontaminated result.

- **Never put `/usr/local/cuda/compat` ahead of the host driver.** This cost a
  day. `Dockerfile.slim` prepended it so the bundled forward-compatibility
  `libcuda` (570.86.10) would win over the driver the NVIDIA container runtime
  injects, on the theory that it made the image portable to hosts with older
  drivers. But CUDA forward compatibility is only supported on **data-center**
  GPUs; on the endpoint's RTX 2000 Ada the bundled `libcuda` and the host kernel
  module are an invalid pair, and the first CUDA call fails with **error 803,
  `cudaErrorSystemDriverMismatch`**. The engine died on `cudaGetDeviceCount` in
  ~250 ms, long before TensorRT.

  Two things made this hard to see, both worth remembering:

  - The API returned only `engine unavailable: BrokenPipeError`. The broken pipe
    is just `_send("uci")` writing to an already-dead process — true of *every*
    startup failure and diagnostic of none. The real message was in the worker's
    **Container** log (not the System log, which shows only image pull).
  - **RunPod reported the worker healthy.** Its own GPU test binary failed with
    the same 803, but it demoted that to a warning, fell back to a check that
    merely counts GPUs ("Memory allocation NOT tested"), and logged "All fitness
    checks passed."

  The fat `Dockerfile` never set `LD_LIBRARY_PATH` at all, which is why the dev
  pod never saw this: the bug arrived with the slim build.

## Measured

**Cold start, model load:** 0.6s from a cached plan, **235.8s** building from
ONNX. Baking a plan is close to mandatory — a cold worker without one blows the
job timeout on its first request.

**End-to-end on the live endpoint** (2026-07-21, RTX 2000 Ada, slim image,
2500 ms `movetime`): cold `delayTime` **6.3s**, warm **1.3s** then **0.98s**;
`executionTime` steady at **2.65–2.75s**. The plan is baked, so cold start is
container pull/start plus the 0.6s load — not a rebuild.

**Search throughput on that GPU:** ~**10,000 nps**, 25k nodes and depth 11–12 in
a 2.5s search. This is the number to beat when evaluating a bigger sm89 part
(open decision 3).

**Where the money actually goes.** Billing runs from worker start to worker
stop, rounded up per second, so a cold isolated search costs roughly 5.5s of
startup + 2.7s of search + the idle timeout — about **13s billed for 2.7s of
searching**, only ~20% of it useful. That ratio, not the GPU, is the thing worth
attacking. `idleTimeout` was raised 5s → **15s** on 2026-07-21: a follow-up
request inside that window skips a startup that would have been billed anyway,
so it is roughly cost-neutral at the break-even (~13s between requests) and buys
warm latency. Past ~15s it stops being neutral and starts funding idle time.

**Image size:** the built image is **12.9 GB**; the base
(`nvcr.io/nvidia/tensorrt:25.01-py3`) is **12.4 GB** of that — 7.47 GB
compressed, 38 layers, one of them 6.27 GB. **Our own layers add ~0.5 GB**
(`/app` is 144 MB: 50 MB build output, 90 MB `networks/`, 3.9 MB source).

Inside the base: `/usr/lib/x86_64-linux-gnu` 6.5 GB, `/usr/local` (CUDA
toolkit) 4.4 GB, `/opt` 573 MB.

| Library | Size | Needed at runtime? |
|---|---|---|
| `libnvinfer_builder_resource_win.so.10.8.0` | 1.9 GB | **Never** — the *Windows* builder, in a Linux image |
| `libnvinfer_builder_resource.so.10.8.0` | 1.9 GB | Only to *build* plans |
| `libnvinfer.so.10` | 638 MB | Yes |
| `libnvinfer_lean.so.10` | 106 MB | Runtime-only TRT; would need linking against it |
| `libnvonnxparser.so.10` | 4.3 MB | Yes (currently) |

`ldd` on the binary links exactly two NVIDIA libs: `libnvinfer.so.10` and
`libnvonnxparser.so.10`. Everything else is CUDA that TRT `dlopen`s.

Candidate runtime bases, compressed: `nvidia/cuda:12.8.0-runtime-ubuntu24.04`
**2.17 GB**, `nvidia/cuda:12.8.0-base-ubuntu24.04` **0.10 GB**.

**Correction to an earlier conclusion:** a multi-stage build to drop
`build-essential`/`cmake`/`ninja`/source is worth **almost nothing** — that is
~4% of the image. The lever is the *runtime base*, not the toolchain.

## Open decisions

1. ~~**Slim runtime base.**~~ **Done, and it beat the estimate.** Built on
   `tensorrt:25.01-py3`, running on `cuda:12.8.0-base` (not `-runtime`) with only
   `libnvinfer` + `libnvonnxparser` copied over: **0.62 GB compressed across 13
   layers**, against the fat image's 7.47 GB across 38. Largest layer is
   `libnvinfer` at 339 MB. Verified by reading the published manifest, so these
   are the shipped numbers, not a local build's.
2. **Dropping the builder resource kills the rebuild fallback.** Without the
   1.9 GB `libnvinfer_builder_resource.so`, a plan that misses its arch cannot
   be rebuilt: hard startup failure instead of slow recovery. Arguably correct
   for serverless (a 236s rebuild blows the timeout anyway, so failing loudly at
   worker init is better), but it inverts a deliberate earlier decision and
   should be chosen, not fallen into. The rebuild path stays useful on the dev pod.
3. **Pin the endpoint's GPU type.** Done: `94cvfk6ted0njo` is pinned to
   `NVIDIA RTX 2000 Ada Generation`, which is sm89 and matches the baked plan.
   Scheduling onto an A100 (sm80) or H100 (sm90) would miss the cache and
   rebuild → timeout. `CMAKE_CUDA_ARCHITECTURES="80;86;89;90"` covers the
   *binary* across those, but TRT plans are not portable — one plan per arch
   would be needed.

   Still open: **is RTX 2000 Ada the right sm89 part?** It is the weakest of
   them. Measured ~9,900 nps / 24.7k nodes per 2.5 s search. Worth comparing
   against a 4090 or L40S before this is settled — cost per search matters more
   than cost per second.
4. **`networks/` is gitignored** (`engine/.gitignore:80`), correctly — it holds a
   61 MB ONNX and a 32 MB plan. But that means the image build depends on an
   untracked local file: a fresh clone produces a plan-less image that works in
   testing and times out on its first cold worker. Wants a fetch script, or a
   build-time check that fails loudly when no plan matching the target arch is
   present.

## Not done

- **Publish the networks release.** `deploy/fetch_networks.sh` is written and
  tested but fetches from a release that does not exist yet; run it once with
  `publish` from a machine holding good files. Until then a fresh clone gets a
  clear error rather than a working download.

- **Concurrency.** `workersMax` is 1, so a second search queues behind the
  first. Raising the cap costs nothing by itself — workers are billed only while
  running — but each additional worker pays its own cold start, and a client
  retry loop can then bill in parallel.

## Security / ops notes

- `dev_server.py` has **no auth and no TLS**; it binds `127.0.0.1`. The SSH
  tunnel is the security boundary, not merely a connectivity convenience.
- `stop.sh` does **not** stop pod billing. Stop or terminate the pod in the
  RunPod console.
- `pod.env` (repo root, not in any git repo) holds the pod host and SSH port.
  Gitignore it before `git init` there. It goes stale whenever a pod is
  recreated — RunPod reassigns host and port, and a stale entry looks like
  "Connection refused" rather than anything about the address being wrong.

- **Deploy by digest, never by a mutable tag.** The endpoint now runs
  `ghcr.io/oh-my-lands/hivemind-engine@sha256:6c103b9b…`, not `:v1`, and it
  should stay that way. Pushing a corrected image over the existing `:v1` tag
  did **not** get picked up: workers kept serving a cached older `:v1` and went
  on failing with error 803, which reads exactly like "the fix didn't work". The
  registry had the right image the whole time. Two signals proved the mismatch —
  the failure carried no `_startup_diagnostic` suffix, and 803 is impossible in
  an image whose `LD_LIBRARY_PATH` lacks the compat prefix.

  With a mutable tag you cannot tell which code is running. Tag each build
  uniquely (git SHA) or pin the digest; the `slim`, `slim-avxfix`, `slim-diag2`
  tags in the local image list are what fighting this looks like.

  The temporary `LD_LIBRARY_PATH` template environment variable used to test the
  fix has been deleted — the image is the single source of truth again.

- API keys: a job-scoped key can call `/run`, `/runsync` and `/health` but 401s
  on `rest.runpod.io` and the GraphQL API, so it cannot read endpoint config or
  templates. Reading those needs a full-scope key — worth knowing before
  debugging, and worth *not* leaving a full-scope key lying around afterward.
