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

**Search budget: prefer `nodes` for analysis.** This is an analysis tool, so it wants
a *quality* guarantee, not a latency one. `movetime` bounds the wall clock and
lets the result vary; `nodes` fixes the work and lets the clock vary. Measured
2026-07-21 at `nodes: 20000`, one warm worker:

| Position | `executionTime` | Nodes | nps | Depth |
|---|---|---|---|---|
| Opening | 2272 ms | **20,016** | 9,821 | 11 |
| Sharp middlegame | 2344 ms | **20,016** | 9,283 | 15 |
| Tactical | 2887 ms | **20,016** | 7,547 | 14 |
| Pawn endgame | 2121 ms | **20,016** | 10,322 | 14 |

The node count is **identical across all four** — not approximately, exactly.

**But the reproducibility conclusion originally drawn from that was wrong, and
is retracted here.** It said a result could therefore be "cached by position,
diffed against an older run, and asserted on in a test." Fixed `nodes` pins the
*work*, not the *answer*. Measured 2026-07-21, seven runs at `nodes: 50000` on
the opening position, `multipv: 3`:

- `nodes` was exactly **50,016** every run.
- `visits` on the top line was **never the same twice**: 22,399 / 22,475 /
  22,527 / 22,543 / 22,556 / 22,568 / 22,592.
- `bestmove` came back `e2e4` twice and `d2d4` five times.

Tree retention was the obvious suspect and was ruled out: two runs under
*identical* conditions — explicit `newgame` reset, then one 50k search — gave
`d2d4` and `e2e4` respectively, while a run with a 200k tree deliberately left
in place agreed with the fresh one. The variation is run-to-run nondeterminism
in the search itself, most likely parallel MCTS threads racing on expansion.

So the `movetime` complaint recorded here — "the same position gave `d2d4` on
one run and `e2e4` on the next, which is disqualifying" — applies to `nodes`
too. That was never the difference between them.

**What survives, and is still the reason to prefer `nodes`:** the work is fixed,
so cost is predictable and billing is bounded; and the work is machine-
independent, so the same request means the same effort on any GPU. What does
*not* survive: any test asserting on `bestmove`, and any workflow that diffs two
runs and reads a changed top line as a real change. Caching by position is still
worth doing as a cost measure — you serve whatever you stored — but it is not
recovering a canonical answer, because there isn't one.

Note the two candidates were within 0.005 q of each other (`d2d4` ≈ −0.0168,
`e2e4` ≈ −0.0217), and `bestmove` follows the **most-visited** move rather than
the highest-q one. The flip is the search resolving a coin toss between equals.
Expect it to be visible in the UI whenever the top two moves are this close.

And wall-clock
spread is only **1.36x** across position types, which is what makes `nodes` safe
despite its one real weakness: it has no graceful degradation. A search that
overruns `executionTimeoutMs` returns *nothing* and still bills for every second,
whereas `movetime` cannot overrun by construction. Size budgets off the slow end
(~7,500 nps), not the average:

| Nodes | Worst case | Use |
|---|---|---|
| 20k | 2.7s | Quick eval |
| 50k | 6.7s | Interactive default |
| 100k | 13s | Solid analysis |
| 250k | 33s | Deep — **not expressible** under `movetime` |
| 500k | 67s | Deepest sane |

Note the last two rows: `MAX_MOVE_TIME_MS` caps `movetime` at 30s, so anything
deeper is reachable *only* through `nodes`. The real ceiling is the endpoint's
`executionTimeoutMs` (600,000 ms — about 4.5M nodes at the slow end), not the
handler's `MAX_NODES` of 50M, which is unreachable by ~8x and protects nothing.

Depth is not the guarantee: the same 20k nodes buys depth 11 in the opening and
15 in the endgame. Search *effort* is fixed; the depth it reaches is not. Worth
knowing before surfacing depth in a UI.

Reserve `movetime` for a live-play path with a clock, where a latency bound
matters more than reproducibility.

**Pass `nodes` explicitly.** `handler.py` takes `nodes` in preference to
`movetime` when both are sent, but the no-argument default is
`DEFAULT_MOVE_TIME_MS` (1000 ms) — a request with neither gets a `movetime`
search, and none of the reproducibility above applies to it.

**Where the money actually goes.** Billing runs from worker start to worker
stop, rounded up per second, so a cold isolated search costs roughly 5.5s of
startup + 2.7s of search + the idle timeout — about **13s billed for 2.7s of
searching**, only ~20% of it useful. That ratio, not the GPU, is the thing worth
attacking. `idleTimeout` was raised 5s → **15s** on 2026-07-21: a follow-up
request inside that window skips a startup that would have been billed anyway,
so it is roughly cost-neutral at the break-even (~13s between requests) and buys
warm latency. Past ~15s it stops being neutral and starts funding idle time.

**The fixed overhead makes small budgets the worst value, not the best.** The
workload here is interactive single-position analysis, not game review, so
batching positions per job — the obvious attack on the ratio — does not apply to
the main path. What does apply: ~20.5s of overhead (5.5s startup + 15s
`idleTimeout`) lands on every request regardless of budget, so against *billed*
seconds nodes are heavily discounted, even though against wall clock they are
not. Worst case (7,221 nps), isolated cold request:

| Budget | Search | Billed | Useful | Billed ms/node |
|---|---|---|---|---|
| 5k | 0.7s | 21.2s | 3% | **4.24** |
| 20k | 2.8s | 23.3s | 12% | 1.16 |
| 50k | 6.9s | 27.4s | 25% | 0.55 |
| 200k | 27.7s | 48.2s | 57% | 0.24 |
| 500k | 69.2s | 89.7s | 77% | **0.18** |

5k cost 44% of what 200k cost and returned 2.5% of the analysis — 18x worse per
node. **Acted on 2026-07-21:** `NODE_OPTIONS` in `EngineLinesPanel.tsx` dropped
the 5k and 20k tiers (floor is now the 50k default) and gained 500k.

**A doubling buys exactly one ply.** Measured 2026-07-21 through the proxy,
opening position, `multipv: 3`, warm worker — 50k → depth 12 (5.1s search, 7s
wall), 100k → depth 13 (11.4s, 14s), 200k → depth 14 (21.1s, 22s). The rungs are
evenly spaced in wall clock, so a 100k tier was considered and rejected: it fills
no gap and is dominated on cost by 200k (0.34 vs 0.24 billed ms/node).

Note also that the top move flipped between 100k (`e2e4`, q −0.0232) and 200k
(`d2d4`, q −0.0216) while the two stayed within 0.004 q of each other. In the
opening those moves are equal within noise, and a ranking flip across budgets is
the search resolving a coin toss, not finding something. Worth knowing before
reading a changed top line as a result.

The hover text was wrong twice over, which is what hid this. It showed *search*
time, which is not the cost; and its figures — 0.4s/1.4s/3.5s/14s — were a flat
14,286 nps from the **RTX 4090 dev pod**, roughly 2x the serverless RTX 2000 Ada,
so every tier looked half its true duration. Labels now carry billed cost, and
depth was dropped from them entirely: at a fixed 20k nodes the same search
reaches depth 11 in the opening and 15 in a pawn endgame, so a per-budget depth
number cannot be honest.

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

   Arch is not the only portability constraint, and not the one that bites
   first — the host CUDA driver is. See decision 3 for the measured failure.
3. **Pin the endpoint's GPU type.** Done: `94cvfk6ted0njo` is pinned to
   `NVIDIA RTX 2000 Ada Generation`, which is sm89 and matches the baked plan.
   Scheduling onto an A100 (sm80) or H100 (sm90) would miss the cache and
   rebuild → timeout. `CMAKE_CUDA_ARCHITECTURES="80;86;89;90"` covers the
   *binary* across those, but TRT plans are not portable — one plan per arch
   would be needed.

   Note the config has two representations and only one is trustworthy. REST
   (`GET rest.runpod.io/v1/endpoints/{id}`) reports the real rule:
   `"gpuTypeIds": ["NVIDIA RTX 2000 Ada Generation"]`. GraphQL renders the same
   config as `gpuIds: "AMPERE_16,-NVIDIA RTX 4000 Ada Generation,-NVIDIA RTX
   A4000,-NVIDIA RTX A4500"` — a pool minus three cards, which reads like a much
   wider selection than is actually in force. Read the REST view. Also note
   `gpuIds` only *accepts* pool ids; individual cards can only be subtracted
   with `-`, so a positive per-card allowlist cannot be expressed there at all.

   **Is RTX 2000 Ada the right sm89 part? No — the endpoint moved to RTX 4000
   Ada on 2026-07-21. See "Switched to RTX 4000 Ada" below.** The claim this
   entry used to make — that the question was moot because the image cannot run
   anywhere else without a rebuild — was too broad. It generalised from one
   card. The 4090 specifically cannot boot; another sm89 part in the *same
   billing pool* boots fine and is measurably faster.

   The 4090 half of the finding still stands, and is worth keeping. Tried it:
   cloned the template at the deployed digest onto a throwaway endpoint pinned
   to `NVIDIA GeForce RTX 4090`. The image pulled fine and then failed at
   container init, before any engine code ran:

   ```
   nvidia-container-cli: requirement error: unsatisfied condition: cuda>=12.8,
   please update your driver to a newer version, or use an earlier cuda container
   ```

   It crash-looped on ~17 s retries until the endpoint was deleted.

   **GPU portability here is gated by the host driver, not by architecture.**
   sm89 is necessary but not sufficient. The base is `tensorrt:25.01-py3` =
   CUDA 12.8; the RTX 2000 Ada pool has drivers new enough, the 4090 pool does
   not. This is invisible until a worker tries to boot — it is not a plan-load
   failure and looks nothing like decision 2's arch miss. Testing another card
   therefore means rebuilding on an older CUDA base *and* regenerating the plan
   against that TensorRT version, not a config change.

   A *bigger* part is still not worth buying. At fixed nodes a 4090 must be
   >2.9x the RTX 2000 Ada to break even on price (L40S >4.1x), and a 61 MB net
   at batch 16 fp16 is bound by kernel-launch latency and CPU-side MCTS tree
   work rather than SM count, so it will not clear that bar. **This prediction
   was then confirmed from the other direction:** the RTX 4000 Ada has ~2.2x the
   CUDA cores of the RTX 2000 Ada and delivered only **1.28x** the throughput.
   Cores do not convert here. Anything priced by capability is a bad trade.

   ### Switched to RTX 4000 Ada (2026-07-21)

   The win was not a bigger card, it was a **free** one. `gpuIds` read
   `AMPERE_16,-NVIDIA RTX 4000 Ada Generation,-NVIDIA RTX A4000,-NVIDIA RTX
   A4500` — the AMPERE_16 pool minus three cards. Serverless bills **per pool**,
   so every member costs the same $0.24/hr; the wildly varying *pod* prices
   ($0.17–$0.50 across these four) are irrelevant and actively misleading here.
   One of the excluded cards was sm89 and simply faster for nothing.

   Both cards, same six positions, same image digest, `nodes: 20000`,
   `multipv: 3`, every run returning exactly 20,016 nodes (2026-07-21):

   | Position | 2000 Ada nps | 4000 Ada nps | speedup | depth |
   |---|---|---|---|---|
   | opening | 9,805 | 11,263 | 1.15x | 11–12 |
   | italian | 9,348 | 11,470 | 1.23x | 12–13 |
   | tactical | 8,579 | 10,507 | 1.22x | 16 |
   | **big-pocket midgame** | **7,077** | **8,196** | **1.16x** | 16 |
   | tactical2 | 9,370 | 11,450 | 1.22x | 15–16 |
   | pawn endgame | 10,813 | 15,027 | 1.39x | 12 |

   **Worst case moved 7,077 → 8,196 nps.** Average speedup ~1.23x, range
   1.15–1.39x. An earlier opening-only measurement said 1.28x and was
   optimistic — do not size anything off a single position.

   **The speedup is smallest exactly where it matters.** Budgets are sized off
   the worst case, and the worst case improved by only **1.16x**, the second-
   lowest figure in the table. This is the CPU-bound thesis showing up again: the
   more the position is limited by host-side tree work, the less a better GPU
   buys.

   **The worst case is pocket size, not tactics.** The slowest position is a
   middlegame holding **18 pieces in hand** (`[QBNPPPPrbbbnnppppp]`), not either
   of the tactical positions. Every pocket piece type can drop on every empty
   square, so the branching factor — and the host-side move generation behind it
   — explodes. An earlier version of this document named the tactical position
   as the worst case at 7,221 nps; a big-pocket position is slightly worse still.
   **When bughouse gets expensive, look at the pockets.**

   Depth at fixed nodes ranged 11–16 across these six, which is the "depth is not
   the guarantee" point in Measured, restated with a wider spread.

   #### Benchmark positions (record these — their absence caused a re-measure)

   The previous baseline recorded *numbers* but not the FENs that produced them,
   so it could not be re-run against a new card and the whole set had to be
   rebuilt. Board B is startpos throughout, to isolate board A. Positions come
   from `engine/tests/`; note the side-to-move must match `team` or the search
   returns an **empty `lines` array** rather than an error.

   ```
   startpos  rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1
   opening   <startpos>                                                    team=white
   italian   r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R[] w KQkq - 0 1        team=white
   tactical  r1bq1b1N/pppbkNpp/4pn2/1B2p3/1b2P3/2N5/PPP2PPP/R1BQK2R[BPqrnnpppp] b KQ - 0 16  team=black
   midgame   r6r/pppk1Ppp/2n1q3/2Nn4/8/2P5/P1P1NPPP/R1B1K2R[QBNPPPPrbbbnnppppp] b KQ - 0 1   team=black
   tactical2 6rk/p5pp/2Q2p2/3p1n2/3P1P2/2N4P/PPP2P1P/6RK[RBr] b - - 0 31                 team=black
   pawnend   4k3/pp3ppp/8/8/8/8/PP3PPP/4K3[] w - - 0 1                                   team=white
   ```

   Rejected as unusable: `6k1/pppbrp2/2p1pQ2/8/2B1P3/2PN4/PPP3K1/8[R] w - - 0 1`
   is a forced mate and terminated at 3,840 nodes, so it measures nothing. Any
   benchmark position must be checked for consuming the full budget first.

   **Do not enable the other two cards in that pool.** `AMPERE_16` is
   mixed-architecture — RTX A4000 and A4500 are Ampere **sm86**, so a worker
   landing there misses the baked sm89 plan and starts the ~236 s ONNX rebuild,
   which blows the job timeout. The original exclusion was right about those two
   and over-broad about the third.

   | GPU | Arch | Baked sm89 plan |
   |---|---|---|
   | RTX 2000 Ada | Ada sm89 | loads |
   | RTX 4000 Ada | Ada sm89 | loads (verified booting) |
   | RTX A4000 | Ampere sm86 | **misses → rebuild → timeout** |
   | RTX A4500 | Ampere sm86 | **misses → rebuild → timeout** |

   The endpoint is now pinned to RTX 4000 Ada alone, so the faster card is
   guaranteed rather than scheduled-if-free. The cost is a smaller availability
   pool; revert by setting `gpuTypeIds` back to `["NVIDIA RTX 2000 Ada
   Generation"]`, or enable both to trade consistency for availability.

   **Re-sized budgets for the RTX 4000 Ada** (worst case 8,196 nps), replacing
   the RTX 2000 Ada figures in **Measured**:

   | Nodes | Worst-case search | Billed (isolated, +20.5 s) | Useful |
   |---|---|---|---|
   | 20k | 2.4 s | 22.9 s | 11% |
   | 50k | 6.1 s | 26.6 s | 23% |
   | 100k | 12.2 s | 32.7 s | 37% |
   | 200k | 24.4 s | 44.9 s | 54% |
   | 500k | 61.0 s | 81.5 s | 75% |

   The execution ceiling for a 600 s `executionTimeoutMs` rises from ~4.33 M to
   **~4.92 M nodes**, so the proxy's `MAX_NODES = 4_000_000` sits further below
   the cliff than before and needs no change.

   Baseline for comparison, if this is ever revisited — four positions at
   `nodes=20000`, reproducible to within 4% across sessions, all returning
   exactly 20,016 nodes:

   | position | exec | depth | effective nps |
   |---|---|---|---|
   | opening | 2269 ms | 11 | 8,822 |
   | middlegame | 2308 ms | 15 | 8,672 |
   | tactical | 2772 ms | 14 | **7,221** |
   | endgame | 2125 ms | 14 | 9,419 |

   Worst case is 7,221 nps, not the ~7,500 estimated earlier. That puts the
   real execution ceiling at ~4.33M nodes for a 600 s `executionTimeoutMs`. A
   `nodes` search that hits the timeout returns nothing and still bills the full
   600 s. **Closed 2026-07-21:** the proxy's `MAX_NODES` in
   `app/api/engine/run/route.ts` was 5,000,000 — ~15% above what can finish —
   and is now 4,000,000, below the ceiling rather than above it.
4. ~~**`networks/` is gitignored**~~ **Closed.** It still is, correctly — 94 MB
   of binaries git would store badly — but the artifacts are now published as
   release `networks-v3.0` and `deploy/fetch_networks.sh` pulls them by pinned
   sha256. A fresh clone runs `./deploy/fetch_networks.sh` and has a complete
   `networks/` in ~10s; a second run is a no-op. `build_and_push.sh` refuses to
   build without a plan and points at the script, so the old failure — an image
   that passes local testing and times out on its first cold worker — cannot be
   produced silently.

   Checksums are pinned rather than trusted: a truncated download looks exactly
   like plan corruption at runtime, which is the same ~236s rebuild.

## Not done

- **Concurrency.** `workersMax` is 1, so a second search queues behind the
  first. Raising the cap costs nothing by itself — workers are billed only while
  running — but each additional worker pays its own cold start, and a client
  retry loop can then bill in parallel.

## Security / ops notes

- **The persistent dev pod is retired (2026-07-21). Serverless is the only
  target.** `start.sh` and `stop.sh` were deleted with it: they existed to
  sequence engine → SSH tunnel → app and to manage a remote process, and a
  serverless endpoint is not a resource you start. Running the app is now
  `npm run dev` (port 3100 lives in `package.json`), and stopping it is Ctrl-C.
  Backups sit at `archive/pod-era/start.sh.pod-era.bak` /
  `stop.sh.pod-era.bak`. That directory is not inside a git repo — they are the
  only copy.

  What survived is the one check that earned its keep: `npm run check:engine`
  (`bughouse-chess-main/scripts/check-engine.sh`) validates `.env.local` and
  health-checks the endpoint. It is deliberately *not* wired into `npm run dev`,
  because UI work does not need a live engine. Run it when analysis misbehaves.

- **The endpoint id and the API key fail independently, and both look generic
  from the app.** A recreated endpoint 404s. A key that is revoked, mistyped or
  from another account 401s. Either way the app shows a failed search;
  `check:engine` names which one.

  **The account has exactly one key and it is full-scope**, so a 401 here means
  revoked, mistyped or wrong account — never a mismatch with the endpoint id.
  RunPod can also issue restricted (job-scoped) keys, which *do* 401 against an
  endpoint they were not issued for (see the API-keys note further down), but
  none currently exist on this account. Verified 2026-07-22: the live key
  returns 200 on `GET https://rest.runpod.io/v1/endpoints` and lists every
  endpoint on the account, which a job-scoped key cannot do.

  An earlier version of this note and of `check-engine.sh` gave the job-scoped
  reading unconditionally, which sends debugging after a cause that cannot
  apply to the key actually in use.

  The consequence that matters: a full-scope key found next to a dead endpoint
  id is **not thereby dead**. It still carries account authority, including
  creating endpoints and incurring billing. Revoking is a console action;
  deleting the file it sits in does nothing.

  What actually happened on 2026-07-21: `.env.local` held `qfxnpl4k3abigf`,
  which had been deleted, while the live id was in `serverless pod keys.txt`.
  The id was the whole failure. Treat any stray copy of a key as live until it
  is revoked in the console — deleting the file does not revoke it.

### Obsolete: pod-era files, moved to `archive/pod-era/`

These lived in the repo root until 2026-07-22; they are now under
`archive/pod-era/`, alongside a README pointing back here. None of them are
referenced by anything in the current workflow. They are kept deliberately, not
overlooked. Nothing breaks if they are deleted — but read the `pod-setup.sh`
entry first, because one of them is dormant rather than dead.

| File | Status |
|---|---|
| `pod-setup.sh` | **Dormant, not dead.** Provisions the engine on a fresh GPU pod. This is the route back if a TensorRT plan ever needs regenerating — see below. |
| `smoke-test.sh` | Obsolete. Runs *on the pod* and verifies engine changes end to end; unusable without one. |
| `pod.env` | Obsolete, and stale by definition — it holds the host and SSH port of a pod that no longer exists. |
| `start.sh.pod-era.bak`, `stop.sh.pod-era.bak` | The deleted orchestration scripts. The analyser root is **not a git repo**, so these are the only copy. |

**Why `pod-setup.sh` is worth keeping.** Baking a TensorRT plan is the one job
serverless cannot do: a rebuild from ONNX takes ~236 s and blows the job
timeout, so a cold worker without a baked plan cannot serve its first request.
Plans are not portable across GPU architecture or TRT version, so a new sm arch,
a new TRT, or a corrupted artifact all mean rebuilding on a real machine. That
path runs through this script.

**Still true of the pod era, if one is ever brought back:** `dev_server.py` has
**no auth and no TLS**; it binds `127.0.0.1`, so the SSH tunnel was the security
boundary rather than a connectivity convenience. `stop.sh` never stopped pod
billing — only the RunPod console does. And `pod.env` went stale on every pod
recreation, surfacing as "Connection refused" rather than as anything about the
address being wrong.

- **Deploy by digest, never by a mutable tag.** The endpoint now runs
  `ghcr.io/oh-my-lands/hivemind-engine@sha256:2f2f073a…`, not `:v1`, and it
  should stay that way. Pushing a corrected image over the existing `:v1` tag
  did **not** get picked up: workers kept serving a cached older `:v1` and went
  on failing with error 803, which reads exactly like "the fix didn't work". The
  registry had the right image the whole time. Two signals proved the mismatch —
  the failure carried no `_startup_diagnostic` suffix, and 803 is impossible in
  an image whose `LD_LIBRARY_PATH` lacks the compat prefix.

  With a mutable tag you cannot tell which code is running. Tag each build
  uniquely (git SHA) or pin the digest; the `slim`, `slim-avxfix`, `slim-diag2`
  tags in the local image list are what fighting this looks like.

  This digest was itself recorded wrong here once (`6c103b9b…`, corrected
  2026-07-21) — a stale digest in the note about pinning digests. Don't retype
  it; read it from the endpoint:

  ```
  curl -s -X POST "https://api.runpod.io/graphql?api_key=$KEY" \
    -H 'Content-Type: application/json' \
    -d '{"query":"query { myself { endpoints { id template { imageName } } } }"}'
  ```

  `GET rest.runpod.io/v1/templates/{id}` 404s for endpoint-bound templates and
  `myself { podTemplates }` returns only RunPod's stock ones, so the nested
  query above is the way in.

  The temporary `LD_LIBRARY_PATH` template environment variable used to test the
  fix has been deleted — the image is the single source of truth again.

- API keys: a job-scoped key can call `/run`, `/runsync` and `/health` but 401s
  on `rest.runpod.io` and the GraphQL API, so it cannot read endpoint config or
  templates. Reading those needs a full-scope key — worth knowing before
  debugging, and worth *not* leaving a full-scope key lying around afterward.
