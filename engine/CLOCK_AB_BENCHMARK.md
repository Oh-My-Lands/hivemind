# Clock-model A/B benchmark

How to measure whether making the search clock-aware makes the engine stronger.

Written 2026-08-03, from the first run of it. Reusable for any change that
alters *how the search reasons about time* while leaving the network alone.

> ### ⚠️ Read before reusing any number in this file. *(2026-08-10)*
>
> Every result recorded here was measured with `ATTACKER_NODE_MULT = 0.5` /
> `DEFENDER_NODE_MULT = 1.5` hardcoded in `ModelEvaluator::playGame`, so the
> configured node budget was never the budget searched. With clocks on the
> multiplier keys off `team_may_sit(currentTeam)` read live, which made it a
> feedback loop: the clock-aware arm halved its own search whenever it won the
> time race.
>
> Pinning both to 1.0 turned the headline **−25 into +93 Elo** (1200 games,
> 751W/11D/438L, `.logs/multfix-20260810/`). The flag battle was unchanged
> (~2.7:1 for clock-awareness either way); board play inverted, 411−584 →
> 624−391 checkmates. **The "wins the clock, loses the board" conclusion this
> document reports was an artifact of the harness.**
>
> The multipliers now default to 1.0 and are printed in every run header. The
> *method* described below is still sound — one net, two search configs, one
> clocked world — but the measured tables (Elo, ply counts, flag rates) all
> predate the fix and need re-measuring before reuse.
>
> **This also withdraws the conclusion this file hands downstream.** The closing
> recommendation — that clock-aware search is "a regression until the network is
> retrained to match" — is dead twice over: there is no regression to explain, and
> the retraining was subsequently done and *lost* (`PHASE2_GATE.md`: continuous
> margin encoding, −81 Elo, worse on the board **and** on the clock). Net position
> after both corrections: clock-aware **search** is worth about +93 Elo at 800
> nodes; a continuous clock **encoding** in the network is not worth keeping. See
> "That conclusion is retracted, twice over" under Results.

---

## What it measures

One network, two search configurations, playing each other:

| arm | sit permission | flagging | clocks charged in tree |
|---|---|---|---|
| **clock-aware** | derived per node from the live clocks | terminal loss | yes |
| **clock-blind** | fixed team bit, as before Phase 4 | not modelled | no |

Both play in the *same* clocked world. The game charges every move and ends on
a flag regardless of which arm the player is. Being blind to the clock is not
exemption from it — the blind arm simply cannot plan around it.

That is the whole design. The interesting comparison is not "clocks on vs
clocks off", it is "in a world with clocks, does modelling them help".

## Why it needed building

The obvious version of this experiment does not work, and it is worth
understanding why before reusing the harness.

`--time-control` sets `EvalSettings::initialTimeDcs`, which is **global to the
match**, and every clock-aware path in the search gates on `board.has_clocks()`
— a property of the one board both players share. So within a game either both
players are clock-aware or neither is:

```
--time-control 0     both blind, identical settings   -> 50% by construction
--time-control 400   both aware, identical settings   -> 50% by construction
```

Neither arm tells you anything. If you run this and get 50%, the harness is not
broken — it is structurally incapable of producing another answer.

`--p1-clocks` / `--p2-clocks` fix it. The blind player's search runs with
`board.set_clocks_visible(false)`, which flips `clocksEnabled` without touching
the clock *values*, turning off every Phase 4 path at once while leaving the
game's clocks intact.

## Running it

Needs a GPU pod with **TensorRT 10 specifically** — v9 lacks
`enqueueV3`/`setTensorAddress`, v11 removed `BuilderFlag::kFP16` and
`engine.cc:83` will not compile against it. `nvcr.io/nvidia/tensorrt:25.01-py3`
ships the right version. See `archive/pod-era/pod-setup.sh` for a full bootstrap.

```bash
./build/hivemind param-eval \
    --model networks/model-*.onnx \
    --games 1200 \
    --time-control 400 \
    --p1-name clock-aware --p1-clocks 1 \
    --p2-name clock-blind --p2-clocks 0 \
    --pgn run.pgn
```

Run it detached (`nohup ... &`) — a 1200-game run is over two hours and you do
not want an SSH drop to kill it.

**Always pass `--pgn`.** The summary alone cannot answer the question you will
actually want answered; see "Reading the result".

### Choosing N

Elo standard error, for the low draw rates bughouse produces:

```
SE(Elo) ≈ 347 / sqrt(N)
```

| N | SE | detects |
|---|---|---|
| 30 | ±63 | nothing — do not draw conclusions |
| 300 | ±20 | a large effect |
| 1200 | ±10 | ~20 Elo reliably |

At ~9-10 games/min, 1200 games is about 2h10m. On a $0.69/hr pod that is ~$1.50,
so there is no reason to run an underpowered match.

### Choosing the time control

`--time-control` is in **deciseconds**. 400 = 40 seconds, not 400.

Measured game lengths at 800 nodes/move:

| tc | avg plies | regime |
|---|---|---|
| 0 | 76.2 | no clock model |
| 40 | 10.0 | every game flags at the same ply — measures arithmetic, not play |
| 100 | 24.0 | same |
| **400** | **58.3** | **clock is a live constraint, games still board-decided** |
| 1800 | 60.7 | clock only ~15% consumed, nearly inert |

**400 is the working choice.** Below it you measure flag timing; at 1800 you
measure almost nothing, because the clock never binds.

Note the tension: real chess.com games are `tc=180` (1800 ds, 3 minutes), so
400 is 4.5× faster than production. It is defensible because the observed flag
rate at 400 (~12.5%) closely matches the real corpus (11% of 132 measured games
ended with someone under 10 s) — but if you are making a claim about production
behaviour rather than about the clock model itself, run 1800 as a third arm.

## Reading the result

The summary gives you win/draw/loss and an Elo estimate:

```
clock-aware:  547 W / 17 D / 636 L      45.6% / 1.4% / 53.0%
Estimated Elo difference: -26
```

**This is not enough.** A clock model can lose overall while doing its job
perfectly, and the summary cannot distinguish that from the model simply being
broken. Split the games by how they ended:

```bash
grep -o '^\[Termination "[^"]*"\]' run.pgn \
  | sed 's/^\[Termination "//;s/"\]$//' \
  | sort | uniq -c | sort -rn
```

```
  197 clock-blind won by checkmate
  134 clock-aware won by checkmate
   34 clock-aware won on time
   14 clock-blind won on time
    4 Game drawn
```

Read it as two separate matches:

- **on the clock** — 34-14, clock-aware wins 70.8%
- **on the board** — 134-197, clock-aware wins 40.5%

That is a completely different story from "−26 Elo". The clock model works; it
converts clock awareness into flag wins.

> The numbers in this worked example are from the pre-fix harness, and the second
> half of the original reading — "it also degrades ordinary board play, and the
> board deficit dominates" — is **retracted**; the board deficit was the node
> handicap. Keep the *method*, which is the point of the section and which held up:
> splitting by termination is what made the aggregate interpretable, and it is
> what later showed the flag result surviving the fix while the board result
> inverted. Read the technique here, not the conclusion.

Significance for a split of size `n`: `SE = sqrt(0.25/n)` in win rate. The 48
flag games give ±7.2%, so 70.8% is ~2.9σ; the 331 board games give ±2.75%, so
40.5% is ~3.5σ. Both real.

Clock annotations in the PGN are the true clocks in seconds when a clock model
is in play, so you can also inspect *how* time was spent, not just who ran out.

## Caveats

Things that are true of this harness and will bite you if you forget them.

**A flag fires one ply late.** The game checks for a flag at the start of a
turn, so a player holding 0.1 s still gets a full move and overruns — you will
see negative clock annotations like `{-0.8}`. The search has no such grace
(`1eb490a` makes a zero clock terminal at the node), so a clock-aware player
believes it dies one ply before it actually does. It is pessimistic about its
own time in a way the blind arm structurally cannot be.

**The baseline arm is not fully blind.** `teamHasTimeAdvantage` in `playGame`
stays clock-derived even for the blind player, because it also feeds
`legal_moves` and both players must agree on the rules of the game they are in.
So the baseline is "cannot reason about the clock", not "knows nothing about
it".

**Node allocation moved with the clock, and this caveat used to say it was
harmless. It was not.** *(corrected 2026-08-10)* The attacker/defender split
(0.5×/1.5× nodes, `ATTACKER_NODE_MULT`) keyed off `teamHasTimeAdvantage`, and the
original text argued that because the bool is clock-derived for *both* players
the handicap is symmetric and cannot favour an arm. That reasoning is wrong, and
it is worth understanding why, because the error is subtle and cost two results.

The multiplier is symmetric in *role* — attacker vs defender — but the roles are
not assigned at random. They are assigned by who is winning the clock, which is
precisely what the treatment under test changes. An arm that succeeds at
acquiring a time advantage thereby moves itself onto the 0.5× side and its
opponent onto the 1.5× side. Symmetry in the rule does not survive a treatment
that predicts which side of the rule you land on: the handicap becomes a function
of the independent variable, applied with a 3× lever.

Both multipliers now default to 1.0. If you are reading a `param-eval` or `eval`
log from before 2026-08-10, it does not record them and the budget it searched
cannot be recovered.

**Fixed nodes is not determinism.** `nodes` pins the work but `visits` and
`bestmove` still vary run to run. Do not expect two runs to match; that is why
N matters.

**Self-play has no clock model at all.** `selfplay.cc` never calls
`set_clocks`. Whatever you conclude here says nothing about training games,
which still cannot show the net a time advantage changing hands.

**The GPU barely matters.** At 800 nodes/move the bottleneck is host-side tree
work, not inference: an RTX 4090 ran 8.51 games/min against an A5000's 8.25, a
3% difference. Rent the cheapest card that provisions, not the fastest.

## Results

Two independent 1200-game runs at `tc=400` against the shipped net
`model-0.97878-0.683-0224-v3.0`. The second added the PGN instrumentation but
changed nothing about play, so the two pool legitimately.

| run | W-D-L (aware) | Elo | z |
|---|---|---|---|
| 1 | 547-17-636 | −25.8 | 2.57 |
| 2 | 568-10-622 | −15.6 | **1.56** |
| **pooled** | 1115-27-1258 | **−20.7** | **2.92** |

**Note run 2 on its own is not significant** (p ≈ 0.12). Two runs of the same
configuration landed 10 Elo apart, which is what an SE of ±10 looks like in
practice and a useful warning against reading a single 1200-game match as
precise. Only pooled (n=2400, p ≈ 0.003) is the aggregate effect established.

The decomposition is far stronger than the aggregate, and replicated across
both runs (run 2 shown):

| | aware-blind | n | aware win rate | z |
|---|---|---|---|---|
| on the clock | 116-43 | 159 | **73.0%** | 5.79 |
| on the board | 452-579 | 1031 | **43.8%** | 3.96 |

Flag rate 13.3%. Clock-aware gains +73 games on the clock and loses −127 on the
board, netting −54.

**This is the result worth carrying forward, not the Elo.** The aggregate is a
weakly-powered difference of two large opposing effects, so it is noisy by
construction — which is exactly why runs 1 and 2 disagree on it while agreeing
closely on the split. If you rerun this benchmark, report the decomposition.

The leading explanation is distribution shift rather than a bug. The network
was trained with time advantage as a fixed binary sign bit; a search that
charges sitting, treats flags as terminal, and lets the advantage change hands
mid-tree is asking the value head about states it was never calibrated on. That
predicts exactly what is observed — degradation concentrated in ordinary board
play.

If that is right, this change is a **regression until the network is retrained
to match**, and the fix is on the evaluator side, not the search side.

### That conclusion is retracted, twice over *(2026-08-10)*

It rested on a board-play deficit that does not exist, and the remedy it proposed
was then tested directly and made things worse. Both halves failed independently:

**The thing it set out to explain was an artifact.** There was no board-play
degradation to attribute to distribution shift. Pinning the node multipliers to
1.0 inverts the board split (411−584 → 624−391 checkmates) and the headline
(−25 → +93 Elo). Clock-aware search is a *gain* of about 93 Elo at 800 nodes, and
the explanation above is an explanation of a measurement error.

**The proposed fix was tried and failed on its own terms.** `PHASE2_GATE.md` is
that retraining: two nets on one corpus differing only in whether channels 31/63
carry the binary sign bit or `tanh(margin / 50 ds)`. If distribution shift were
the story, the continuous arm should have recovered board play. Measured at equal
nodes it **loses on the board** — 39.6% of 2257 checkmates, −74 Elo there and −81
overall — and it also loses the flag battle 108−28, the one class of game the
finer margin most directly governs. So "retrain the network to match" is not a
pending fix; it is a closed negative result.

What survives is narrower and worth stating plainly: clock-aware **search** helps,
and a continuous clock **encoding** in the network does not. Those are independent
claims about different components, and the harness bug had inverted the sign of
both. Do not cite this section's distribution-shift argument as motivation for
further encoding work without reading the Phase 2 result first.

---

## What upstream's removal of `engine/src/rl/` costs this harness

Assessed 2026-08-04, against upstream `origin/main` at `f15b64d`. Nothing here
is merged into this branch yet — this exists so the decision is already costed
when the clock work is picked back up.

### `param-eval` is gone upstream, and not replaced

Upstream's `566c40c` deletes `engine/src/rl/` outright — all ten files, about
108 KB. That directory *is* this benchmark: `model_eval.cc` (31 KB) implements
the `param-eval` subcommand, its per-player settings, and the Elo summary
quoted above.

It was not moved or rewritten. `origin/main`'s `main.cc` now dispatches only
`bench` and `perft`; `selfplay`, `eval` and `param-eval` are all gone as
subcommands, with no replacement offering a two-player match. So merging
upstream into this branch without deciding this deletes the ability to rerun
the benchmark.

Merging `origin/main` into this branch conflicts in 16 files, four of them
modify/delete on `rl/gamepgn.*` and `rl/model_eval.*` — those four are this
decision, surfacing as conflicts.

### Only half of `rl/` is actually needed

The A/B uses:

| file | why |
|---|---|
| `model_eval.{cc,h}` | the `param-eval` harness itself |
| `gamepgn.{cc,h}` | `--pgn`, which "Reading the result" above requires |
| `rl_settings.h` | `EvalSettings::initialTimeDcs`, i.e. `--time-control` |

`selfplay.cc`, `training_data_writer.*` and `gui_state_writer.h` are
training-data machinery this benchmark never calls. Taking upstream's deletion
for those and keeping only the three above halves the fork-local surface.

### What the port has to fix

Most of what `model_eval.cc` calls survives upstream's refactor unchanged:
`board.san_move`, `make_moves`, `legal_moves`, `is_checkmate`, `is_draw`, and
`agent->run_search`. The clock-specific calls (`has_clocks`,
`set_clocks_visible`, `set_clocks`, `team_may_sit`, `team_flagged`) are this
branch's own additions and upstream never touched them.

Three things genuinely break:

1. **`Agent::get_root_node()` is gone.** `rootNode` is now a private member of
   `Agent` with no public accessor. `model_eval.cc` calls it once. Fix: re-add
   the accessor, one line.

2. **Dirichlet noise is gone entirely** — not just the `SearchOptions` fields
   but the whole path: `agent.cc`'s application of it, `node.h`'s
   `apply_dirichlet_noise`, and `utils.h`'s `generate_dirichlet_noise`.
   `origin/main` contains no reference to dirichlet anywhere in `engine/src`.
   This costs the A/B **nothing**: `model_eval.cc:260` already sets
   `dirichletEpsilon = 0.0f` because eval wants no noise. Delete the plumbing
   rather than restoring it. (Restoring *self-play* is a different matter and
   would need the whole path back.)

3. **`SearchOptions::selfplay()` and `::eval()` are gone**; only `uci()`
   remains. Replace those call sites with explicit field assignment.

### Recommendation

Keep-ours on `model_eval.{cc,h}`, `gamepgn.{cc,h}` and `rl_settings.h`; take
upstream's deletion for the other five files. The one-time fix is small and
bounded — an accessor, deleting dead noise plumbing, and two constructor call
sites. The ongoing cost is the real one: those five files become permanently
fork-maintained against an upstream that has removed them, so every future
upstream merge re-raises the same four modify/delete conflicts.

### The numbers above are stale regardless

Both runs were measured against a search that no longer exists. Upstream's
merge changes the search itself: real data-race fixes, ~9.8% more nps, and
`f15b64d` quadrupling `PW_COEFFICIENT` from 1.0 to 4.0, which widens interior
progressive widening 8 -> 32 allowed children at 1000 visits. The −26 Elo and
the on-the-clock/on-the-board split were both produced under the old search.

Rerun before acting on the "regression until the network is retrained"
conclusion. The distribution-shift explanation is about the value head and is
probably unaffected, but the decomposition is what this benchmark exists to
report, and it has not been measured against the current search.

**Resolved 2026-08-10, and the guess above was wrong.** It was rerun, and the
distribution-shift explanation was *not* unaffected — it was the casualty. The
"regression until the network is retrained" conclusion is withdrawn on both
counts: there is no regression (+93 Elo once node multipliers are pinned) and
retraining does not help (`PHASE2_GATE.md`, −81 Elo). The decomposition has now
been measured on the current search; see the banner at the top of this file and
the retraction under "Results".
