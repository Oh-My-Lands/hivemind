# Clock-model A/B benchmark

How to measure whether making the search clock-aware makes the engine stronger.

Written 2026-08-03, from the first run of it. Reusable for any change that
alters *how the search reasons about time* while leaving the network alone.

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
converts clock awareness into flag wins. It also degrades ordinary board play,
and because flags are only ~12.5% of games the board deficit dominates: +20
games gained on the clock, −63 lost on the board.

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

**Node allocation moves with the clock.** The attacker/defender split
(0.5×/1.5× nodes, `ATTACKER_NODE_MULT`) keys off `teamHasTimeAdvantage`, which
is clock-derived for *both* players — so it is symmetric and does not favour
an arm, but it does mean node budget varies within a game.

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
