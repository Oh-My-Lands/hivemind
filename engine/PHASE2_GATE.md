# The Phase 2 gate — binary vs continuous sit margin

How to run the match that decides whether the continuous clock encoding is worth
keeping, and what has to be true before the number it prints means anything.

Companion to `CLOCK_AB_BENCHMARK.md`, which measures a *search* change with one
network. This measures an *encoding* change with two networks trained for it.

---

## What it measures

Two networks, trained on the same corpus, differing only in what channels 31 and
63 carry:

| arm | channels 31 and 63 |
|---|---|
| **A — binary** | `margin > 0 ? 1.0 : 0.0`, the Phase 0 sign bit |
| **B — continuous** | `tanh(margin / 50 ds)`, signed, in [-1, 1] |

Same 14.6M-sample corpus, same 7 epochs, same schedule. `TimeEncoding::Mode`
selects which one the engine writes at inference, per player, so both play in
one match.

## The prerequisite that was missing for a month

Until `ca29e8c` the engine could not present arm B's encoding at all.
`board_to_planes` took a `bool` and wrote `1.0f : 0.0f`; `TimeEncoding::Mode`
existed in the header, was documented as "exposed as a UCI combo option", and
was referenced by nothing but its own unit tests. Running the gate before that
commit would have finished, printed an Elo, and measured arm B's response to an
input distribution it had never seen.

Three things had to be true, and none were:

1. **The engine encodes the mode the network expects.** `--new-encoding` /
   `--old-encoding`, defaulting to binary, rejecting anything else outright.
2. **The margin is per board.** Channels 31 and 63 are different quantities —
   each team member races the player of their own colour on the *other* board.
   They differ on 10% of corpus positions under binary and 88% under continuous.
   One team-wide bit cannot express that.
3. **There are clocks.** `eval` parsed no `--time-control`, so `initialTimeDcs`
   stayed 0 and the margin never varied. The encoding under test was constant.

A fourth was missing for longer than that, and it invalidated the first run of
this gate:

4. **Both sides search the same number of nodes.** Until 2026-08-10
   `ModelEvaluator::playGame` scaled the configured budget by
   `ATTACKER_NODE_MULT = 0.5` for the team up on time and
   `DEFENDER_NODE_MULT = 1.5` for the team down — a 3x swing keyed on
   `team_may_sit(currentTeam)`, read live once clocks exist. `--nodes 800` never
   meant 800. In *this* gate that knob is pointed straight at the treatment: the
   thing under test is how a net handles the clock, so whichever arm was better
   or worse at acquiring a time advantage had its search silently rescaled for
   it. Both multipliers now default to 1.0 and are echoed in the header;
   `--new-attacker-mult` / `--new-defender-mult` (and `--old-`) reproduce the old
   behaviour if you need to.

If you are reusing this harness, check all four before spending money. The
failure mode is not an error; it is a plausible number.

## Running it

Needs TensorRT 10 (see `CLOCK_AB_BENCHMARK.md` for why v9 and v11 both fail) and
a network whose ONNX has a **dynamic batch dimension** — see below.

```bash
./build/hivemind eval \
    --new networks/<arm-B>.onnx --old networks/<arm-A>.onnx \
    --new-encoding continuous --old-encoding binary \
    --games 2400 --nodes 800 --time-control 400 \
    --pgn gate.pgn --verbose
```

Both encodings, the time control, and the node multipliers are echoed before the
first game. Paste those four lines into whatever you write up; they are the
difference between a result and a number. The multipliers are the newest of the
four and the reason the 2026-08-08 run had to be thrown away — a log that does
not state them cannot be interpreted.

Fixed **nodes**, not time — otherwise the match measures throughput. `--nodes
800` and `--time-control 400` match the earlier clock A/B, so the two are
comparable; 400 ds is where the clock binds without every game flagging at the
same ply (that calibration is in the companion doc and is not repeated here).

### N

`SE(Elo) ≈ 347 / sqrt(N)`, so 1200 gives ±10 and 2400 gives ±7.1.

Run 2400 rather than 1200. Not for the aggregate — for the split. The result
worth reading is the decomposition into games decided on the clock and games
decided on the board, and at 1200 the clock split was only ~48 games (±7.2% in
win rate). The earlier benchmark also needed two 1200-game runs pooled before
its aggregate was significant, and doubling up front costs about $0.40.

Measured, not estimated: 12.60 games/min on an RTX 3090 (190.5 min) and 17.84 on
the 2026-08-10 pod (134.5 min). The original 14.3 games/min projection was
optimistic by 14% against the 3090; budget 3h10m there and ~2h15m on faster
silicon.

### The ONNX must be batchable

A model exported before `19ee114` has a batch dimension of 1 frozen into it by
onnxsim, and TensorRT refuses it:

```
Error Code 4: Input tensor obs has static dimensions that don't match kMIN
dimensions in profile index 0. Input dimensions are [1,64,8,8] but profile
dimensions are [16,64,8,8].
```

Check before renting anything:

```python
import onnx
d = onnx.load(path).graph.input[0].type.tensor_type.shape.dim[0]
assert d.WhichOneof("value") == "dim_param", "static batch; the engine cannot batch it"
```

## Reading the result

Do not read the Elo alone — see `CLOCK_AB_BENCHMARK.md`, "Reading the result".
A clock encoding can lose overall while doing exactly what it was built to do,
and the aggregate is a weakly-powered difference of two opposing effects.

```bash
grep -o '^\[Termination "[^"]*"\]' gate.pgn \
  | sed 's/^\[Termination "//;s/"\]$//' | sort | uniq -c | sort -rn
```

Split into "on the clock" and "on the board" and report both, with
`SE = sqrt(0.25/n)` on each. The earlier benchmark's two runs disagreed by 10
Elo on the aggregate while agreeing closely on the split.

### What this match is a test of

`CLOCK_AB_BENCHMARK.md` ends with a prediction: clock-aware search lost on the
board because "the network was trained with time advantage as a fixed binary
sign bit", and a search that charges sitting and treats flags as terminal was
asking the value head about states it was never calibrated on. It concluded the
clock-aware search is "a regression until the network is retrained to match".

Phase 2 is that retraining. If the explanation was right, arm B should recover
board play, not just flag wins — so the decomposition is the test of the
mechanism, and the aggregate is only the accounting.

**It was not right.** Once the node multipliers were pinned, arm B loses on the
board as well as on the clock. See "Results".

### What the validation loss already says, and does not

| | val_loss | policy_acc | value_acc_sign |
|---|---:|---:|---:|
| A (binary) | 0.9490 | 0.6876 | 0.6485 |
| B (continuous) | 0.8584 | 0.7201 | 0.6488 |

Arm B's entire gain is in the policy head; the value head is flat to three
decimals. Sitting is a `pass` *move*, so a finer margin helping policy is the
expected shape — but the severity bands read the value head, and on this
evidence the encoding has not moved it. Board play is decided by both. Do not
predict the match from these numbers.

## Results

**Arm B loses to arm A by 81 Elo. The continuous encoding is not worth keeping on
this evidence.**

The gate has been run twice. The 2026-08-08 run reported +129 Elo for arm B and
was wrong — it carried the 0.5x/1.5x node handicap described in prerequisite 4.
The 2026-08-10 rerun changed nothing but those multipliers and reversed the
result. Both are recorded below, because the size of the gap between them is the
most useful thing this document now contains.

| run | multipliers | arm B W–D–L | Elo | arm B mates | arm A mates | arm B flags | arm A flags |
|---|---|---:|---:|---:|---:|---:|---:|
| 2026-08-08 | 0.5 / 1.5 | 1623–7–770 | +129 | 1611 | 701 | 17 | 140 |
| **2026-08-10** | **1.0 / 1.0** | **921–7–1472** | **−81** | **893** | **1364** | **28** | **108** |

Rerun: 2400 games, 134.5 min at 17.84 games/min, `--nodes 800 --time-control 400`,
same `-dyn` networks. Header as echoed:

```
  New model encoding: continuous
  Old model encoding: binary
  Time control: 400 ds
  Node multipliers: new att 1 / def 1, old att 1 / def 1
```

Score 0.3852. Colour-balanced: +452 −744 as White, +469 −728 as Black. Game
length averaged 59.4 plies (20 min, 97 max).

### The split

| decided | arm B win rate | n | Elo |
|---|---:|---:|---:|
| on the board (checkmate) | 39.6% ± 1.0pp | 2257 | −74 |
| on the clock (flag) | 20.6% ± 4.3pp | 136 | −235 |

**The board result inverted; the clock result did not.** Arm B went from 69.6% of
mates to 39.6% — a 210 Elo swing on the aggregate — while the flag split moved
only from 19.5% to 20.6%, well inside its own error bar. That asymmetry is the
whole story. Arm B loses the clock race, so under the old multipliers it was the
1.5x defender in nearly every game while arm A was throttled to 0.5x; the board
"win" was reading that node surplus, and the flag losses were the one measurement
the handicap could not flatter, because they are what caused it.

**So the mechanism prediction failed.** `CLOCK_AB_BENCHMARK.md` predicted that
retraining the network to match a clock-aware search would recover board play.
Retrained to match, arm B is *worse* on the board at equal nodes. Whatever the
continuous margin does for the policy head in training does not survive contact
with search at 800 nodes.

**The clock finding is the survivor, and it is now the only finding.** Arm B loses
flag games 108–28, 20.6% ± 4.3pp, ~6.9 SE from even and better powered than the
2026-08-08 version of the same claim. The encoding built to make the sit margin
legible makes the engine worse at the one class of game the margin directly
governs — and it no longer has a board-play gain sitting next to it as
compensation. The thing to separate is still whether B sits longer per move or
plays longer games into the same flag; the games are in
`.logs/multfix-20260810/gate.pgn`.

**The value head was untouched throughout.** Arm B's training gain was entirely
in the policy head (`val_loss` 0.9490 → 0.8584, `policy_acc` 0.688 → 0.720,
`value_acc_sign` 0.6485 → 0.6488). The severity bands read the value head, so
nothing in either run says the bands improve. What changed is that the 2026-08-08
write-up could point to board play as the reason to keep the arm anyway. That
reason is gone.

### Scope of the −81

800 nodes only. This net family's ranking has reversed between 800 and 3200 nodes
before, and a rerun at 3200 has not been done — at that budget the clock decides
most games before the board does, which is the separate harness problem
`--equal-time` exists for. Do not quote −81 as an unqualified strength
difference; quote it as an 800-node result, the same way +129 should have been.

Artifacts:
- **`.logs/multfix-20260810/`** — the run that stands (`gate.sh`, `gate.log`,
  `gate.pgn`, `out-driver.log`, `build.log`). Checksum-verified off the pod
  before it was stopped.
- `.logs/phase2-gate-20260808/` — the retracted run (`gate.pgn`, `gate.log`,
  `gate.sh`, `make_dynamic.py`, smoke run, build log). Kept for the comparison;
  its `gate.log` states +129 with no record of the multipliers, which is what
  made the error survive for two days.

The repaired networks are `weights/phase2-{binary,continuous}/*-v3.0-dyn.onnx`;
the static-batch exports beside them will not load and should not be used.
