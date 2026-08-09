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

If you are reusing this harness, check all three before spending money. The
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

Both encodings and the time control are echoed before the first game. Paste
those three lines into whatever you write up; they are the difference between a
result and a number.

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

At 14.3 games/min on an RTX 3090, 2400 games is 2h47m.

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

Run 2026-08-08 on an RTX 3090, 2400 games, 190.5 min at 12.60 games/min. The
throughput estimate above (14.3 games/min, 2h47m) was optimistic by 14%; budget
3h10m. Networks were the `-dyn` re-exports — the arms as first trained had a
static batch dimension and TensorRT would not load them (`19ee114`).

```
  New model encoding: continuous
  Old model encoding: binary
  Time control: 400 ds
```

| | arm B (continuous) | arm A (binary) | draws |
|---|---:|---:|---:|
| games | 1623 | 770 | 7 |

Score 0.6777, **+129 Elo ± 7.1**. Colour-balanced: +817 −381 as White, +806 −389
as Black. Game length averaged 56.7 plies (19 min, 100 max).

### The split

| decided | arm B win rate | n | Elo |
|---|---:|---:|---:|
| on the board (checkmate) | 69.6% ± 1.0pp | 2306 | +144 |
| on the clock (flag) | 19.5% ± 5.4pp | 87 | −246 |

**The mechanism prediction holds.** `CLOCK_AB_BENCHMARK.md` concluded that
clock-aware search lost on the board because the network was trained with time
advantage as a fixed binary sign bit, and that it was "a regression until the
network is retrained to match". Retrained to match, arm B gains 144 Elo *on the
board*. It is not winning by flag-farming, which was the outcome that would have
made the aggregate meaningless. The board split is also 20× better powered than
the clock split and carries 96% of the games; it, not the +129, is the result.

**The clock split inverts, and it is not noise.** Arm B loses flag games 70–17,
19.5% ± 5.4pp, about 5.6 SE from even. The encoding built to make the sit margin
legible made the engine measurably worse at the one class of game the margin
directly governs. It costs little here — 3.6% of games, so splitting those 87
evenly instead would raise the aggregate only to +137 — but it is unexplained, and it is the open
question this match leaves behind rather than closes. The 70 games arm B lost on
time are in `gate.pgn`; the thing to separate is whether B sits longer per move
or simply plays longer games into the same flag.

**The value head is untouched.** Arm B's training gain was entirely in the policy
head (`val_loss` 0.9490 → 0.8584, `policy_acc` 0.688 → 0.720, `value_acc_sign`
0.6485 → 0.6488), and the match does not change what that implies: the severity
bands read the value head, so nothing here says the bands improve. Sitting is a
`pass` move and a finer margin helping the policy that chooses it is the expected
shape. Board play improved anyway, which is what makes the result worth having.

Artifacts: `.logs/phase2-gate-20260808/` (`gate.pgn`, `gate.log`, `gate.sh`,
`make_dynamic.py`, smoke run, build log). The repaired networks are
`weights/phase2-{binary,continuous}/*-v3.0-dyn.onnx`; the static-batch exports
beside them will not load and should not be used.
