#!/usr/bin/env python3
"""Tests for the sit-margin clock encoding.

The PARITY_TABLE below is duplicated verbatim in
engine/tests/test_time_encoding.cc. If you change the squash function, change
both -- a network trained through one and evaluated through the other reads a
different input distribution, and nothing else in the system would notice.
"""

import os
import sys

import chess
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants import BOARD_A, BOARD_B
from src.domain.board import BughouseBoard
from src.domain.board2planes import board2planes
from src.domain.time_encoding import (PLANE_QUANT_OFFSET, PLANE_QUANT_SCALE,
                                      TIME_PLANE_CHANNEL_A,
                                      TIME_PLANE_CHANNEL_B, TimeEncoding,
                                      dequantize_margin_plane, encode_margin,
                                      flip_margins, quantize_margin_plane,
                                      quantize_planes, squash_margin)

BINARY = TimeEncoding.BINARY
CONTINUOUS = TimeEncoding.CONTINUOUS

# (margin in deciseconds, expected squashed value). Mirrored in the C++ test.
PARITY_TABLE = [
    (-800, -1.00000000),
    (-400, -0.99999977),
    (-100, -0.96402758),
    (-10, -0.19737532),
    (0, 0.00000000),
    (10, 0.19737532),
    (20, 0.37994896),
    (40, 0.66403677),
    (80, 0.92166855),
    (160, 0.99668240),
]


def test_squash_matches_parity_table():
    """Values must agree with the C++ mirror to float tolerance."""
    for margin_dcs, expected in PARITY_TABLE:
        assert squash_margin(margin_dcs) == pytest.approx(expected, abs=1e-6), (
            f"squash_margin({margin_dcs}) drifted"
        )


def test_squash_is_odd():
    assert squash_margin(0) == 0.0
    for m in (1, 10, 55, 300, 2000):
        assert squash_margin(-m) == pytest.approx(-squash_margin(m))


def test_squash_is_strictly_monotonic_where_it_has_to_be():
    """Strict only out to +/-30s -- see the C++ mirror for why.

    tanh saturates, and it saturates at a slightly different margin in the
    engine's float32 than in this float64. The two agree to ~1e-8 there and
    quantise to the same byte from 15s upward, so the difference is invisible
    to the network; what must hold is that the encoding stays ordered across
    the range it was scaled for, whose top tier is 16s.
    """
    values = [squash_margin(m) for m in range(-300, 301, 25)]
    assert all(a < b for a, b in zip(values, values[1:]))


def test_squash_never_decreases():
    values = [squash_margin(m) for m in range(-3000, 3001, 25)]
    assert all(a <= b for a, b in zip(values, values[1:]))


def test_squash_saturates_within_range():
    """Everything stays inside [-1, 1] so the quantisation cannot overflow."""
    for m in (-100000, -3000, 0, 3000, 100000):
        assert -1.0 <= squash_margin(m) <= 1.0


def test_quantize_round_trip():
    for margin_dcs, _ in PARITY_TABLE:
        squashed = squash_margin(margin_dcs)
        stored = quantize_margin_plane(squashed)
        assert 0 <= stored <= 255, f"{stored} outside uint8 range"
        assert dequantize_margin_plane(stored) == pytest.approx(squashed, abs=1.0 / PLANE_QUANT_SCALE)


def test_quantize_preserves_sign():
    """The whole point of the affine map: a negative margin must not wrap."""
    for margin_dcs in (-1, -10, -50, -400):
        stored = quantize_margin_plane(squash_margin(margin_dcs))
        assert stored < PLANE_QUANT_OFFSET
        assert dequantize_margin_plane(stored) < 0
    for margin_dcs in (1, 10, 50, 400):
        stored = quantize_margin_plane(squash_margin(margin_dcs))
        assert stored > PLANE_QUANT_OFFSET
        assert dequantize_margin_plane(stored) > 0


def test_flip_margins_is_negate_and_swap():
    assert flip_margins(0.3, -0.7) == (0.7, -0.3)
    # Applying it twice returns the original pair.
    a, b = 0.42, -0.17
    assert flip_margins(*flip_margins(a, b)) == (a, b)


# ---------------------------------------------------------------------------
# The two encodings
# ---------------------------------------------------------------------------

def test_binary_encoding_is_the_sign_of_the_continuous_one():
    """The property the whole Phase 2 A/B rests on.

    generate_planes gates pass-samples on `plane > 0.0`, so if the two encodings
    ever disagreed about sign the arms would be trained on differently composed
    datasets and the comparison would confound encoding with corpus -- the
    "baseline trap" in TIME_MODEL_PLAN.md, reintroduced by the back door.
    """
    for margin_dcs in (-4000, -400, -50, -10, -1, 0, 1, 10, 50, 400, 4000):
        binary = encode_margin(margin_dcs, BINARY)
        continuous = encode_margin(margin_dcs, CONTINUOUS)
        assert (binary > 0.0) == (continuous > 0.0), (
            f"encodings disagree on sign at {margin_dcs} dcs"
        )


def test_binary_encoding_is_zero_or_one():
    for margin_dcs in (-400, -1, 0, 1, 400):
        assert encode_margin(margin_dcs, BINARY) in (0.0, 1.0)


def test_level_clocks_do_not_permit_sitting_under_either_encoding():
    """Zero margin is not an advantage; > 0.0 must exclude it in both arms."""
    assert encode_margin(0, BINARY) == 0.0
    assert encode_margin(0, CONTINUOUS) == 0.0


def test_binary_survives_the_affine_map_exactly():
    """Arm A stores 0/1 through the margin path and must read back unchanged.

    One storage format serves both arms, so the loader does not have to know
    which wrote the shard. That is only safe if binary round-trips exactly.
    """
    for value in (0.0, 1.0):
        stored = quantize_margin_plane(value)
        assert 0 <= stored <= 255
        assert dequantize_margin_plane(stored) == value


def test_default_encoding_is_binary():
    """The deployed network expects the sign bit and would read garbage
    from a continuous plane, so binary has to be what you get by default."""
    board = _board_with_clocks(600, 300, 250, 480)
    default = board2planes(board, chess.WHITE)
    explicit = board2planes(board, chess.WHITE, time_encoding=BINARY)
    assert np.array_equal(default, explicit)


def test_binary_planes_match_the_pre_harvest_formula():
    """Pins that this refactor did not change Phase 0's semantics.

    board2planes used to write `time_advantage(team_side) > 0` into 31 and
    `time_advantage(not team_side) < 0` into 63. Routing both through
    sit_margin has to be exactly equivalent, or arm A is not the baseline it
    claims to be.
    """
    offset = 32
    for clocks in [(600, 300, 250, 480), (100, 100, 600, 600),
                   (300, 300, 300, 300), (450, 610, 620, 440)]:
        board = _board_with_clocks(*clocks)
        for team in (chess.WHITE, chess.BLACK):
            planes = board2planes(board, team, time_encoding=BINARY)
            old_a = 1.0 if board.time_advantage(team) > 0 else 0.0
            old_b = 1.0 if board.time_advantage(not team) < 0 else 0.0
            assert planes[TIME_PLANE_CHANNEL_A][0, 0] == old_a, clocks
            assert planes[offset + 31][0, 0] == old_b, clocks


# ---------------------------------------------------------------------------
# Margin algebra on a real board
# ---------------------------------------------------------------------------

def _board_with_clocks(a_white, a_black, b_white, b_black):
    """Build a board with named clocks, in deciseconds.

    `BughouseBoard.times` is indexed `[board][colour]` using python-chess
    colours, where `chess.WHITE is True` -- so index 1 is white, not 0. Assign
    by colour rather than by literal position; Stockfish orders colours the
    other way round and a positional literal here would silently transpose.
    """
    board = BughouseBoard(1800)
    times = [[0, 0], [0, 0]]
    times[BOARD_A][chess.WHITE] = a_white
    times[BOARD_A][chess.BLACK] = a_black
    times[BOARD_B][chess.WHITE] = b_white
    times[BOARD_B][chess.BLACK] = b_black
    board.set_times(times)
    return board


def test_sit_margin_is_diagonal_not_same_board():
    """Board A's margin is against the board-B player of the same colour."""
    # A-White 60.0s, A-Black 10.0s, B-White 20.0s, B-Black 50.0s
    board = _board_with_clocks(600, 100, 200, 500)

    # White team = A-White + B-Black.
    # A-White (600) races their diagonal, B-White (200)  -> +400
    assert board.sit_margin(chess.WHITE, BOARD_A) == 400
    # B-Black (500) races their diagonal, A-Black (100)  -> +400
    assert board.sit_margin(chess.WHITE, BOARD_B) == 400

    # Black team = A-Black + B-White, the negated-and-swapped pair.
    assert board.sit_margin(chess.BLACK, BOARD_A) == -400
    assert board.sit_margin(chess.BLACK, BOARD_B) == -400


def test_the_two_boards_can_disagree():
    """The case the old encoding could not represent at all."""
    # A-White is far ahead of B-White; B-Black is far behind A-Black.
    board = _board_with_clocks(600, 600, 100, 100)

    assert board.sit_margin(chess.WHITE, BOARD_A) == 500   # 600 - 100
    assert board.sit_margin(chess.WHITE, BOARD_B) == -500  # 100 - 600

    # A single team-wide bit would have to pick one of these and be wrong about
    # the other; the team is level overall but the boards are opposites.
    assert board.team_uptime(chess.WHITE) == 0


def test_team_uptime_matches_viewer_formula():
    """Must equal getTeamTimeDiffDeciseconds in app/utils/board/clockAdvantage.ts."""
    a_white, a_black, b_white, b_black = 631, 512, 407, 288
    board = _board_with_clocks(a_white, a_black, b_white, b_black)

    expected = (a_white + b_black) - (a_black + b_white)
    assert board.team_uptime(chess.WHITE) == expected
    assert board.team_uptime(chess.BLACK) == -expected


def test_margins_of_the_two_teams_are_negate_and_swap():
    """The invariant the search relies on to avoid carrying clocks down the tree."""
    board = _board_with_clocks(613, 244, 501, 158)

    white = (board.sit_margin(chess.WHITE, BOARD_A), board.sit_margin(chess.WHITE, BOARD_B))
    black = (board.sit_margin(chess.BLACK, BOARD_A), board.sit_margin(chess.BLACK, BOARD_B))

    assert flip_margins(*white) == black


# ---------------------------------------------------------------------------
# Plane construction
# ---------------------------------------------------------------------------

def test_planes_carry_per_board_margins():
    board = _board_with_clocks(600, 600, 100, 100)
    planes = board2planes(board, chess.WHITE, time_encoding=CONTINUOUS)

    assert planes[TIME_PLANE_CHANNEL_A][0, 0] == pytest.approx(squash_margin(500))
    assert planes[TIME_PLANE_CHANNEL_B][0, 0] == pytest.approx(squash_margin(-500))
    # The regression this change exists to prevent: the two planes were equal.
    assert planes[TIME_PLANE_CHANNEL_A][0, 0] != planes[TIME_PLANE_CHANNEL_B][0, 0]


def test_time_planes_are_constant_across_squares():
    board = _board_with_clocks(400, 300, 200, 100)
    for encoding in (BINARY, CONTINUOUS):
        planes = board2planes(board, chess.WHITE, time_encoding=encoding)
        for channel in (TIME_PLANE_CHANNEL_A, TIME_PLANE_CHANNEL_B):
            assert (planes[channel] == planes[channel][0, 0]).all()


def test_flip_augmentation_swaps_margins_without_negating():
    """`flip` relabels which board is A, so the margins swap and keep their sign.

    This is what makes the block-swap augmentation in
    src/training/data_loaders.flip_bughouse_sample correct for these planes. It
    was trivially true when both planes held the same value; now it is not.
    """
    board = _board_with_clocks(600, 600, 100, 100)

    planes = board2planes(board, chess.WHITE, flip=False, time_encoding=CONTINUOUS)
    flipped = board2planes(board, chess.WHITE, flip=True, time_encoding=CONTINUOUS)

    assert flipped[TIME_PLANE_CHANNEL_A][0, 0] == pytest.approx(planes[TIME_PLANE_CHANNEL_B][0, 0])
    assert flipped[TIME_PLANE_CHANNEL_B][0, 0] == pytest.approx(planes[TIME_PLANE_CHANNEL_A][0, 0])


def test_quantize_planes_round_trips_a_real_sample():
    board = _board_with_clocks(600, 300, 250, 480)
    for encoding in (BINARY, CONTINUOUS):
        planes = board2planes(board, chess.WHITE, time_encoding=encoding)

        stored = quantize_planes(planes)
        assert stored.dtype.name == "uint8"

        for channel in (TIME_PLANE_CHANNEL_A, TIME_PLANE_CHANNEL_B):
            recovered = dequantize_margin_plane(float(stored[channel][0, 0]))
            assert recovered == pytest.approx(planes[channel][0, 0], abs=1.0 / PLANE_QUANT_SCALE)


def test_quantize_planes_keeps_pocket_counts():
    """Guards the truncation that a bare astype(uint8) would introduce."""
    board = _board_with_clocks(600, 600, 600, 600)
    board.boards[BOARD_A].pockets[chess.WHITE].add(chess.KNIGHT)
    board.boards[BOARD_A].pockets[chess.WHITE].add(chess.KNIGHT)
    board.boards[BOARD_A].pockets[chess.WHITE].add(chess.KNIGHT)

    planes = board2planes(board, chess.WHITE)
    stored = quantize_planes(planes)

    # Knight is the second of the five droppable types, own pocket starts at 12.
    assert stored[13][0, 0] == 3


def test_quantize_planes_leaves_binary_planes_binary():
    board = _board_with_clocks(600, 300, 250, 480)
    planes = board2planes(board, chess.WHITE)
    stored = quantize_planes(planes)

    # Constant plane 26 is all ones; the piece planes are 0/1.
    assert (stored[26] == 1).all()
    assert set(stored[0].flatten().tolist()) <= {0, 1}


def test_write_read_round_trip_through_the_loader_boundary():
    """quantize_planes and dequantize_planes must invert each other on a whole
    sample, not only on the margin channels -- they are the only two places the
    storage format is known, and a shard written by one and read by the other
    is what training actually consumes."""
    import torch

    from src.domain.time_encoding import dequantize_planes

    board = _board_with_clocks(600, 300, 250, 480)
    board.boards[BOARD_A].pockets[chess.WHITE].add(chess.ROOK)
    board.boards[BOARD_B].pockets[chess.BLACK].add(chess.PAWN)

    for encoding in (BINARY, CONTINUOUS):
        planes = board2planes(board, chess.WHITE, time_encoding=encoding)
        stored = quantize_planes(planes)

        loaded = torch.tensor(
            np.frombuffer(stored.tobytes(), dtype=np.uint8).astype(np.float32)
        ).view(-1, 64, 8, 8)
        dequantize_planes(loaded)

        assert loaded[0].numpy() == pytest.approx(planes, abs=1.0 / PLANE_QUANT_SCALE)


def test_rl_shards_keep_their_raw_binary_margin():
    """The RL and supervised on-disk formats differ in channels 31/63.

    engine/src/rl/training_data_writer.cc thresholds every non-pocket channel to
    a raw 0/1, self-play's time-advantage flag being a per-game parity constant
    rather than a clock. Applying the affine inverse to that would read a stored
    1 as -1.27 and feed the value head a plane far outside its training range,
    which nothing downstream would flag.
    """
    import torch

    from src.domain.time_encoding import dequantize_planes

    rl_style = torch.zeros(1, 64, 8, 8)
    rl_style[:, TIME_PLANE_CHANNEL_A, :, :] = 1.0   # as the C++ writer stores it
    rl_style[:, TIME_PLANE_CHANNEL_B, :, :] = 0.0
    rl_style[:, 13, :, :] = 3.0                     # a raw pocket count

    dequantize_planes(rl_style, margins_are_affine=False)

    assert rl_style[0, TIME_PLANE_CHANNEL_A, 0, 0].item() == 1.0
    assert rl_style[0, TIME_PLANE_CHANNEL_B, 0, 0].item() == 0.0
    assert rl_style[0, 13, 0, 0].item() == pytest.approx(3.0 / 16.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
