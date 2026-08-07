"""Encoding of bughouse clock state into network input planes.

There is a C++ mirror of this file at ``engine/src/time_encoding.h``. The two
must agree exactly or the network is trained on one distribution and evaluated
on another; ``tests/test_time_encoding.py`` and
``engine/tests/test_time_encoding.cc`` check the same table of values on both
sides to catch drift.

Background
----------
A team's ability to sit is not a property of the team, it is a property of each
*player* relative to their **diagonal** opponent. Teams are diagonal pairings
(board-A White partners board-B Black), so the player who decides whether to
sit on board A is racing the board-B player of the same colour: if that
opponent sits, board A never receives the piece it is waiting for, and the two
clocks burn against each other.

That makes the decision variable a signed *margin* in deciseconds, one per
board, not a single team-wide bit.

The two encodings
-----------------
``BINARY`` writes the sign of that margin, which is Phase 0's semantics and
what the deployed network expects. ``CONTINUOUS`` writes the squashed magnitude.
Both are produced by this one module so the Phase 2 A/B differs in the encoding
and nothing else -- see ``TIME_MODEL_PLAN.md``, "The baseline trap".
"""

import math
from enum import Enum

import numpy as np

from src.constants import MAX_NUM_DROPS

# Deciseconds of margin that map to tanh(1) ~= 0.76. Chosen so the range the
# frontend already treats as meaningful (`getAdvantageTier` in
# app/utils/board/clockAdvantage.ts buckets at 1s/2s/4s/8s/16s) spreads across
# most of the output range rather than saturating immediately:
#   1s -> 0.20   2s -> 0.38   4s -> 0.66   8s -> 0.92   16s -> 1.00
TIME_SQUASH_SCALE_DCS = 50.0

# Plane values are persisted as uint8 (see ShardWriter.add_sample and
# engine/src/rl/training_data_writer.cc), so a signed value in [-1, 1] needs an
# affine map into that range before it is written and the inverse on load.
# 100/128 keeps two decimal places and leaves headroom at both ends.
PLANE_QUANT_SCALE = 100.0
PLANE_QUANT_OFFSET = 128

# Channels holding the sit margin, one per board.
TIME_PLANE_CHANNEL_A = 31
TIME_PLANE_CHANNEL_B = 63

# Channel ranges that hold something other than a 0/1 indicator, and so need an
# explicit quantisation to survive uint8 storage. Mirrors
# `GameSampleBuffer::add_position` in engine/src/rl/training_data_writer.cc.
POCKET_CHANNELS_A = slice(12, 22)
POCKET_CHANNELS_B = slice(44, 54)


class TimeEncoding(str, Enum):
    """Which representation of the sit margin goes into channels 31 and 63.

    Mirrors the ``TimeEncoding`` UCI combo option on the engine side. This is
    the *only* thing that differs between the two Phase 2 training arms, so it
    is threaded through plane generation rather than being a build-time
    constant.
    """

    BINARY = "binary"
    CONTINUOUS = "continuous"


def squash_margin(margin_dcs: float) -> float:
    """Map a signed clock margin in deciseconds onto [-1, 1].

    Positive means the player on that board is ahead of their diagonal
    opponent and can afford to sit; 0.0 means the clocks are level.
    """
    return math.tanh(margin_dcs / TIME_SQUASH_SCALE_DCS)


def encode_margin(margin_dcs: float, encoding: TimeEncoding) -> float:
    """Encode one board's sit margin for storage in its plane.

    The two encodings agree on sign -- ``BINARY`` is 1.0 exactly when
    ``CONTINUOUS`` is positive -- which is what lets `generate_planes` apply the
    same ``> 0.0`` sit gate to both arms and so hold training-set composition
    fixed across the A/B. A test pins that equivalence.
    """
    if encoding == TimeEncoding.BINARY:
        return 1.0 if margin_dcs > 0 else 0.0
    return squash_margin(margin_dcs)


def quantize_margin_plane(value: float) -> int:
    """Affine-map a squashed margin in [-1, 1] into the uint8 storage range."""
    return int(round(value * PLANE_QUANT_SCALE)) + PLANE_QUANT_OFFSET


def dequantize_margin_plane(value: float) -> float:
    """Inverse of :func:`quantize_margin_plane`."""
    return (value - PLANE_QUANT_OFFSET) / PLANE_QUANT_SCALE


def flip_margins(margin_a: float, margin_b: float) -> tuple[float, float]:
    """Re-express a pair of margins from the opposing team's perspective.

    With ``ts`` the team's colour, its margins are

        mA  =  A[ts]  - B[ts]          (board-A member vs their diagonal)
        mB  =  B[~ts] - A[~ts]         (board-B member vs their diagonal)

    The opposing team's are the same two expressions with ``ts`` negated, which
    reduces to negating and swapping:

        mA' =  A[~ts] - B[~ts] = -mB
        mB' =  B[ts]  - A[ts]  = -mA

    The search relies on this: it holds the root team's margins and derives the
    opponent's per node rather than carrying four clocks down the tree.

    Note this is *not* the board-swap flip augmentation, which relabels which
    board is which for the same team and so is a plain swap with no negation.
    """
    return (-margin_b, -margin_a)


def quantize_planes(planes: np.ndarray) -> np.ndarray:
    """Convert float input planes into the uint8 form written to parquet.

    This is the single write boundary. Binary planes round to 0/1, pocket planes
    are rescaled from ``count / MAX_NUM_DROPS`` back to their raw 0-16 count, and
    the two margin planes are affine-mapped so their sign survives. A plain
    ``astype(np.uint8)`` truncates pocket counts to zero and wraps negative
    margins, which is why this has to be explicit.

    The margin channels take the affine map under *both* encodings. A binary
    1.0 stores as 228 and reads back as exactly 1.0, so the tensor the network
    sees is unchanged; keeping one storage format means the loader does not
    have to know which arm wrote the shard.
    """
    out = np.zeros(planes.shape, dtype=np.uint8)

    binary = np.ones(planes.shape[0], dtype=bool)
    binary[POCKET_CHANNELS_A] = False
    binary[POCKET_CHANNELS_B] = False
    binary[TIME_PLANE_CHANNEL_A] = False
    binary[TIME_PLANE_CHANNEL_B] = False

    out[binary] = (planes[binary] > 0.5).astype(np.uint8)

    # board2planes emits pockets as count / MAX_NUM_DROPS to mirror
    # engine/src/planes.cc:90; storing that directly truncates 1/16 to 0 and
    # zeroes twenty channels. Rescale here rather than in board2planes so that
    # function stays an exact mirror of the C++.
    out[POCKET_CHANNELS_A] = np.rint(planes[POCKET_CHANNELS_A] * MAX_NUM_DROPS)
    out[POCKET_CHANNELS_B] = np.rint(planes[POCKET_CHANNELS_B] * MAX_NUM_DROPS)

    for channel in (TIME_PLANE_CHANNEL_A, TIME_PLANE_CHANNEL_B):
        out[channel] = np.rint(planes[channel] * PLANE_QUANT_SCALE) + PLANE_QUANT_OFFSET

    return out


def dequantize_planes(x_tensor, margins_are_affine=True):
    """Invert :func:`quantize_planes` on a loaded ``(N, 64, 8, 8)`` tensor.

    In place, and returned for convenience. This is the single read boundary --
    both `load_parquet_shard` and `load_rl_parquet_shard` go through it, so the
    two cannot drift apart and neither can drift from the writer.

    ``margins_are_affine`` selects between the two on-disk formats, which differ
    only in channels 31 and 63:

    - **Supervised shards** (`ShardWriter`, this module's `quantize_planes`)
      store the margin through the affine map, so a negative one survives.
    - **RL shards** (`GameSampleBuffer::add_position` in
      engine/src/rl/training_data_writer.cc) store a raw 0/1 there, because
      self-play's time-advantage flag is a per-game constant chosen by parity
      rather than a clock -- there is no magnitude to preserve. Applying the
      affine inverse to those would read a stored 1 as -1.27.

    The two converge if and only if self-play is ever given real clocks; until
    then the flag records a genuine difference rather than papering over one.
    """
    x_tensor[:, POCKET_CHANNELS_A, :, :] /= MAX_NUM_DROPS
    x_tensor[:, POCKET_CHANNELS_B, :, :] /= MAX_NUM_DROPS

    if margins_are_affine:
        for channel in (TIME_PLANE_CHANNEL_A, TIME_PLANE_CHANNEL_B):
            x_tensor[:, channel, :, :] = (
                x_tensor[:, channel, :, :] - PLANE_QUANT_OFFSET
            ) / PLANE_QUANT_SCALE

    return x_tensor
