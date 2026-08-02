"""Mirror of engine/src/time_control.h, asserted on the same table of values.

The C++ header prices sitting inside the search; board.py's `time_advantage`
feeds the training planes. If the two drift, the engine is trained on one notion
of clock advantage and searches with another, and nothing crashes to say so.

`test_shared_table` here and `TEST(Mirror, SharedTableOfValues)` in
engine/tests/logic/test_time_control.cc assert the identical table.
"""

import chess
import pytest

from src.domain.board import BughouseBoard

# Mirrors TimeControl::MOVE_COST_DCS / SIT_COST_DCS.
MOVE_COST_DCS = 9
SIT_COST_DCS = 9


def member_is_white(team_is_white: bool, board: int) -> bool:
    """Colour a team's member plays on `board`. Teams are diagonal pairings."""
    return team_is_white if board == 0 else not team_is_white


def sit_margin(times, team_is_white: bool, board: int) -> int:
    """Signed margin against the DIAGONAL opponent, in deciseconds.

    `times` is indexed [board][colour] exactly as BughouseBoard.times is, where
    python-chess has WHITE == True == 1.
    """
    mine = member_is_white(team_is_white, board)
    return times[board][mine] - times[1 - board][mine]


def team_margin(times, team_is_white: bool) -> int:
    return sit_margin(times, team_is_white, 0) + sit_margin(times, team_is_white, 1)


def team_flagged(times, team_is_white: bool) -> bool:
    """Either member out of time loses it for the team -- an OR, not a compare."""
    return (times[0][member_is_white(team_is_white, 0)] <= 0
            or times[1][member_is_white(team_is_white, 1)] <= 0)


def margin_bucket(margin_dcs: int) -> int:
    mag = abs(margin_dcs)
    if mag < 10:
        tier = 0
    elif mag < 20:
        tier = 1
    elif mag < 40:
        tier = 2
    elif mag < 80:
        tier = 3
    elif mag < 160:
        tier = 4
    else:
        tier = 5
    return -tier if margin_dcs < 0 else tier


def make_times(a_white, a_black, b_white, b_black):
    """times[board][colour], colour indexed by python-chess WHITE == True."""
    return [
        {chess.WHITE: a_white, chess.BLACK: a_black},
        {chess.WHITE: b_white, chess.BLACK: b_black},
    ]


# (a_white, a_black, b_white, b_black, team_is_white, m0, m1, team, bucket0)
SHARED_TABLE = [
    (1800, 1800, 1800, 1800, True,     0,     0,     0, 0),
    (1800, 1000, 1000, 1800, True,   800,   800,  1600, 5),
    (1000,  800,  500,  200, True,   500,  -600,  -100, 5),
    (1000,  800,  500,  200, False,  600,  -500,   100, 5),
    (1805, 1800, 1800, 1800, True,     5,     0,     5, 0),
    (1810, 1800, 1800, 1800, True,    10,     0,    10, 1),
]


@pytest.mark.parametrize("aw,ab,bw,bb,team,m0,m1,team_m,bucket0", SHARED_TABLE)
def test_shared_table(aw, ab, bw, bb, team, m0, m1, team_m, bucket0):
    times = make_times(aw, ab, bw, bb)
    assert sit_margin(times, team, 0) == m0
    assert sit_margin(times, team, 1) == m1
    assert team_margin(times, team) == team_m
    assert margin_bucket(sit_margin(times, team, 0)) == bucket0


def test_margin_matches_board_time_advantage():
    """board A's margin is exactly BughouseBoard.time_advantage(team_side).

    This is the join between the two representations: whatever the search prices
    must be the quantity the planes encode.
    """
    board = BughouseBoard(1800)
    board.set_times([[800, 1000], [200, 500]])  # [board][colour], BLACK first

    times = make_times(1000, 800, 500, 200)
    assert sit_margin(times, chess.WHITE, 0) == board.time_advantage(chess.WHITE)
    # Board B's member plays the other colour, hence the negation -- the aliasing
    # bug fixed in 92774eb was writing board A's answer into board B's plane.
    assert sit_margin(times, chess.WHITE, 1) == -board.time_advantage(chess.BLACK)


def test_diagonal_not_same_board():
    times = make_times(1000, 800, 500, 200)
    # A-White races B-White (500), not the same-board A-Black (800).
    assert sit_margin(times, chess.WHITE, 0) == 1000 - 500
    assert sit_margin(times, chess.WHITE, 0) != 1000 - 800


def test_margins_are_antisymmetric_between_rivals():
    times = make_times(1000, 800, 500, 200)
    assert sit_margin(times, True, 0) == -sit_margin(times, False, 1)
    assert sit_margin(times, True, 1) == -sit_margin(times, False, 0)


def test_team_margin_is_zero_sum():
    times = make_times(1000, 800, 500, 200)
    assert team_margin(times, True) == -team_margin(times, False)


def test_flag_is_or_over_members():
    times = make_times(1000, 800, 500, 200)
    assert not team_flagged(times, True)
    times[1][chess.BLACK] = 0  # B-Black flags
    assert team_flagged(times, True)      # white team is out
    assert not team_flagged(times, False)


def test_bucket_symmetry_and_range():
    for m in range(0, 401):
        assert margin_bucket(m) == -margin_bucket(-m)
    for m in range(-100000, 100000, 997):
        assert -5 <= margin_bucket(m) <= 5


def test_bucket_is_monotonic():
    prev = margin_bucket(-100000)
    for m in range(-100000, 100000, 97):
        b = margin_bucket(m)
        assert b >= prev
        prev = b
