#include <gtest/gtest.h>

#include "../src/board.h"
#include "../src/constants.h"
#include "../src/time_control.h"
#include "Fairy-Stockfish/src/bitboard.h"
#include "Fairy-Stockfish/src/piece.h"
#include "Fairy-Stockfish/src/position.h"
#include "Fairy-Stockfish/src/thread.h"
#include "Fairy-Stockfish/src/types.h"

// Clocks inside Board: charging, refunding, and hashing.
//
// The arithmetic itself is covered without a GPU in engine/tests/logic. What
// needs a real Board is the part that can only go wrong in context: that a ply
// charges the right player, that unmake refunds exactly what make took, and
// that a position with no clock model behaves as it did before clocks existed.
class ClockTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        Stockfish::pieceMap.init();
        Stockfish::variants.init();
        Stockfish::Bitboards::init();
        Stockfish::Position::init();
        Stockfish::Threads.set(1);
        init_policy_index();
    }

    // Both boards at the start position, all four clocks at 3 minutes.
    static Board fresh() {
        Board b;
        b.set_fen(BOARD_A, b.startingFen);
        b.set_fen(BOARD_B, b.startingFen);
        for (int board = 0; board < 2; ++board) {
            b.clocks.set(board, true, 1800);
            b.clocks.set(board, false, 1800);
        }
        return b;
    }

    static Stockfish::Move first_legal(Board& b, int board) {
        for (const Stockfish::ExtMove& m :
             Stockfish::MoveList<Stockfish::LEGAL>(*b.pos[board])) {
            return m;
        }
        return Stockfish::MOVE_NONE;
    }
};

TEST_F(ClockTest, NoTeamLeavesClocksUntouched) {
    Board b = fresh();
    const Stockfish::Move mA = first_legal(b, BOARD_A);

    b.make_moves(mA, Stockfish::MOVE_NONE);  // defaults to NO_TEAM

    EXPECT_EQ(b.clocks.get(0, true), 1800);
    EXPECT_EQ(b.clocks.get(0, false), 1800);
    EXPECT_EQ(b.clocks.get(1, true), 1800);
    EXPECT_EQ(b.clocks.get(1, false), 1800);
}

TEST_F(ClockTest, MoveChargesTheMoverAndNobodyElse) {
    Board b = fresh();
    const Stockfish::Move mA = first_legal(b, BOARD_A);

    b.make_moves(mA, Stockfish::MOVE_NONE, Board::WHITE_TEAM);

    // A-White moved and pays.
    EXPECT_EQ(b.clocks.get(0, true), 1800 - TimeControl::MOVE_COST_DCS);
    // Nobody else moved.
    EXPECT_EQ(b.clocks.get(0, false), 1800);
    EXPECT_EQ(b.clocks.get(1, true), 1800);
    // B-Black is the white team's other member, but it is White to move on
    // board B at the start position, so they are not on turn and do not sit.
    EXPECT_EQ(b.clocks.get(1, false), 1800);
}

TEST_F(ClockTest, SittingCostsTheSitter) {
    Board b = fresh();
    // White team is on turn on board A (White to move) and passes there. That
    // is a sit, not a no-op, and it has to cost.
    b.make_moves(Stockfish::MOVE_NONE, Stockfish::MOVE_NONE, Board::WHITE_TEAM);

    EXPECT_EQ(b.clocks.get(0, true), 1800 - TimeControl::SIT_COST_DCS)
        << "A-White was on turn and sat; the clock must move";
    // B-Black is not on turn at the start position, so sitting there is free.
    EXPECT_EQ(b.clocks.get(1, false), 1800);
}

TEST_F(ClockTest, SittingChargesOnlyTheBoardWhereTheTeamIsOnTurn) {
    Board b = fresh();
    // At the start position White is to move on *both* boards, so each team is
    // on turn on exactly one of them: the white team through A-White, the black
    // team through B-White. A double pass therefore costs each team one clock,
    // not two and not zero.
    b.make_moves(Stockfish::MOVE_NONE, Stockfish::MOVE_NONE, Board::BLACK_TEAM);

    // A-Black is the black team's board-A member and is not on turn.
    EXPECT_EQ(b.clocks.get(0, false), 1800);
    // B-White is their board-B member and is on turn, so sitting costs.
    EXPECT_EQ(b.clocks.get(1, true), 1800 - TimeControl::SIT_COST_DCS);
}

TEST_F(ClockTest, UnmakeRefundsExactlyWhatMakeCharged) {
    Board b = fresh();
    const Stockfish::Move mA = first_legal(b, BOARD_A);

    const TimeControl::Clocks before = b.clocks;
    b.make_moves(mA, Stockfish::MOVE_NONE, Board::WHITE_TEAM);
    ASSERT_NE(b.clocks.get(0, true), before.get(0, true)) << "make must charge";
    b.unmake_moves(mA, Stockfish::MOVE_NONE, Board::WHITE_TEAM);

    for (int board = 0; board < 2; ++board) {
        for (bool white : {true, false}) {
            EXPECT_EQ(b.clocks.get(board, white), before.get(board, white))
                << "board " << board << " white=" << white;
        }
    }
}

TEST_F(ClockTest, UnmakeRefundsASitToo) {
    Board b = fresh();
    const TimeControl::Clocks before = b.clocks;

    b.make_moves(Stockfish::MOVE_NONE, Stockfish::MOVE_NONE, Board::WHITE_TEAM);
    b.unmake_moves(Stockfish::MOVE_NONE, Stockfish::MOVE_NONE, Board::WHITE_TEAM);

    for (int board = 0; board < 2; ++board) {
        for (bool white : {true, false}) {
            EXPECT_EQ(b.clocks.get(board, white), before.get(board, white));
        }
    }
}

TEST_F(ClockTest, RepeatedSittingEventuallyFlags) {
    Board b = fresh();
    // Give A-White barely any time; sitting must run it out rather than
    // continuing for free, which is the whole point of the phase.
    b.clocks.set(0, true, 3 * TimeControl::SIT_COST_DCS);

    EXPECT_FALSE(b.team_flagged(Board::WHITE_TEAM));
    for (int i = 0; i < 3; ++i) {
        b.make_moves(Stockfish::MOVE_NONE, Stockfish::MOVE_NONE, Board::WHITE_TEAM);
    }
    EXPECT_TRUE(b.team_flagged(Board::WHITE_TEAM))
        << "three sits at " << TimeControl::SIT_COST_DCS << " ds each must exhaust "
        << 3 * TimeControl::SIT_COST_DCS << " ds";
    EXPECT_FALSE(b.team_flagged(Board::BLACK_TEAM));
}

TEST_F(ClockTest, CopyConstructorCarriesTheClocks) {
    Board b = fresh();
    b.clocks.set(0, true, 1234);

    Board copy(b);

    EXPECT_EQ(copy.clocks.get(0, true), 1234)
        << "search threads copy the board; zeroed clocks would read as flagged";
    EXPECT_EQ(copy.clocks.get(1, false), 1800);
}

TEST_F(ClockTest, HashIgnoresClocksWithoutAnActingTeam) {
    Board b = fresh();
    const unsigned long plain = b.hash_key(false);
    EXPECT_EQ(b.hash_key_with_clocks(Board::NO_TEAM), plain)
        << "clock-free hashing must match what it was before clocks existed";
}

TEST_F(ClockTest, HashSeparatesPositionsAcrossABucketBoundary) {
    Board a = fresh();
    Board c = fresh();

    // Same pieces on both, but a is level and c is 16s up on the A diagonal --
    // different buckets, so they must not share a transposition entry.
    c.clocks.set(0, true, 1800 + 200);

    EXPECT_NE(a.hash_key_with_clocks(Board::WHITE_TEAM),
              c.hash_key_with_clocks(Board::WHITE_TEAM));
}

TEST_F(ClockTest, HashSharesEntriesWithinABucket) {
    Board a = fresh();
    Board c = fresh();

    // A quarter-second apart: same bucket, so these should transpose. If the
    // margin were hashed per decisecond instead, the table would degenerate
    // into storing every node exactly once.
    c.clocks.set(0, true, 1800 + 2);

    EXPECT_EQ(TimeControl::margin_bucket(a.sit_margin(Board::WHITE_TEAM, 0)),
              TimeControl::margin_bucket(c.sit_margin(Board::WHITE_TEAM, 0)));
    EXPECT_EQ(a.hash_key_with_clocks(Board::WHITE_TEAM),
              c.hash_key_with_clocks(Board::WHITE_TEAM));
}
