#include <gtest/gtest.h>

#include "time_control.h"

using namespace TimeControl;

namespace {

// A deliberately asymmetric position. Every clock differs, so any swapped index
// or flipped sign shows up rather than cancelling.
//
//   board A:  white 1000   black 800
//   board B:  white  500   black 200
//
// White team is A-White + B-Black, so its members hold 1000 and 200.
// Black team is A-Black + B-White, so its members hold 800 and 500.
Clocks asymmetric() {
    Clocks c;
    c.set(0, true, 1000);
    c.set(0, false, 800);
    c.set(1, true, 500);
    c.set(1, false, 200);
    return c;
}

}  // namespace

TEST(Clocks, IndexedByIsWhiteNotByStockfishColor) {
    Clocks c = asymmetric();
    EXPECT_EQ(c.get(0, true), 1000);
    EXPECT_EQ(c.get(0, false), 800);
    EXPECT_EQ(c.get(1, true), 500);
    EXPECT_EQ(c.get(1, false), 200);
}

TEST(Clocks, ChargeSubtracts) {
    Clocks c = asymmetric();
    c.charge(0, true, MOVE_COST_DCS);
    EXPECT_EQ(c.get(0, true), 1000 - MOVE_COST_DCS);
    // Nothing else moved.
    EXPECT_EQ(c.get(0, false), 800);
    EXPECT_EQ(c.get(1, true), 500);
    EXPECT_EQ(c.get(1, false), 200);
}

TEST(MemberIsWhite, TeamPlaysOppositeColoursOnTheTwoBoards) {
    EXPECT_TRUE(member_is_white(true, 0));    // white team on A is White
    EXPECT_FALSE(member_is_white(true, 1));   // ... and Black on B
    EXPECT_FALSE(member_is_white(false, 0));  // black team on A is Black
    EXPECT_TRUE(member_is_white(false, 1));   // ... and White on B
}

// The core claim: the comparison is against the same colour on the OTHER board,
// never against the same-board opponent.
TEST(SitMargin, ComparesAgainstTheDiagonalOpponent) {
    Clocks c = asymmetric();

    // A-White (1000) races B-White (500), not A-Black (800).
    EXPECT_EQ(sit_margin(c, true, 0), 1000 - 500);
    // B-Black (200) races A-Black (800), not B-White (500).
    EXPECT_EQ(sit_margin(c, true, 1), 200 - 800);

    // And from the other team's side.
    EXPECT_EQ(sit_margin(c, false, 0), 800 - 200);   // A-Black races B-Black
    EXPECT_EQ(sit_margin(c, false, 1), 500 - 1000);  // B-White races A-White
}

TEST(SitMargin, IsAntisymmetricBetweenDiagonalRivals) {
    Clocks c = asymmetric();
    // A-White's margin and B-White's margin are the same race, opposite signs.
    EXPECT_EQ(sit_margin(c, true, 0), -sit_margin(c, false, 1));
    // Likewise B-Black against A-Black.
    EXPECT_EQ(sit_margin(c, true, 1), -sit_margin(c, false, 0));
}

TEST(SitMargin, TheTwoBoardsCarryDifferentAnswers) {
    Clocks c = asymmetric();
    // The defect this whole design replaces: one bit, reused for both boards.
    EXPECT_NE(sit_margin(c, true, 0), sit_margin(c, true, 1));
    EXPECT_GT(sit_margin(c, true, 0), 0);  // A-White comfortable
    EXPECT_LT(sit_margin(c, true, 1), 0);  // B-Black in trouble
}

TEST(SitMargin, LevelClocksGiveZero) {
    Clocks c;
    c.set(0, true, 1800);
    c.set(0, false, 1800);
    c.set(1, true, 1800);
    c.set(1, false, 1800);
    EXPECT_EQ(sit_margin(c, true, 0), 0);
    EXPECT_EQ(sit_margin(c, true, 1), 0);
    EXPECT_EQ(team_margin(c, true), 0);
}

TEST(TeamMargin, MatchesTheFrontendDiagonalSumFormula) {
    Clocks c = asymmetric();
    const int aW = 1000, aB = 800, bW = 500, bB = 200;
    // getTeamTimeDiffDeciseconds: (A.white + B.black) - (A.black + B.white)
    EXPECT_EQ(team_margin(c, true), (aW + bB) - (aB + bW));
    EXPECT_EQ(team_margin(c, false), (aB + bW) - (aW + bB));
}

TEST(TeamMargin, IsZeroSumBetweenTheTwoTeams) {
    Clocks c = asymmetric();
    EXPECT_EQ(team_margin(c, true), -team_margin(c, false));
}

TEST(TeamFlagged, IsAnOrOverBothMembersNotAComparison) {
    Clocks c = asymmetric();
    EXPECT_FALSE(team_flagged(c, true));
    EXPECT_FALSE(team_flagged(c, false));

    // Flag only B-Black: the white team is out even though A-White has 1000.
    c.set(1, false, 0);
    EXPECT_TRUE(team_flagged(c, true));
    EXPECT_FALSE(team_flagged(c, false));
}

TEST(TeamFlagged, NegativeCountsAsFlagged) {
    Clocks c = asymmetric();
    c.set(0, true, -1);
    EXPECT_TRUE(team_flagged(c, true));
}

TEST(TeamFlagged, ChargingPastZeroFlags) {
    Clocks c;
    c.set(0, true, SIT_COST_DCS);
    c.set(0, false, 1000);
    c.set(1, true, 1000);
    c.set(1, false, 1000);
    EXPECT_FALSE(team_flagged(c, true));
    c.charge(0, true, SIT_COST_DCS);  // exactly to zero
    EXPECT_TRUE(team_flagged(c, true));
}

TEST(MarginBucket, IsSymmetricAboutZero) {
    for (int m = 0; m <= 400; ++m) {
        EXPECT_EQ(margin_bucket(m), -margin_bucket(-m)) << "margin " << m;
    }
}

TEST(MarginBucket, UsesTheOneTwoFourEightSixteenSecondLadder) {
    EXPECT_EQ(margin_bucket(0), 0);
    EXPECT_EQ(margin_bucket(9), 0);     // under 1s is level for hashing
    EXPECT_EQ(margin_bucket(10), 1);    // 1s
    EXPECT_EQ(margin_bucket(19), 1);
    EXPECT_EQ(margin_bucket(20), 2);    // 2s
    EXPECT_EQ(margin_bucket(39), 2);
    EXPECT_EQ(margin_bucket(40), 3);    // 4s
    EXPECT_EQ(margin_bucket(79), 3);
    EXPECT_EQ(margin_bucket(80), 4);    // 8s
    EXPECT_EQ(margin_bucket(159), 4);
    EXPECT_EQ(margin_bucket(160), 5);   // 16s and up saturates
    EXPECT_EQ(margin_bucket(100000), 5);
}

TEST(MarginBucket, IsMonotonicAndStaysInRange) {
    int prev = margin_bucket(-100000);
    for (int m = -100000; m <= 100000; m += 7) {
        const int b = margin_bucket(m);
        EXPECT_GE(b, prev) << "not monotonic at " << m;
        EXPECT_GE(b, -5);
        EXPECT_LE(b, 5);
        prev = b;
    }
}

TEST(MarginBucket, IndexCoversExactlyTheKeyTable) {
    for (int b = -5; b <= 5; ++b) {
        const int i = margin_bucket_index(b);
        EXPECT_GE(i, 0);
        EXPECT_LT(i, MARGIN_BUCKET_COUNT);
    }
    EXPECT_EQ(margin_bucket_index(-5), 0);
    EXPECT_EQ(margin_bucket_index(5), MARGIN_BUCKET_COUNT - 1);
}

// Shared table, asserted identically by tests/test_time_control_mirror.py.
// The two languages must agree exactly or the search prices sitting one way and
// the training pipeline encodes it another.
TEST(Mirror, SharedTableOfValues) {
    struct Case { int aW, aB, bW, bB; bool teamIsWhite; int m0, m1, team, bucket0; };
    const Case cases[] = {
        // level
        {1800, 1800, 1800, 1800, true,     0,     0,     0,  0},
        // white team up on both diagonals
        {1800, 1000, 1000, 1800, true,   800,   800,  1600,  5},
        // the asymmetric fixture, both perspectives
        {1000,  800,  500,  200, true,   500,  -600,  -100,  5},
        {1000,  800,  500,  200, false,  600,  -500,   100,  5},
        // sub-second differences bucket to level
        {1805, 1800, 1800, 1800, true,     5,     0,     5,  0},
        // one second exactly
        {1810, 1800, 1800, 1800, true,    10,     0,    10,  1},
    };
    for (const Case& c : cases) {
        Clocks k;
        k.set(0, true, c.aW);
        k.set(0, false, c.aB);
        k.set(1, true, c.bW);
        k.set(1, false, c.bB);
        EXPECT_EQ(sit_margin(k, c.teamIsWhite, 0), c.m0);
        EXPECT_EQ(sit_margin(k, c.teamIsWhite, 1), c.m1);
        EXPECT_EQ(team_margin(k, c.teamIsWhite), c.team);
        EXPECT_EQ(margin_bucket(sit_margin(k, c.teamIsWhite, 0)), c.bucket0);
    }
}

TEST(Costs, AreTheMeasuredMedianNotTheMean) {
    // p50 of 14,243 live plies was 9 ds; the mean was 20.1 and is inflated by
    // the sitting these constants price. If someone "improves" these to the
    // mean, this fails and points at the reason.
    EXPECT_EQ(MOVE_COST_DCS, 9);
    EXPECT_EQ(SIT_COST_DCS, 9);
}
