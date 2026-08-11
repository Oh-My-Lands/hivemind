#include <gtest/gtest.h>

#include "time_alloc.h"

using namespace TimeAlloc;

namespace {

Config arc_config() {
    Config cfg;
    cfg.mode = Mode::ARC;
    return cfg;
}

Config flat_config() {
    Config cfg;
    cfg.mode = Mode::FLAT;
    return cfg;
}

}  // namespace

// FIXED is the pre-existing benchmark and must not consult the clock at all --
// callers keep their own constant node budget.
TEST(TimeAllocMode, FixedIgnoresTheClock) {
    Config cfg;  // defaults to FIXED
    EXPECT_EQ(allocate_nodes(1200, 0, cfg), allocate_nodes(30, 40, cfg));
}

TEST(TimeAllocPhase, ArcPeaksInTheMiddlegameAndFallsAtBothEnds) {
    // The shape that motivates the mode: openings are premoved, middlegames are
    // thought about, endgames are scrambled.
    EXPECT_LT(phase_multiplier(2), phase_multiplier(12));
    EXPECT_LT(phase_multiplier(12), phase_multiplier(25));
    EXPECT_GT(phase_multiplier(25), phase_multiplier(45));
    EXPECT_GT(phase_multiplier(45), phase_multiplier(70));
}

TEST(TimeAllocPhase, ArcAveragesToAboutOne) {
    // The multipliers are normalised against the corpus mean ply cost, so a
    // whole game's worth of them should sit near 1. If this drifts, ARC and FLAT
    // are no longer spending the same bank and the A/B compares two things.
    double total = 0.0;
    for (int ply = 0; ply < 45; ++ply) {
        total += phase_multiplier(ply);
    }
    const double mean = total / 45;
    EXPECT_NEAR(mean, 1.0, 0.15);
}

TEST(TimeAllocArc, SpendsMoreInTheMiddlegameThanTheOpening) {
    const Config cfg = arc_config();
    EXPECT_LT(allocate_nodes(1200, 2, cfg), allocate_nodes(1200, 25, cfg));
}

TEST(TimeAllocFlat, IsUnshapedByPhase) {
    // FLAT spreads the clock over the remaining moves and nothing else, which is
    // what makes it the control arm for ARC. At ply 12 there are 45-12 = 33
    // moves left, so the budget is exactly 1200/33 deciseconds' worth of nodes.
    const Config cfg = flat_config();
    const long expected = std::lround(1200.0 / 33 * cfg.nodesPerDecisecond);
    EXPECT_EQ(allocate_nodes(1200, 12, cfg), static_cast<size_t>(expected));

    // And where the phase curve is furthest from 1, the two modes must disagree.
    EXPECT_NE(allocate_nodes(1200, 2, cfg), allocate_nodes(1200, 2, arc_config()));
    EXPECT_NE(allocate_nodes(1200, 25, cfg), allocate_nodes(1200, 25, arc_config()));
}

TEST(TimeAllocBudget, ShrinksAsTheClockShrinks) {
    const Config cfg = arc_config();
    EXPECT_GT(allocate_nodes(1200, 20, cfg), allocate_nodes(600, 20, cfg));
    EXPECT_GT(allocate_nodes(600, 20, cfg), allocate_nodes(100, 20, cfg));
}

TEST(TimeAllocBudget, NeverAllocatesMoreThanTheClockCanPay) {
    Config cfg = arc_config();
    // Two deciseconds left: the affordability clamp has to outrank minNodes,
    // or the player searches a bank it does not have and walks past its flag.
    const size_t nodes = allocate_nodes(2, 20, cfg);
    EXPECT_LE(nodes, static_cast<size_t>(2 * cfg.nodesPerDecisecond));
    EXPECT_GE(nodes, 1u);
}

TEST(TimeAllocBudget, FlaggedClockStillReturnsSomethingSearchable) {
    const Config cfg = arc_config();
    EXPECT_GE(allocate_nodes(0, 20, cfg), 1u);
    EXPECT_GE(allocate_nodes(-50, 20, cfg), 1u);
}

TEST(TimeAllocBudget, LateGameDoesNotDivideByZeroOrGoNegative) {
    // expectedOwnPlies is 45; a game running to 200 plies must not invert the
    // remaining-move count and hand out a negative budget.
    const Config cfg = arc_config();
    for (int ply : {44, 45, 46, 100, 200}) {
        const size_t nodes = allocate_nodes(600, ply, cfg);
        EXPECT_GE(nodes, 1u) << "ply " << ply;
        EXPECT_LE(nodes, cfg.maxNodes) << "ply " << ply;
    }
}

TEST(TimeAllocBudget, RespectsTheCeiling) {
    Config cfg = arc_config();
    cfg.maxNodes = 500;
    EXPECT_LE(allocate_nodes(100000, 25, cfg), 500u);
}

TEST(TimeAllocCost, RoundsToDecisecondsWithAFloorOfOne) {
    EXPECT_EQ(cost_dcs(44, 44), 1);
    EXPECT_EQ(cost_dcs(88, 44), 2);
    EXPECT_EQ(cost_dcs(440, 44), 10);
    // A move that rounds to zero would be free, and unlimited free moves mean a
    // player can never flag.
    EXPECT_EQ(cost_dcs(1, 44), 1);
    EXPECT_EQ(cost_dcs(0, 44), 1);
}

TEST(TimeAllocCost, SurvivesANonsenseExchangeRate) {
    EXPECT_EQ(cost_dcs(800, 0), 1);
    EXPECT_EQ(cost_dcs(800, -5), 1);
}

// The property the whole scheme rests on: a game's total spend is bounded by the
// clock, so allocation is free per move but not free overall.
TEST(TimeAllocBudget, WholeGameSpendIsBoundedByTheClock) {
    const Config cfg = arc_config();
    int clock = 1200;
    int plies = 0;
    while (clock > 0 && plies < 500) {
        const size_t nodes = allocate_nodes(clock, plies, cfg);
        clock -= cost_dcs(nodes, cfg.nodesPerDecisecond);
        ++plies;
    }
    EXPECT_LE(clock, 0);
    // A 1200 ds bank at ~44 nodes/ds should last a game-like number of plies,
    // not three moves and not a thousand.
    EXPECT_GT(plies, 25);
    EXPECT_LT(plies, 300);
}
