#include <gtest/gtest.h>

#include "../src/board.h"
#include "../src/constants.h"
#include "../src/planes.h"
#include "../src/time_encoding.h"
#include <vector>
#include "Fairy-Stockfish/src/bitboard.h"
#include "Fairy-Stockfish/src/piece.h"
#include "Fairy-Stockfish/src/position.h"
#include "Fairy-Stockfish/src/thread.h"
#include "Fairy-Stockfish/src/types.h"

// What reaches channels 31 and 63 of the network input.
//
// The encoder arithmetic is covered without a GPU in engine/tests/logic. What
// needs a real Board is the wiring: that the margin is derived per board rather
// than from one team-wide bit, that the two Phase 2 encodings actually differ
// where they are supposed to, and that a position with no clock model produces
// exactly the planes it did before any of this existed.
class PlaneEncodingTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        Stockfish::pieceMap.init();
        Stockfish::variants.init();
        Stockfish::Bitboards::init();
        Stockfish::Position::init();
        Stockfish::Threads.set(1);
        init_policy_index();
    }

    static Board fresh() {
        Board b;
        b.set_fen(BOARD_A, b.startingFen);
        b.set_fen(BOARD_B, b.startingFen);
        return b;
    }

    // Channels are 8x8; the margin planes are constant across the board, so one
    // cell identifies the whole plane. Asserted rather than assumed, below.
    static constexpr int CH_A = TimeEncoding::TIME_PLANE_CHANNEL_A;
    static constexpr int CH_B = TimeEncoding::TIME_PLANE_CHANNEL_B;

    static float plane_value(const std::vector<float>& planes, int channel) {
        return planes[channel * 64];
    }

    static std::vector<float> fill(Board& board, Stockfish::Color teamSide,
                                   bool bit, TimeEncoding::Mode mode) {
        std::vector<float> planes(NB_INPUT_VALUES(), -99.0f);
        board_to_planes(board, planes.data(), teamSide,
                        plane_margins(board, teamSide, bit, mode));
        return planes;
    }
};

// The pre-Phase 2 contract: no clocks, so both planes carry the team bit and
// nothing about the input has changed. This is the path bench, perft, self-play
// and a clock-blind player all take.
TEST_F(PlaneEncodingTest, WithoutClocksBothPlanesCarryTheTeamBit) {
    Board b = fresh();
    ASSERT_FALSE(b.has_clocks());

    for (TimeEncoding::Mode mode : {TimeEncoding::Mode::BINARY,
                                    TimeEncoding::Mode::CONTINUOUS}) {
        auto on = fill(b, Stockfish::WHITE, true, mode);
        EXPECT_FLOAT_EQ(plane_value(on, CH_A), 1.0f);
        EXPECT_FLOAT_EQ(plane_value(on, CH_B), 1.0f);

        auto off = fill(b, Stockfish::WHITE, false, mode);
        EXPECT_FLOAT_EQ(plane_value(off, CH_A), 0.0f);
        EXPECT_FLOAT_EQ(plane_value(off, CH_B), 0.0f);
    }
}

// The defect this wiring exists to fix. White's board-A member leads their
// diagonal opponent (B-White) while their board-B member trails theirs
// (A-Black), so the two planes must disagree -- one bit for both boards cannot
// represent this position, and it is 10% of the supervised corpus under BINARY.
TEST_F(PlaneEncodingTest, MarginsAreDerivedPerBoardNotPerTeam) {
    Board b = fresh();
    b.set_clocks(/*aWhite=*/1800, /*aBlack=*/1800, /*bWhite=*/1000, /*bBlack=*/1000);

    // White team = A-White + B-Black.
    //   board A margin = A-White - B-White = 1800 - 1000 = +800
    //   board B margin = B-Black - A-Black = 1000 - 1800 = -800
    EXPECT_EQ(b.sit_margin(Board::WHITE_TEAM, 0), 800);
    EXPECT_EQ(b.sit_margin(Board::WHITE_TEAM, 1), -800);

    auto binary = fill(b, Stockfish::WHITE, false, TimeEncoding::Mode::BINARY);
    EXPECT_FLOAT_EQ(plane_value(binary, CH_A), 1.0f);
    EXPECT_FLOAT_EQ(plane_value(binary, CH_B), 0.0f);

    auto continuous = fill(b, Stockfish::WHITE, false, TimeEncoding::Mode::CONTINUOUS);
    EXPECT_GT(plane_value(continuous, CH_A), 0.0f);
    EXPECT_LT(plane_value(continuous, CH_B), 0.0f);
}

// The fallback bit is ignored once there are real clocks. If it were not, a
// clock-aware player would be reading the pre-Phase 4 team bit instead of the
// margin, and the encodings would be indistinguishable in a match.
TEST_F(PlaneEncodingTest, ClockDerivedMarginOverridesTheFallbackBit) {
    Board b = fresh();
    b.set_clocks(1800, 1800, 1000, 1000);

    auto withBitSet = fill(b, Stockfish::WHITE, true, TimeEncoding::Mode::BINARY);
    auto withBitClear = fill(b, Stockfish::WHITE, false, TimeEncoding::Mode::BINARY);
    EXPECT_FLOAT_EQ(plane_value(withBitSet, CH_A), plane_value(withBitClear, CH_A));
    EXPECT_FLOAT_EQ(plane_value(withBitSet, CH_B), plane_value(withBitClear, CH_B));
}

// Perspective. The same position seen by the other team must negate both
// margins -- this is what previously required an explicit negation at the call
// site and is now a property of deriving from teamSide.
TEST_F(PlaneEncodingTest, TheOpposingTeamSeesNegatedMargins) {
    Board b = fresh();
    b.set_clocks(1800, 1800, 1000, 1000);

    auto white = fill(b, Stockfish::WHITE, false, TimeEncoding::Mode::CONTINUOUS);
    auto black = fill(b, Stockfish::BLACK, false, TimeEncoding::Mode::CONTINUOUS);

    EXPECT_NEAR(plane_value(white, CH_A), -plane_value(black, CH_A), 1e-6f);
    EXPECT_NEAR(plane_value(white, CH_B), -plane_value(black, CH_B), 1e-6f);
}

// The two modes agree on sign everywhere. This is the invariant that keeps the
// sit *gate* identical across the two training arms, so a match between them
// measures the network's reading of the margin and not a rules difference.
TEST_F(PlaneEncodingTest, TheTwoEncodingsAgreeOnSign) {
    Board b = fresh();
    for (int bWhite : {200, 800, 1000, 1400, 1800, 2600, 3600}) {
        b.set_clocks(1800, 1800, bWhite, 1800);
        auto binary = fill(b, Stockfish::WHITE, false, TimeEncoding::Mode::BINARY);
        auto continuous = fill(b, Stockfish::WHITE, false, TimeEncoding::Mode::CONTINUOUS);
        for (int ch : {CH_A, CH_B}) {
            EXPECT_EQ(plane_value(binary, ch) > 0.0f, plane_value(continuous, ch) > 0.0f)
                << "channel " << ch << " at bWhite=" << bWhite;
        }
    }
}

// Continuous carries magnitude where binary saturates: two positions the sign
// bit cannot tell apart must produce different planes. This is the entire
// premise of Phase 2, stated as a test.
TEST_F(PlaneEncodingTest, ContinuousDistinguishesMarginsBinaryCannot) {
    Board b = fresh();

    b.set_clocks(1800, 1800, 1790, 1800);   // +10 ds on board A
    auto small = fill(b, Stockfish::WHITE, false, TimeEncoding::Mode::CONTINUOUS);
    auto smallBinary = fill(b, Stockfish::WHITE, false, TimeEncoding::Mode::BINARY);

    b.set_clocks(1800, 1800, 800, 1800);    // +1000 ds on board A
    auto large = fill(b, Stockfish::WHITE, false, TimeEncoding::Mode::CONTINUOUS);
    auto largeBinary = fill(b, Stockfish::WHITE, false, TimeEncoding::Mode::BINARY);

    EXPECT_FLOAT_EQ(plane_value(smallBinary, CH_A), plane_value(largeBinary, CH_A));
    EXPECT_LT(plane_value(small, CH_A), plane_value(large, CH_A));
}

// The margin planes are uniform across all 64 cells, like every other scalar
// plane. A per-cell write would still train and still run, and would quietly
// mean something different to a conv stem.
TEST_F(PlaneEncodingTest, MarginPlanesAreConstantAcrossTheBoard) {
    Board b = fresh();
    b.set_clocks(1800, 1800, 1000, 1000);
    auto planes = fill(b, Stockfish::WHITE, false, TimeEncoding::Mode::CONTINUOUS);

    for (int ch : {CH_A, CH_B}) {
        for (int cell = 1; cell < 64; ++cell) {
            ASSERT_FLOAT_EQ(planes[ch * 64 + cell], planes[ch * 64])
                << "channel " << ch << " cell " << cell;
        }
    }
}

// Parity with the Python encoder, which is what actually wrote the training
// data. Values from src/domain/time_encoding.py's table, via the same tanh
// scale; if either side is retuned this fails on both and neither can drift
// silently.
TEST_F(PlaneEncodingTest, ContinuousMatchesTheTrainingEncoder) {
    Board b = fresh();
    struct Row { int marginDcs; float expected; };
    // squash(m) = tanh(m / 50)
    const Row rows[] = {
        {0,    0.0f},
        {25,   0.46211716f},
        {50,   0.76159416f},
        {100,  0.96402758f},
        {-50, -0.76159416f},
    };
    for (const Row& row : rows) {
        // Board A margin = A-White - B-White, so move B-White by -margin.
        b.set_clocks(1800, 1800, 1800 - row.marginDcs, 1800);
        ASSERT_EQ(b.sit_margin(Board::WHITE_TEAM, 0), row.marginDcs);
        auto planes = fill(b, Stockfish::WHITE, false, TimeEncoding::Mode::CONTINUOUS);
        EXPECT_NEAR(plane_value(planes, CH_A), row.expected, 1e-6f)
            << "margin " << row.marginDcs;
    }
}
