/*
 * Tests for the sit-margin *encoding* -- turning a margin into a plane value.
 *
 * PARITY_TABLE below is duplicated verbatim in tests/test_time_encoding.py.
 * If you change the squash function, change both -- a network trained through
 * one encoding and evaluated through the other reads a different input
 * distribution, and nothing else in the system would notice.
 *
 * The margin algebra itself (diagonal pairing, sign conventions, flag
 * semantics) belongs to time_control.h and is covered by test_time_control.cc.
 * Nothing here re-derives it.
 */

#include <gtest/gtest.h>

#include <cmath>

#include "time_encoding.h"

namespace {

struct ParityRow {
    float marginDcs;
    float expected;
};

// Mirrored in tests/test_time_encoding.py PARITY_TABLE.
const ParityRow PARITY_TABLE[] = {
    {-800.0f, -1.00000000f},
    {-400.0f, -0.99999977f},
    {-100.0f, -0.96402758f},
    {-10.0f,  -0.19737532f},
    {0.0f,     0.00000000f},
    {10.0f,    0.19737532f},
    {20.0f,    0.37994896f},
    {40.0f,    0.66403677f},
    {80.0f,    0.92166855f},
    {160.0f,   0.99668240f},
};

TEST(TimeEncoding, SquashMatchesPythonParityTable) {
    for (const ParityRow& row : PARITY_TABLE) {
        EXPECT_NEAR(TimeEncoding::squash_margin(row.marginDcs), row.expected, 1e-6f)
            << "squash_margin(" << row.marginDcs << ") drifted from the Python mirror";
    }
}

TEST(TimeEncoding, SquashIsOdd) {
    EXPECT_FLOAT_EQ(TimeEncoding::squash_margin(0.0f), 0.0f);
    for (float m : {1.0f, 10.0f, 55.0f, 300.0f, 2000.0f}) {
        EXPECT_NEAR(TimeEncoding::squash_margin(-m), -TimeEncoding::squash_margin(m), 1e-7f);
    }
}

TEST(TimeEncoding, SquashIsStrictlyMonotonicWhereItHasToBe) {
    // Strict only out to +/-30s. Beyond roughly +/-41.5s float32 tanh has
    // saturated to exactly 1.0 and successive margins compare equal -- the
    // Python mirror runs in float64 and saturates slightly later, but the two
    // agree to ~1e-8 there and quantise to the same byte from 15s upward, so
    // the difference is unobservable to the network. What must hold is that
    // the encoding stays ordered across the range it was scaled for: the
    // 1s/2s/4s/8s/16s tier ladder tops out at 160 dcs.
    float previous = -2.0f;
    for (int m = -300; m <= 300; m += 25) {
        float v = TimeEncoding::squash_margin(static_cast<float>(m));
        EXPECT_GT(v, previous) << "not ordered at " << m << " dcs";
        previous = v;
    }
}

TEST(TimeEncoding, SquashNeverDecreases) {
    float previous = -2.0f;
    for (int m = -3000; m <= 3000; m += 25) {
        float v = TimeEncoding::squash_margin(static_cast<float>(m));
        EXPECT_GE(v, previous) << "went backwards at " << m << " dcs";
        previous = v;
    }
}

TEST(TimeEncoding, SquashIsBounded) {
    // Bounded, so the affine quantisation cannot overflow uint8.
    for (float m : {-100000.0f, -3000.0f, 0.0f, 3000.0f, 100000.0f}) {
        float v = TimeEncoding::squash_margin(m);
        EXPECT_GE(v, -1.0f);
        EXPECT_LE(v, 1.0f);
        const uint8_t stored = TimeEncoding::quantize_margin_plane(v);
        EXPECT_GE(stored, 28);
        EXPECT_LE(stored, 228);
    }
}

TEST(TimeEncoding, QuantizeRoundTrips) {
    for (const ParityRow& row : PARITY_TABLE) {
        const float squashed = TimeEncoding::squash_margin(row.marginDcs);
        const uint8_t stored = TimeEncoding::quantize_margin_plane(squashed);
        EXPECT_NEAR(TimeEncoding::dequantize_margin_plane(static_cast<float>(stored)),
                    squashed, 1.0f / TimeEncoding::PLANE_QUANT_SCALE);
    }
}

TEST(TimeEncoding, QuantizePreservesSign) {
    // The point of the affine map: a bare uint8 cast would wrap these.
    for (float m : {-1.0f, -10.0f, -50.0f, -400.0f}) {
        const uint8_t stored = TimeEncoding::quantize_margin_plane(TimeEncoding::squash_margin(m));
        EXPECT_LT(stored, TimeEncoding::PLANE_QUANT_OFFSET);
        EXPECT_LT(TimeEncoding::dequantize_margin_plane(stored), 0.0f);
    }
    for (float m : {1.0f, 10.0f, 50.0f, 400.0f}) {
        const uint8_t stored = TimeEncoding::quantize_margin_plane(TimeEncoding::squash_margin(m));
        EXPECT_GT(stored, TimeEncoding::PLANE_QUANT_OFFSET);
        EXPECT_GT(TimeEncoding::dequantize_margin_plane(stored), 0.0f);
    }
}

TEST(TimeEncoding, BinarySurvivesTheAffineMapExactly) {
    // One storage format serves both training arms, so the loader never has to
    // know which wrote a shard. That is only safe if 0/1 round-trips exactly.
    for (float v : {0.0f, 1.0f}) {
        const uint8_t stored = TimeEncoding::quantize_margin_plane(v);
        EXPECT_FLOAT_EQ(TimeEncoding::dequantize_margin_plane(stored), v);
    }
}

TEST(TimeEncoding, TheTwoModesAgreeOnSign) {
    // The property the Phase 2 A/B rests on. Sit permission is read off the
    // sign of this plane, so if the modes disagreed anywhere the two arms would
    // be trained on differently composed datasets and the comparison would
    // confound the encoding with the corpus.
    for (int m : {-4000, -400, -50, -10, -1, 0, 1, 10, 50, 400, 4000}) {
        const float binary = TimeEncoding::encode_margin(m, TimeEncoding::Mode::BINARY);
        const float continuous = TimeEncoding::encode_margin(m, TimeEncoding::Mode::CONTINUOUS);
        EXPECT_EQ(binary > 0.0f, continuous > 0.0f)
            << "modes disagree on sign at " << m << " dcs";
    }
}

TEST(TimeEncoding, BinaryModeIsZeroOrOne) {
    for (int m : {-400, -1, 0, 1, 400}) {
        const float v = TimeEncoding::encode_margin(m, TimeEncoding::Mode::BINARY);
        EXPECT_TRUE(v == 0.0f || v == 1.0f);
    }
}

TEST(TimeEncoding, LevelClocksAreNotAnAdvantageInEitherMode) {
    EXPECT_FLOAT_EQ(TimeEncoding::encode_margin(0, TimeEncoding::Mode::BINARY), 0.0f);
    EXPECT_FLOAT_EQ(TimeEncoding::encode_margin(0, TimeEncoding::Mode::CONTINUOUS), 0.0f);
}

TEST(TimeEncoding, MarginPlanesEncodeTheDiagonalMargins) {
    // A-White 60.0s, A-Black 60.0s, B-White 10.0s, B-Black 10.0s.
    // White team: A-White leads B-White by +500, B-Black trails A-Black by -500.
    // Mirrored in tests/test_time_encoding.py::test_planes_carry_per_board_margins.
    TimeControl::Clocks c;
    c.set(0, true, 600);
    c.set(0, false, 600);
    c.set(1, true, 100);
    c.set(1, false, 100);

    const auto continuous =
        TimeEncoding::margin_planes(c, true, TimeEncoding::Mode::CONTINUOUS);
    EXPECT_NEAR(continuous.a, TimeEncoding::squash_margin(500.0f), 1e-7f);
    EXPECT_NEAR(continuous.b, TimeEncoding::squash_margin(-500.0f), 1e-7f);

    // The case the old single-bit encoding could not represent: one member is
    // ahead while the other is behind by the same amount.
    const auto binary = TimeEncoding::margin_planes(c, true, TimeEncoding::Mode::BINARY);
    EXPECT_FLOAT_EQ(binary.a, 1.0f);
    EXPECT_FLOAT_EQ(binary.b, 0.0f);
}

}  // namespace
