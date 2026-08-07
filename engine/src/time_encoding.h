#pragma once

#include <cmath>
#include <cstdint>

#include "time_control.h"

/**
 * @file time_encoding.h
 * @brief Encoding of bughouse clock state into network input planes.
 *
 * This is the C++ mirror of `src/domain/time_encoding.py`. The two must agree
 * exactly or the network is trained on one distribution and evaluated on
 * another; `tests/test_time_encoding.py` and `engine/tests/test_time_encoding.cc`
 * check the same table of values on both sides to catch drift.
 *
 * Scope: this header is only about turning a margin into a *plane value*. The
 * clocks themselves, and the margin derived from them, belong to
 * `time_control.h` and are used from there rather than redefined -- two
 * definitions of `sit_margin` that disagreed would be exactly the silent
 * training/inference skew this mirror discipline exists to prevent.
 */

namespace TimeEncoding {

/// Deciseconds of margin mapping to tanh(1). Chosen so the range the viewer
/// already treats as meaningful (1s/2s/4s/8s/16s tiers in clockAdvantage.ts)
/// spreads across most of the output range instead of saturating at once.
constexpr float SQUASH_SCALE_DCS = 50.0f;

/// Affine map used to fit a signed margin into uint8 training-data storage.
constexpr float PLANE_QUANT_SCALE = 100.0f;
constexpr int PLANE_QUANT_OFFSET = 128;

/// Channels holding the sit margin, one per board.
constexpr int TIME_PLANE_CHANNEL_A = 31;
constexpr int TIME_PLANE_CHANNEL_B = 63;

/**
 * @brief Which representation of the sit margin goes into channels 31 and 63.
 *
 * Mirrors `TimeEncoding` in src/domain/time_encoding.py, and is exposed as a
 * UCI combo option. BINARY is the deployed network's encoding; CONTINUOUS
 * requires a network trained against it and will read as garbage otherwise.
 */
enum class Mode {
    BINARY,
    CONTINUOUS,
};

/**
 * @brief Maps a signed clock margin in deciseconds onto [-1, 1].
 *
 * Positive means the player on that board leads their diagonal opponent and
 * can afford to sit; 0.0 means the clocks are level.
 */
inline float squash_margin(float marginDcs) {
    return std::tanh(marginDcs / SQUASH_SCALE_DCS);
}

/**
 * @brief Encodes one board's sit margin for its input plane.
 *
 * The two modes agree on sign -- BINARY is 1.0 exactly when CONTINUOUS is
 * positive -- which is what keeps the sit gate identical across the Phase 2
 * training arms.
 */
inline float encode_margin(int marginDcs, Mode mode) {
    if (mode == Mode::BINARY) {
        return marginDcs > 0 ? 1.0f : 0.0f;
    }
    return squash_margin(static_cast<float>(marginDcs));
}

inline uint8_t quantize_margin_plane(float value) {
    return static_cast<uint8_t>(std::lround(value * PLANE_QUANT_SCALE) + PLANE_QUANT_OFFSET);
}

inline float dequantize_margin_plane(float value) {
    return (value - static_cast<float>(PLANE_QUANT_OFFSET)) / PLANE_QUANT_SCALE;
}

/**
 * @brief A team's two plane values, already encoded.
 *
 * `a` belongs to the team's board-A member, `b` to their board-B member.
 */
struct MarginPlanes {
    float a = 0.0f;
    float b = 0.0f;
};

/**
 * @brief Builds a team's two encoded margins from the four raw clocks.
 *
 * Teams are diagonal pairings, so each member races the player of *their own
 * colour* on the other board -- the one who gates their supply of pieces. That
 * derivation lives in TimeControl::sit_margin; this only encodes its result.
 */
inline MarginPlanes margin_planes(const TimeControl::Clocks& clocks,
                                  bool teamIsWhiteOnBoardA,
                                  Mode mode) {
    return MarginPlanes{
        encode_margin(TimeControl::sit_margin(clocks, teamIsWhiteOnBoardA, 0), mode),
        encode_margin(TimeControl::sit_margin(clocks, teamIsWhiteOnBoardA, 1), mode)
    };
}

} // namespace TimeEncoding
