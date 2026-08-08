#pragma once

#include "board.h"
#include "utils.h"
#include "time_encoding.h"
#include "Fairy-Stockfish/src/types.h"

/**
 * @brief Encodes the clock state into the two sit-margin plane values.
 *
 * Channels 31 and 63 are per *board*, not per team: a team's ability to sit is
 * a property of each member against their diagonal opponent, so the two can and
 * usually do differ. In the supervised corpus they differ on 10% of positions
 * under BINARY and 88% under CONTINUOUS.
 *
 * When the clock model is off -- `bench`, `perft`, self-play, or a clock-blind
 * player whose clocks have been hidden -- there is no margin to encode and both
 * planes fall back to the caller's team bit, which is exactly the pre-Phase 2
 * behaviour.
 *
 * @param board The current bughouse board state.
 * @param teamSide The colour whose perspective the planes are built from.
 * @param teamHasTimeAdvantage Fallback bit, used only when the board has no clocks.
 * @param mode BINARY for a network trained on the sign bit, CONTINUOUS for one
 *        trained on the squashed margin. Feeding a network the other one reads
 *        as garbage; see TimeEncoding::Mode.
 */
TimeEncoding::MarginPlanes plane_margins(const Board& board,
                                         Stockfish::Color teamSide,
                                         bool teamHasTimeAdvantage,
                                         TimeEncoding::Mode mode);

/**
 * @brief Converts a bughouse board state into neural network input planes.
 *
 * This function maps the current board configuration into a series of float planes,
 * which can then be used as input for machine learning models or further processing.
 *
 * @param board The current bughouse board state.
 * @param inputPlanes Pointer to an array of floats where the resulting planes are stored.
 * @param teamSide The color representing the current team's perspective.
 * @param margins The two encoded sit margins, from plane_margins().
 */
void board_to_planes(Board& board, float* inputPlanes, Stockfish::Color teamSide,
                     TimeEncoding::MarginPlanes margins);
