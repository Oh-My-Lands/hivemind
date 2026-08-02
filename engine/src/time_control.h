#pragma once

#include <cstdlib>

/**
 * @file time_control.h
 * @brief Clock state, sit margins, move costs and flag detection.
 *
 * Everything here is deciseconds, matching chess.com's timestamps and
 * src/domain/board.py's `times`.
 *
 * This header deliberately depends on nothing -- not Stockfish, not the engine,
 * not CUDA -- so it can be unit tested without a GPU toolchain. The engine
 * converts at the call site (`side == Stockfish::WHITE`).
 *
 * Why a margin and not a bit
 * --------------------------
 * A team's ability to sit is not a property of the team, it is a property of
 * each *player* against their DIAGONAL opponent. Teams are diagonal pairings
 * (board-A White partners board-B Black), so the player deciding whether to sit
 * on board A is racing the board-B player of the same colour: if that opponent
 * sits, board A never receives the piece it is waiting for and the two clocks
 * burn against each other. That makes the decision variable a signed margin,
 * one per board, not one team-wide bit.
 */

namespace TimeControl {

/**
 * Cost of one ply to the player who makes it.
 *
 * Measured 2026-08-02 over 135 live chess.com bughouse games / 14,243 plies,
 * all at tc=180: p50 = 9 ds, p75 = 18, p90 = 35, p99 = 216, mean = 20.1.
 *
 * The median, not the mean. Mean is 2.2x median because the tail *is* the
 * sitting behaviour this constant exists to price -- using it would bake the
 * modelled quantity into the model.
 */
constexpr int MOVE_COST_DCS = 9;

/**
 * Cost of one sit ply to the sitter.
 *
 * Equal to MOVE_COST_DCS on purpose, and not as a fudge: a sit lasts until
 * something happens on the other board, and one partner-board ply is drawn from
 * the same distribution as any other ply. So sitting N plies costs ~9N ds and is
 * bounded by the clock -- which is the pressure the search currently lacks,
 * where a sit is MOVE_NONE and decrements nothing.
 */
constexpr int SIT_COST_DCS = 9;

/**
 * Team uptime, in deciseconds, at which a team is treated as able to sit.
 *
 * Matches MODE_SIT_THRESHOLD_DCS in the frontend's engineMode.ts, so the engine
 * and the UI answer "is this team up on time" the same way. It is a floor
 * rather than a deadband: below it a team is treated as "go", the conservative
 * default, because that is the smaller action space and cannot invent a
 * double-sit the team has not earned.
 */
constexpr int SIT_THRESHOLD_DCS = 15;

/**
 * @brief The four clocks of a bughouse game, in deciseconds.
 *
 * Indexed [board][isWhite], board 0 = A and 1 = B.
 *
 * NOTE the second index is a *bool meaning "is white"*, chosen to mirror
 * src/domain/board.py's `times[board_num][colour]`, where python-chess has
 * WHITE == True == 1. Stockfish's Color is the other way round (WHITE == 0), so
 * a Stockfish::Color must never be used as this index directly -- convert with
 * `(c == Stockfish::WHITE)`. Getting this wrong swaps every clock in the game
 * and still runs.
 */
struct Clocks {
    int t[2][2] = {{0, 0}, {0, 0}};

    constexpr int get(int board, bool isWhite) const {
        return t[board][isWhite ? 1 : 0];
    }
    constexpr void set(int board, bool isWhite, int dcs) {
        t[board][isWhite ? 1 : 0] = dcs;
    }
    constexpr void charge(int board, bool isWhite, int dcs) {
        t[board][isWhite ? 1 : 0] -= dcs;
    }
};

/**
 * @brief Colour played by a team's member on a given board.
 *
 * Teams are diagonal: the team identified by `teamIsWhite` plays that colour on
 * board A and the opposite colour on board B.
 */
constexpr bool member_is_white(bool teamIsWhite, int board) {
    return board == 0 ? teamIsWhite : !teamIsWhite;
}

/**
 * @brief Signed sit margin for a team's member on `board`, in deciseconds.
 *
 * Positive means that member is ahead of their diagonal opponent and can afford
 * to sit. The diagonal opponent is the player of the *same colour* on the other
 * board -- which is what makes this the right comparison rather than the
 * same-board opponent, whose clock is frozen while it is this player's turn.
 *
 * Mirrors board.py's `sit_margin`; for board A and a white team this is exactly
 * `time_advantage(WHITE)`.
 */
constexpr int sit_margin(const Clocks& c, bool teamIsWhite, int board) {
    const bool mine = member_is_white(teamIsWhite, board);
    return c.get(board, mine) - c.get(1 - board, mine);
}

/**
 * @brief Team-level uptime: both members' margins summed.
 *
 * For the white team this is (A.white + B.black) - (A.black + B.white), the same
 * quantity as `getTeamTimeDiffDeciseconds` in the frontend's clockAdvantage.ts.
 */
constexpr int team_margin(const Clocks& c, bool teamIsWhite) {
    return sit_margin(c, teamIsWhite, 0) + sit_margin(c, teamIsWhite, 1);
}

/**
 * @brief Has either member of this team run out of time?
 *
 * A bughouse team loses if *either* player flags, so this is an OR over the two
 * members, not a comparison between them.
 */
constexpr bool team_flagged(const Clocks& c, bool teamIsWhite) {
    return c.get(0, member_is_white(teamIsWhite, 0)) <= 0
        || c.get(1, member_is_white(teamIsWhite, 1)) <= 0;
}

/**
 * @brief Coarse bucket of a signed margin, for folding into the position hash.
 *
 * The margin must not enter the Zobrist key at full resolution: two positions
 * one decisecond apart would become different table entries and the
 * transposition table would stop transposing entirely.
 *
 * Magnitude tiers are the 1s/2s/4s/8s/16s ladder the frontend already uses for
 * this quantity (`getAdvantageTier`), so engine and UI bucket it the same way.
 * One deliberate difference: tier 0 here means "level to within a second",
 * whereas the frontend reserves 0 for exactly equal, because for hashing the
 * near-level cases should share a bucket rather than split on noise.
 *
 * @return -5..5 inclusive, so 11 Zobrist keys cover it.
 */
constexpr int margin_bucket(int marginDcs) {
    const int mag = marginDcs < 0 ? -marginDcs : marginDcs;
    int tier;
    if      (mag <  10) tier = 0;
    else if (mag <  20) tier = 1;
    else if (mag <  40) tier = 2;
    else if (mag <  80) tier = 3;
    else if (mag < 160) tier = 4;
    else                tier = 5;
    return marginDcs < 0 ? -tier : tier;
}

/// Number of distinct values margin_bucket can return (-5..5).
constexpr int MARGIN_BUCKET_COUNT = 11;

/// Shifts a bucket into [0, MARGIN_BUCKET_COUNT) for indexing a key table.
constexpr int margin_bucket_index(int bucket) { return bucket + 5; }

}  // namespace TimeControl
