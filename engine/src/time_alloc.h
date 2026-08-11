#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

/**
 * @file time_alloc.h
 * @brief Virtual-time node allocation: deciding how much search one move buys.
 *
 * Deciseconds throughout, as in time_control.h, and this header depends on
 * nothing for the same reason -- the arithmetic below is the part most likely
 * to be subtly wrong, and it should be checkable on a laptop with no GPU
 * toolchain. See engine/tests/logic/.
 *
 * The problem this solves
 * ----------------------
 * A benchmark game gives every move a fixed node budget and charges the clock a
 * fixed TimeControl::MOVE_COST_DCS regardless. Search effort and clock spend are
 * therefore two unrelated systems, and neither resembles bughouse: a real player
 * allocates a scarce clock across a game, spending it unevenly.
 *
 * Real time cannot be the currency here. Games in ModelEvaluator run serially on
 * a shared GPU, so wall-clock budgets would make every result depend on device
 * load, and the benchmark's cost would be fixed by the time control rather than
 * by the hardware. So the clock is spent in *nodes* at a fixed exchange rate:
 * the engine picks a node budget per move, the clock is debited nodes/k, and a
 * clock at zero is a flag. Allocation is free and unequal, the constraint is
 * global rather than per-move, and the whole thing stays deterministic.
 *
 * What this is not
 * ----------------
 * This is the *game's* cost model, not the search's. Inside the tree the engine
 * still has to price hypothetical future plies it has not searched yet, and for
 * that TimeControl::MOVE_COST_DCS remains the only thing available. The two
 * models coexist on purpose and should not be conflated.
 */

namespace TimeAlloc {

/**
 * @brief Nodes bought by one decisecond of clock.
 *
 * Calibrated so a typical game's total node spend matches a chosen reference
 * budget per move:  k = reference_nodes / mean_ply_cost_dcs = 800 / 18.2 ~= 44.
 *
 * The *mean*, deliberately -- the opposite of the choice MOVE_COST_DCS makes
 * with the same corpus. That constant answers "what does one typical ply cost",
 * where the median is right and the mean is inflated by the sitting tail. This
 * one answers "how much clock does a whole game consume", which is a total, and
 * totals are governed by the mean. Re-derived 2026-08-10 over 10.9M plies:
 * tc=120 mean 18.2 ds (median 9), tc=180 mean 22.3 (median 10). Using 9 here
 * would hand out half the node bank a real game actually spends.
 */
constexpr int NODES_PER_DECISECOND = 44;

/**
 * @brief Which allocation policy a player uses.
 *
 * FIXED is the pre-existing benchmark: a constant node budget every move, and a
 * clock charged the flat model cost. It is the arm the other two are measured
 * against, and it must stay bit-identical to what it was.
 *
 * FLAT and ARC both spend the clock in nodes and differ only in whether the
 * per-move budget is shaped by game phase. That difference is the experiment:
 * same net, same total bank, one policy spreading it evenly and one spending it
 * where humans do.
 */
enum class Mode {
    FIXED,
    FLAT,
    ARC,
};

/**
 * @brief How much longer than an average ply this ply is worth thinking about.
 *
 * Measured 2026-08-10 from 199,447 chess.com games / 8.5M tc=120 plies, as the
 * mean cost of a player's Nth own ply divided by that player's mean ply cost
 * (18.2 ds). Openings are premoved, middlegames are thought about, endgames are
 * scrambled -- a 6x swing that a single constant cannot express:
 *
 *   ply    0-5   5-10  10-15  15-20  20-30  30-40  40-60   60+
 *   mean   4.5   11.5   20.4   26.5   27.4   21.9   14.6   7.2  ds
 *   mult   0.25  0.63   1.12   1.46   1.51   1.20   0.80   0.40
 *
 * Indexed by the player's *own* ply count, not the macro-ply of the game -- the
 * two differ by a factor of two, and using the wrong one shifts the whole curve
 * a phase early.
 *
 * This is a human reference policy, not a tuned one. It is a starting point and
 * a baseline to beat, which is exactly what makes ARC-vs-FLAT a real question.
 */
inline float phase_multiplier(int ownPly) {
    if (ownPly <  5) return 0.25f;
    if (ownPly < 10) return 0.63f;
    if (ownPly < 15) return 1.12f;
    if (ownPly < 20) return 1.46f;
    if (ownPly < 30) return 1.51f;
    if (ownPly < 40) return 1.20f;
    if (ownPly < 60) return 0.80f;
    return 0.40f;
}

struct Config {
    Mode mode = Mode::FIXED;

    /// Exchange rate. Must be positive; allocate_nodes divides by it.
    int nodesPerDecisecond = NODES_PER_DECISECOND;

    /// Own plies a player expects to make in a game. 45 is the corpus median
    /// for a two-minute game; it only sets how fast the clock is spread, and
    /// minRemainingMoves stops it dividing by zero or by a negative once a game
    /// runs past it.
    int expectedOwnPlies = 45;
    int minRemainingMoves = 10;

    /// A search below the floor is not worth running; one above the ceiling is
    /// a single move eating a bank meant for a game. Both are guards, not
    /// policy -- if either binds often, the exchange rate is wrong.
    size_t minNodes = 32;
    size_t maxNodes = 20000;
};

/**
 * @brief Clock cost of a decided node budget, in deciseconds.
 *
 * Floored at 1: a move that rounds to zero would be free, and a player able to
 * make unlimited free moves can never flag, which quietly removes the terminal
 * state this whole model exists to create. One decisecond is also about what a
 * premove really costs (corpus p10 = 1 ds), so the floor is not a fudge.
 */
inline int cost_dcs(size_t nodes, int nodesPerDecisecond) {
    if (nodesPerDecisecond <= 0) {
        return 1;
    }
    const long dcs = std::lround(static_cast<double>(nodes) / nodesPerDecisecond);
    return static_cast<int>(std::max(1L, dcs));
}

/**
 * @brief Node budget for this move.
 *
 * @param remainingClockDcs Deciseconds on the *shorter* of the acting team's two
 *        clocks. A team loses when either member flags, so the minimum is the
 *        binding constraint; averaging the pair would let a healthy partner mask
 *        a member about to lose the game.
 * @param ownPly How many plies this player has already made.
 *
 * Never returns a budget the clock cannot pay. Note what that does at the end of
 * a lost race: with almost nothing left the allocation collapses toward the
 * floor and the player keeps moving cheaply until it flags, which is what a
 * human scramble looks like and is the correct behaviour rather than a guard
 * against it.
 */
inline size_t allocate_nodes(int remainingClockDcs, int ownPly, const Config& cfg) {
    const int k = std::max(1, cfg.nodesPerDecisecond);

    if (cfg.mode == Mode::FIXED || remainingClockDcs <= 0) {
        return cfg.minNodes;
    }

    const int remainingMoves = std::max(cfg.minRemainingMoves,
                                        cfg.expectedOwnPlies - ownPly);
    const double baseDcs = static_cast<double>(remainingClockDcs) / remainingMoves;
    const double mult = (cfg.mode == Mode::ARC) ? phase_multiplier(ownPly) : 1.0;

    long nodes = std::lround(baseDcs * mult * k);
    nodes = std::clamp(nodes,
                       static_cast<long>(cfg.minNodes),
                       static_cast<long>(cfg.maxNodes));

    // Clamped last, so affordability outranks the floor: spending a bank you do
    // not have would let a player search past its own flag.
    const long affordable = static_cast<long>(remainingClockDcs) * k;
    nodes = std::min(nodes, affordable);

    return static_cast<size_t>(std::max(1L, nodes));
}

}  // namespace TimeAlloc
