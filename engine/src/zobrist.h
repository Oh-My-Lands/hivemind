#pragma once

#include "Fairy-Stockfish/src/types.h"
#include "time_control.h"

namespace Stockfish {
  namespace Zobrist {
    // Ply-based Zobrist keys for position hashing (specific to this project)
    // Other Zobrist keys are defined in Fairy-Stockfish's position.cpp
    const int MAX_PLY = 1024;
    extern Key ply[MAX_PLY];
    
    // Time advantage key for MCGS transposition detection
    // Positions with different time advantage states are treated as distinct
    extern Key timeAdvantage;

    // One key per board per signed sit-margin bucket, for the clock-aware hash.
    // Bucketed, not per-decisecond: see Board::hash_key_with_clocks for why the
    // resolution has to be coarse or the transposition table stops transposing.
    extern Key marginBucket[2][TimeControl::MARGIN_BUCKET_COUNT];
  }
}