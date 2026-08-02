#include "zobrist.h"
#include <random>

namespace Stockfish {
namespace Zobrist {

// Definition of ply-based Zobrist keys (specific to this project)
// Other Zobrist keys are defined in Fairy-Stockfish
Key ply[MAX_PLY];

// Time advantage key - XOR'd into hash when team has time advantage
Key timeAdvantage;

// Per-board, per-bucket sit-margin keys for the clock-aware hash
Key marginBucket[2][TimeControl::MARGIN_BUCKET_COUNT];

// Static initialization
namespace {
    struct ZobristInit {
        ZobristInit() {
            std::mt19937_64 rng(1070372);  // Fixed seed for reproducibility
            
            // Initialize ply-based Zobrist keys
            for (int i = 0; i < MAX_PLY; ++i) {
                ply[i] = rng();
            }
            
            // Initialize time advantage key
            timeAdvantage = rng();

            // Initialize sit-margin bucket keys. Drawn after timeAdvantage so
            // the existing keys keep their values and hashes of clock-free
            // positions are unchanged from before the clock model existed.
            for (int board = 0; board < 2; ++board) {
                for (int b = 0; b < TimeControl::MARGIN_BUCKET_COUNT; ++b) {
                    marginBucket[board][b] = rng();
                }
            }
        }
    } static zobristInit;
}

} // namespace Zobrist
} // namespace Stockfish
