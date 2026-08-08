#pragma once

#include <atomic>
#include <thread>
#include <vector>
#include "searchthread.h"
#include "board.h"
#include "node.h"
#include "engine.h"
#include "search_params.h"
#include "transposition_table.h"
#include "gc_thread.h"
#include "globals.h"
#include "joint_action.h"
#include "rl/rl_settings.h"

/**
 * @brief Search options to configure Agent::run_search behavior.
 */
struct SearchOptions {
    // Stopping conditions (one must be set, unless infinite)
    size_t targetNodes = 0;      // Stop after this many nodes (0 = use time)
    int moveTimeMs = 0;          // Stop after this many milliseconds (0 = use nodes)
    bool infinite = false;       // Search until stopped; ignores targetNodes/moveTimeMs

    // UCI mode options
    bool verbose = false;        // Output UCI info strings (info, bestmove)
    bool checkMateIn1 = false;   // Check for immediate mate before search
    int multiPV = 1;             // Number of principal variations to output
    int analysisBoard = 0;       // Board whose moves MultiPV lines are grouped by (0 = A, 1 = B)

    // Self-play exploration options
    float dirichletAlpha = 0.0f;   // Dirichlet noise alpha (0 = no noise)
    float dirichletEpsilon = 0.0f; // Fraction of prior to replace with noise (0 = no noise)

    // Which representation of the sit margin channels 31 and 63 carry. This is
    // a property of the *network* being searched with, not of the position, so
    // two players running different networks in one match need their own value.
    TimeEncoding::Mode timeEncoding = TimeEncoding::Mode::BINARY;
    
    // Convenience constructors
    static SearchOptions uci(int moveTimeMs, int multiPV = 1, int analysisBoard = 0) {
        SearchOptions opts;
        opts.moveTimeMs = moveTimeMs;
        opts.verbose = true;
        opts.checkMateIn1 = true;
        opts.multiPV = multiPV;
        opts.analysisBoard = analysisBoard;
        return opts;
    }

    static SearchOptions uci_infinite(int multiPV = 1, int analysisBoard = 0) {
        SearchOptions opts;
        opts.infinite = true;
        opts.verbose = true;
        opts.checkMateIn1 = true;
        opts.multiPV = multiPV;
        opts.analysisBoard = analysisBoard;
        return opts;
    }
    
    static SearchOptions selfplay(size_t nodes, const RLSettings& settings) {
        SearchOptions opts;
        opts.targetNodes = nodes;
        opts.verbose = false;
        opts.checkMateIn1 = false;
        opts.dirichletAlpha = settings.dirichletAlpha;
        opts.dirichletEpsilon = settings.dirichletEpsilon;
        return opts;
    }
    
    static SearchOptions selfplay(int moveTimeMs, const RLSettings& settings) {
        SearchOptions opts;
        opts.moveTimeMs = moveTimeMs;
        opts.verbose = false;
        opts.checkMateIn1 = false;
        opts.dirichletAlpha = settings.dirichletAlpha;
        opts.dirichletEpsilon = settings.dirichletEpsilon;
        return opts;
    }
};

/**
 * @brief Manages multi-threaded MCGS (Monte Carlo Graph Search) for Bughouse.
 *
 * Runs multiple search threads in parallel, each with its own engine instance.
 * All threads share the same search graph with thread-safe node operations.
 * Uses a transposition table to detect when different move sequences reach
 * the same position, enabling more efficient value estimation.
 */
class Agent {
private:
    std::vector<SearchThread*> searchThreads;
    std::atomic<bool> running;                            
    shared_ptr<Node> rootNode;
    std::unique_ptr<TranspositionTable> transpositionTable;  // MCGS transposition table
    int numThreads;                                          // Number of search threads
    
    // Tree reuse support (CrazyAra-style)
    std::shared_ptr<Node> ownNextRoot_;      // Expected next root after our move
    std::shared_ptr<Node> opponentsNextRoot_; // Expected next root after opponent's move
    uint64_t lastSearchHash_ = 0;            // Hash of last search position
    
    // Garbage collection thread for async tree cleanup
    GCThread gcThread_;

public:
    /**
     * @brief Constructs a multi-threaded Agent with MCGS support.
     * @param numThreads Number of search threads (0 = use SearchParams::NUM_SEARCH_THREADS)
     */
    Agent(int numThreads = 0);

    /**
     * @brief Destructor to clean up resources.
     */
    ~Agent();

    /**
     * @brief Unified search function for both UCI and self-play modes.
     * @param board The board on which to perform the search.
     * @param engines A vector of engine pointers to use during the search.
     * @param side The side to move.
     * @param teamHasTimeAdvantage If true, team is ahead on time and can double-sit.
     * @param options Search options (stopping conditions, verbosity, noise).
     * @return The best joint action found.
     */
    JointActionCandidate run_search(Board& board, const std::vector<Engine*>& engines, 
                                    Stockfish::Color side, bool teamHasTimeAdvantage,
                                    const SearchOptions& options);
    
    /**
     * @brief Extracts PV line starting from a specific child index.
     * @param board The current board position.
     * @param childIdx The child index to start the PV from.
     * @param maxDepth Maximum number of moves to extract in the PV.
     * @return Space-separated sequence of joint moves.
     */
    std::string extract_pv_from_child(Board& board, int childIdx, int maxDepth);

    /**
     * @brief One candidate move on the board being analysed.
     *
     * The search works on joint actions (moveA, moveB), so a single move on our
     * board appears once per partner-board pairing. For analysis we want one
     * line per distinct move of ours, so those children are collapsed into a
     * group and their statistics combined.
     */
    struct RootMoveGroup {
        Stockfish::Move myMove = Stockfish::MOVE_NONE;  // MOVE_NONE = sit
        int totalVisits = 0;        // summed over every partner pairing
        float weightedQ = 0.0f;     // visit-weighted mean Q, parent's perspective
        float summedPrior = 0.0f;   // marginal policy for myMove
        size_t representativeIdx = 0;  // most-visited child in the group; drives the PV
    };

    /**
     * @brief Groups root children by their move on the analysed board.
     *
     * Groups are ordered by total visits descending, except that the group
     * containing the move extract_best_move() would pick is hoisted to the
     * front so PV 1 always agrees with bestmove.
     *
     * @param analysisBoard Board whose moves to group by (0 = A, 1 = B).
     * @return Groups in display order; empty if the root is not expanded.
     */
    std::vector<RootMoveGroup> group_root_children(int analysisBoard) const;

    /**
     * @brief Prints a single UCI info line for one candidate move.
     *
     * Emits the standard "score cp" alongside the raw q, visit count and policy
     * prior. The extra fields are non-standard, but this engine already reports
     * joint moves that no stock UCI GUI parses, and Q is close to meaningless
     * without the visit count that backs it.
     *
     * @param pvIdx Zero-based PV rank; emits "multipv N" when multiPV > 1.
     */
    void emit_pv_line(Board& board, const RootMoveGroup& group, int pvIdx, int multiPV,
                      int depth, int nodes, int nps, int hashfull, size_t tbhits,
                      double elapsedMs);

    /**
     * @brief Extracts the best move from the root node after search.
     * @param board The board state for move formatting.
     * @return String representation of the best joint move.
     */
    std::string extract_best_move(Board& board);

    /**
     * @brief Extracts the principal variation (PV) by following most-visited children.
     * @param board The current board position.
     * @param maxDepth Maximum number of moves to extract in the PV.
     * @return Space-separated sequence of joint moves.
     */
    std::string extract_pv(Board& board, int maxDepth);

    /**
     * @brief Sets the running state of the agent.
     * @param value Boolean indicating whether the agent should be running.
     */
    void set_is_running(bool value);

    /**
     * @brief Checks if the agent is currently running.
     * @return true if running, false otherwise.
     */
    bool is_running();
    
    /**
     * @brief Set the hash table size in MB.
     * 
     * Resizes the transposition table used for MCGS.
     * @param sizeMB Size in megabytes (1 - 33554432)
     */
    void setHashSize(size_t sizeMB);
    
    /**
     * @brief Get the transposition table for stats reporting.
     */
    TranspositionTable* getTranspositionTable() const {
        return transpositionTable.get();
    }
    
    /**
     * @brief Get the root node after search for extracting visit distributions.
     * Used for AlphaZero-style training data generation.
     * @return Shared pointer to the root node, or nullptr if no search has been run.
     */
    std::shared_ptr<Node> get_root_node() const {
        return rootNode;
    }
    
    /**
     * @brief Try to reuse the search tree from a previous search.
     * 
     * Checks if the given position matches either ownNextRoot_ or opponentsNextRoot_.
     * If found, reuses that subtree as the new root. The old tree portions are
     * queued for async garbage collection.
     * 
     * @param positionHash Hash of the current position
     * @return Shared pointer to reusable root, or nullptr if no reuse possible
     */
    std::shared_ptr<Node> try_reuse_tree(uint64_t positionHash);
    
    /**
     * @brief Store next-root candidates for tree reuse.
     * 
     * Called after search completes to save references to likely next positions:
     * - ownNextRoot_: Most-visited child (our expected move)
     * - opponentsNextRoot_: Most-visited grandchild (opponent's response)
     */
    void store_next_root_candidates();
    
    /**
     * @brief Clear the tree reuse state.
     * Called when starting a new game.
     */
    void clear_tree_reuse() {
        ownNextRoot_.reset();
        opponentsNextRoot_.reset();
        lastSearchHash_ = 0;
    }
};
