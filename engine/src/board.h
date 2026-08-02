#pragma once

#include "constants.h"
#include "time_control.h"
#include "zobrist.h"

#include <sstream>
#include <string>
#include <algorithm>
#include <iostream>
#include <vector>
#include <functional>

#include "Fairy-Stockfish/src/apiutil.h"
#include "Fairy-Stockfish/src/position.h"
#include "Fairy-Stockfish/src/thread.h"
#include "Fairy-Stockfish/src/types.h"
#include "Fairy-Stockfish/src/uci.h"

/**
 * @brief Represents a chess board with dual perspectives.
 *
 * This class encapsulates the state of a chess board, including position data and move history,
 * and provides methods to manipulate and query the board state using FEN strings and UCI moves.
 */
class Board {
    public: 
        std::unique_ptr<Stockfish::Position> pos[2]; ///< Array of board positions.
        Stockfish::StateListPtr states[2];             ///< Array of state history pointers.
        const std::string startingFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"; ///< Standard starting position in FEN notation.
        
        /// History of board-only position keys for repetition detection (ignores pocket pieces)
        std::vector<uint64_t> positionHistory[2];

        /// The four clocks, deciseconds. Zero throughout means "no clock model",
        /// which is how every caller that does not pass an acting team to
        /// make_moves leaves it -- see the actingTeam parameter there.
        TimeControl::Clocks clocks;

        /// Whether the clock model is active; see set_clocks.
        bool clocksEnabled = false;

        Board();
        Board(const Board& board);

        /**
        * @brief Generates a hash key for the board.
        * @param teamHasTimeAdvantage Whether the team to move has time advantage (affects sitting rules)
        * @return long unsigned int Combined hash of the positions.
        */
        unsigned long hash_key(bool teamHasTimeAdvantage = false) {
            auto k0 = pos[0]->key() ^ Stockfish::Zobrist::ply[game_ply(0)];
            auto k1 = pos[1]->key() ^ Stockfish::Zobrist::ply[game_ply(1)];
            // Combines the two keys using a hash_combine technique.
            auto combined = k0 ^ (k1 + 0x9e3779b97f4a7c15UL + (k0 << 6) + (k0 >> 2));
            // XOR in time advantage key if team is up on time
            return teamHasTimeAdvantage ? (combined ^ Stockfish::Zobrist::timeAdvantage) : combined;
        }

        /**
        * @brief Hash key including the clock state, for the acting team.
        *
        * Two positions identical on both boards but with different clocks are
        * genuinely different positions -- one can afford to sit and the other
        * cannot -- so they must not share a transposition entry.
        *
        * The margin is bucketed rather than hashed at full resolution. At one
        * key per decisecond nothing would ever transpose: two nodes a single
        * tick apart would be distinct entries and the table would degenerate
        * into a very expensive way of storing each node once.
        */
        unsigned long hash_key_with_clocks(int actingTeam) {
            auto k0 = pos[0]->key() ^ Stockfish::Zobrist::ply[game_ply(0)];
            auto k1 = pos[1]->key() ^ Stockfish::Zobrist::ply[game_ply(1)];
            auto combined = k0 ^ (k1 + 0x9e3779b97f4a7c15UL + (k0 << 6) + (k0 >> 2));

            if (actingTeam == NO_TEAM || !has_clocks()) {
                return combined;
            }
            for (int board = 0; board < 2; ++board) {
                const int bucket = TimeControl::margin_bucket(sit_margin(actingTeam, board));
                combined ^= Stockfish::Zobrist::marginBucket[board]
                                                            [TimeControl::margin_bucket_index(bucket)];
            }
            return combined;
        }

        /**
         * @brief Computes a hash key for a single board ignoring pocket pieces.
         * Used for 3-fold repetition detection where only board position matters.
         * @param board_num The board index.
         * @return uint64_t Hash of the board position without pocket pieces.
         */
        uint64_t board_only_key(int board_num) {
            // Extract the FEN and hash only the board-relevant parts (not pocket)
            std::string fenStr = pos[board_num]->fen();
            
            // FEN format: piece_placement side_to_move castling en_passant halfmove fullmove [pocket]
            // We only want: piece_placement + side_to_move + castling + en_passant
            std::stringstream ss(fenStr);
            std::string piecePlacement, sideToMove, castling, enPassant;
            ss >> piecePlacement >> sideToMove >> castling >> enPassant;

            size_t pos = piecePlacement.find('[');
            if (pos != std::string::npos) {
                piecePlacement = piecePlacement.substr(0, pos);
            }
            
            // Create a normalized string for hashing (ignore halfmove, fullmove, pocket)
            std::string boardOnlyFen = piecePlacement + " " + sideToMove + " " + castling + " " + enPassant;
            
            // Use std::hash for the string
            return std::hash<std::string>{}(boardOnlyFen);
        }

        /**
         * @brief Adds current position to history for repetition tracking.
         * @param board_num The board index.
         */
        void record_position(int board_num) {
            positionHistory[board_num].push_back(board_only_key(board_num));
        }

        /**
         * @brief Removes the last position from history (for unmake_moves).
         * @param board_num The board index.
         */
        void unrecord_position(int board_num) {
            if (!positionHistory[board_num].empty()) {
                positionHistory[board_num].pop_back();
            }
        }

        /**
         * @brief Clears position history for a board.
         * @param board_num The board index.
         */
        void clear_position_history(int board_num) {
            positionHistory[board_num].clear();
        }

        /**
         * @brief Swaps the positions and states of the two boards.
         */
        void swap_boards() {
            std::swap(pos[0], pos[1]);
            std::swap(states[0], states[1]);
            std::swap(positionHistory[0], positionHistory[1]);
        }

        void set(std::string fen);

        /**
         * @brief Sets a single board from one half of a dual FEN.
         * @param board_num The board index.
         * @param line FEN for that board, already trimmed and without the '|'.
         */
        void set_board(int board_num, const std::string& line); 
        void push_move(int board_num, Stockfish::Move move);
        /// actingTeam values for make_moves/unmake_moves.
        static constexpr int NO_TEAM = -1;     ///< Do not touch the clocks at all.
        static constexpr int WHITE_TEAM = 0;   ///< A-White + B-Black.
        static constexpr int BLACK_TEAM = 1;   ///< A-Black + B-White.

        /**
        * @brief Applies one team's joint action.
        *
        * @param actingTeam Which team is moving, or NO_TEAM to leave the clocks
        *        untouched. It cannot be inferred from the moves: MOVE_NONE on a
        *        board means either "not their turn" or "they sat", and those
        *        cost different amounts of time. The double-sit case has no real
        *        move to infer from at all, and that is precisely the action that
        *        most needs charging.
        *
        *        Defaulting to NO_TEAM keeps every caller that has no clock model
        *        -- bench, perft -- behaving exactly as before.
        */
        void make_moves(Stockfish::Move moveA, Stockfish::Move moveB, int actingTeam = NO_TEAM);
        void unmake_moves(Stockfish::Move moveA, Stockfish::Move moveB, int actingTeam = NO_TEAM);

        /// Signed sit margin for `actingTeam`'s member on `board`, deciseconds.
        int sit_margin(int actingTeam, int board) const {
            return TimeControl::sit_margin(clocks, actingTeam == WHITE_TEAM, board);
        }

        /// True when either of `actingTeam`'s members has run out of time.
        bool team_flagged(int actingTeam) const {
            return TimeControl::team_flagged(clocks, actingTeam == WHITE_TEAM);
        }

        /// Charges (sign +1) or refunds (sign -1) one ply on `board`.
        void charge_clock(int board, Stockfish::Move move, int actingTeam, int sign);

        /**
        * @brief Turns the clock model on, with all four clocks in deciseconds.
        *
        * Until this is called the clocks are inert: make_moves charges nothing
        * however it is called, so threading an acting team through the search
        * changes no behaviour on its own. That matters because charging from an
        * uninitialised zero would drive every clock negative and make
        * team_flagged read true at every node.
        */
        void set_clocks(int aWhite, int aBlack, int bWhite, int bBlack) {
            clocks.set(0, true, aWhite);
            clocks.set(0, false, aBlack);
            clocks.set(1, true, bWhite);
            clocks.set(1, false, bBlack);
            clocksEnabled = true;
        }

        /// True when a clock model is in play.
        bool has_clocks() const { return clocksEnabled; }
        void pop_move(int board_num);
        std::vector<Stockfish::Move> legal_moves(int board_num);
        std::vector<std::pair<int, Stockfish::Move>> legal_moves(Stockfish::Color side, bool teamHasTimeAdvantage = false);

        /**
         * @brief Adds a piece to the player's hand.
         * @param board_num The index of the board.
         * @param p The piece to add.
         */
        void add_to_hand(int board_num, Stockfish::Piece p) {
            pos[board_num]->add_to_hand(p);
        }

        /**
         * @brief Removes a piece from the player's hand.
         * @param board_num The index of the board.
         * @param p The piece to remove.
         */
        void remove_from_hand(int board_num, Stockfish::Piece p) {
            pos[board_num]->remove_from_hand(p);
        }

        /**
         * @brief Returns the FEN string for the specified board.
         * @param board_num The board index.
         * @return std::string FEN representation of the board.
         */
        std::string fen(int board_num) {
            return pos[board_num]->fen(); 
        }

        /**
         * @brief Sets the board state from a FEN string.
         * @param board_num The board index.
         * @param fen FEN string representing the board state.
         */
        void set_fen(int board_num, std::string fen) {
            states[board_num] = Stockfish::StateListPtr(new std::deque<Stockfish::StateInfo>(1));
            states[board_num]->emplace_back();
            pos[board_num]->set(Stockfish::variants.find("bughouse")->second, fen, false, &states[board_num]->back(), Stockfish::Threads.main());
            // Reset position history for this board
            clear_position_history(board_num);
            record_position(board_num);
        }

        /**
         * @brief Returns a string representing the pieces held in hand.
         * @param board_num The board index.
         * @return std::string A string encoding the pieces in hand.
         */
        std::string get_hand(int board_num) {
            std::string hand; 
            for (Stockfish::Color color : {Stockfish::WHITE, Stockfish::BLACK}) {
                for (Stockfish::PieceType piece = Stockfish::QUEEN; piece >= Stockfish::PAWN; --piece) {
                    int pocket_cnt = count_in_hand(board_num, color, piece);
                    hand += std::string(pocket_cnt, pos[board_num]->piece_to_char()[Stockfish::make_piece(color, piece)]);
                }
            }
            return hand; 
        }
        
        /**
         * @brief Determines if a move is a capture.
         * @param board_num The board index.
         * @param move The move to evaluate.
         * @return true if the move is a capture, false otherwise.
         */
        bool is_capture(int board_num, Stockfish::Move move) {
            return pos[board_num]->capture(move);
        }

        /**
         * @brief Retrieves the character representing the captured piece.
         * @param board_num The board index.
         * @param move The move to evaluate.
         * @return char Character corresponding to the captured piece.
         */
        char get_captured_piece(int board_num, Stockfish::Move move) {
            Stockfish::Color us = side_to_move(board_num);
            Stockfish::Color them = ~us;
            Stockfish::Square to = to_sq(move);
            Stockfish::Piece captured = type_of(move) == Stockfish::EN_PASSANT ? make_piece(them, Stockfish::PAWN) : pos[board_num]->piece_on(to);
            
            return pos[board_num]->piece_to_char()[captured];
        }
        
        /**
         * @brief Counts the number of a specific piece in hand.
         * @param board_num The board index.
         * @param c The color of the piece.
         * @param pt The type of the piece.
         * @return int Number of pieces of the specified type in hand.
         */
        int count_in_hand(int board_num, Stockfish::Color c, Stockfish::PieceType pt) {
            return pos[board_num]->count_in_hand(c, pt);
        }

        /**
         * @brief Returns the side to move on the specified board.
         * @param board_num The board index.
         * @return Stockfish::Color The color of the side to move.
         */
        Stockfish::Color side_to_move(int board_num) {
            return pos[board_num]->side_to_move();
        }

        /**
         * @brief Returns a bitboard representing pieces of a specific type and color.
         * @param board_num The board index.
         * @param c The color of the pieces.
         * @param pt The piece type.
         * @return Stockfish::Bitboard Bitboard for the pieces.
         */
        Stockfish::Bitboard pieces(int board_num, Stockfish::Color c, Stockfish::PieceType pt) {
            return pos[board_num]->pieces(c, pt); 
        }

        /**
         * @brief Returns a bitboard representing all pieces of a specific color.
         * @param board_num The board index.
         * @param c The color of the pieces.
         * @return Stockfish::Bitboard Bitboard for the pieces.
         */
        Stockfish::Bitboard pieces(int board_num, Stockfish::Color c) {
            return pos[board_num]->pieces(c); 
        }

        /**
         * @brief Returns the en passant square for the specified board.
         * @param board_num The board index.
         * @return Stockfish::Square The en passant square.
         */
        Stockfish::Square ep_square(int board_num) {
            return pos[board_num]->ep_square();
        }

        /**
         * @brief Returns a bitboard of promoted pieces.
         * @param board_num The board index.
         * @return Stockfish::Bitboard Bitboard for promoted pieces.
         */
        Stockfish::Bitboard promoted_pieces(int board_num) { 
            return pos[board_num]->promotedPieces; 
        }

        /**
         * @brief Returns the current game ply.
         * @param board_num The board index.
         * @return int The game ply count.
         */
        int game_ply(int board_num) { 
            return pos[board_num]->game_ply(); 
        }

        /**
         * @brief Checks if castling is possible given the castling rights.
         * @param board_num The board index.
         * @param cr The castling rights.
         * @return true if castling is possible, false otherwise.
         */
        bool can_castle(int board_num, Stockfish::CastlingRights cr) { 
            return pos[board_num]->can_castle(cr); 
        }

        /**
         * @brief Returns the count for the fifty-move rule.
         * @param board_num The board index.
         * @return int The fifty-move rule counter.
         */
        int rule50_count(int board_num) { 
            return pos[board_num]->rule50_count(); 
        }

        /**
         * @brief Converts a move to a UCI string.
         * @param board_num The board index.
         * @param move The move to convert.
         * @return std::string The UCI move string.
         */
        std::string uci_move(int board_num, Stockfish::Move move) { 
            if (move == Stockfish::MOVE_NONE) {
                return "pass";
            }

            // Get the UCI move string
            std::string move_str = Stockfish::UCI::move(*pos[board_num], move).c_str();

            // Concatenate board_num and move_str without a space
            return move_str;
        }

        /**
         * @brief Converts a move to SAN (Standard Algebraic Notation) string.
         * @param board_num The board index.
         * @param move The move to convert.
         * @return std::string The SAN move string (e.g., "e4", "Nf3", "O-O").
         */
        std::string san_move(int board_num, Stockfish::Move move) { 
            if (move == Stockfish::MOVE_NONE) {
                return "pass";
            }

            // Get the SAN move string using Fairy-Stockfish's move_to_san
            return Stockfish::SAN::move_to_san(*pos[board_num], move, Stockfish::NOTATION_SAN);
        }

        bool is_checkmate(Stockfish::Color side, bool teamHasTimeAdvantage = false);
        bool check_mate_in_one(Stockfish::Color side);
        
        /**
         * @brief Checks if partner can capture a piece that could block a check.
         * Used for proper bughouse checkmate detection.
         * @param board_in_check The board index where the player is in check (0 or 1)
         * @param checked_side The color of the player being checked on that board
         * @param teamHasTimeAdvantage If true, partner may capture in the future even if not their turn
         * @return true if partner can provide a blocking piece, false otherwise
         */
        bool can_partner_provide_blocking_piece(int board_in_check, Stockfish::Color checked_side, bool teamHasTimeAdvantage = false);

        bool is_in_check(int board_num) {
            return pos[board_num]->checkers();
        }

        /**
         * @brief Check if a SPECIFIC color's king is being attacked on a board.
         * This checks if the given color's king is in check, regardless of whose turn it is.
         * @param board_num The board index (0 or 1)
         * @param color The color whose king to check
         * @return true if that color's king is being attacked by opponent pieces
         */
        bool is_king_attacked(int board_num, Stockfish::Color color) {
            Stockfish::Square kingSquare = pos[board_num]->square<Stockfish::KING>(color);
            if (kingSquare == Stockfish::SQ_NONE) {
                return false;  // King not on board (shouldn't happen in normal chess)
            }
            // Check if any opponent pieces attack the king's square
            Stockfish::Bitboard attackers = pos[board_num]->attackers_to(kingSquare, ~color);
            return attackers != 0;
        }

        /**
         * @brief Check if either board is in a draw state (3-fold repetition ignoring pocket pieces).
         * @param ply The current search depth (used for repetition detection)
         *            When ply > 0, 2-fold repetition within search is treated as draw.
         *            When ply = 0, requires 3-fold repetition.
         * @return true if either board has reached a draw condition.
         */
        bool is_draw(int ply = 0) {
            return is_draw_on_board(0, ply) || is_draw_on_board(1, ply); 
        }

        /**
         * @brief Check if a specific board is in a draw state (3-fold repetition ignoring pocket pieces).
         * @param board_num The board index.
         * @param ply The current search depth (used for repetition detection)
         *            When ply > 0, 2-fold repetition within search is treated as draw.
         *            When ply = 0, requires 3-fold repetition.
         * @return true if the board has reached a draw condition.
         */
        bool is_draw_on_board(int board_num, int ply = 0) {
            // Check 50-move rule using Fairy-Stockfish's built-in detection
            if (pos[board_num]->rule50_count() >= 100) {
                return true;
            }
            
            // Check for 3-fold (or 2-fold in search) repetition ignoring pocket pieces
            uint64_t currentKey = board_only_key(board_num);
            const auto& history = positionHistory[board_num];
            
            // Count how many times this position has occurred in PREVIOUS positions
            // (excluding the current position which is the last entry in history)
            int repetitionCount = 0;
            
            // When ply > 0, we're in search - 2-fold repetition is a draw
            // When ply = 0, we need 3-fold repetition (current + 2 previous = 3 occurrences)
            int threshold = (ply > 0) ? 1 : 2;
            
            // Check all positions except the last one (current position)
            size_t historySize = history.size();
            for (size_t i = 0; i + 1 < historySize; ++i) {
                if (history[i] == currentKey) {
                    repetitionCount++;
                    if (repetitionCount >= threshold) {
                        return true;
                    }
                }
            }
            
            return false;
        }
};
