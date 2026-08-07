import argparse
import hashlib
import os
import chess
import numpy as np
import polars as pl
from tqdm import tqdm
import uuid

from src.domain.board import BughouseBoard
from src.domain.board2planes import board2planes
from src.domain.move2planes import mirrorMoveUCI, make_map
from src.domain.time_encoding import (TIME_PLANE_CHANNEL_A,
                                      TIME_PLANE_CHANNEL_B, TimeEncoding,
                                      quantize_planes)
from src.utils.game_reader import TrainingGameReader, process_parquet_file


class ShardWriter:
    def __init__(self, output_dir, samples_per_shard=2 ** 16, shard_name=None):
        self.output_dir = output_dir
        self.samples_per_shard = samples_per_shard
        self.shard_name = shard_name
        self.buffer = []
        os.makedirs(output_dir, exist_ok=True)

    def add_sample(self, x, policy_idx, value):
        # quantize_planes is the single write boundary: it rescales pockets out
        # of their count / MAX_NUM_DROPS form (a bare uint8 cast truncates 1/16
        # to 0 and zeroes twenty channels) and affine-maps the signed margin
        # planes (a bare cast wraps every negative one). data_loaders
        # .dequantize_planes is its exact inverse.
        self.buffer.append({
            "x": quantize_planes(np.asarray(x)).tobytes(),
            "y_policy_idx": (int(policy_idx[0]), int(policy_idx[1])),
            "y_value": float(value)
        })
        if len(self.buffer) >= self.samples_per_shard:
            self.write_shard()

    def write_shard(self):
        if not self.buffer: return
        name = self.shard_name or f"shard_{uuid.uuid4().hex[:8]}.parquet"
        save_path = os.path.join(self.output_dir, name)
        pl.DataFrame(self.buffer).write_parquet(save_path, compression="zstd")
        print(f"Saved {len(self.buffer)} samples to {save_path}")
        self.buffer = []


def is_validation_pair(pair_key, val_fraction=0.02):
    """Deterministically hold out a fraction of *games* for validation.

    Keyed on pair_key so both boards of a game land on the same side; splitting
    per row would leak board A into train while board B sits in val, and the two
    are the same game seen from two seats. Hashed rather than sampled so the
    split is reproducible across runs, and md5 rather than hash() because the
    latter is salted per process.
    """
    digest = hashlib.md5(pair_key.encode()).digest()
    return (int.from_bytes(digest[:4], "big") / 2 ** 32) < val_fraction


def generate_planes(samples_per_shard=2 ** 16, val_fraction=0.02,
                    time_encoding=TimeEncoding.BINARY, planes_dir=None,
                    max_games=None):
    labels = make_map()
    data_dir = 'data'
    games_path = os.path.join(data_dir, 'games.parquet')

    # Each Phase 2 arm gets its own directory. They must be generated from this
    # one code path, differing only in `time_encoding` -- training an arm on
    # separately-produced planes reintroduces exactly the corpus confound the
    # A/B exists to avoid (TIME_MODEL_PLAN.md, "The baseline trap").
    planes_dir = planes_dir or os.path.join(data_dir, 'planes')
    train_writer = ShardWriter(os.path.join(planes_dir, 'train'), samples_per_shard)

    # train_loop.train_supervised loads the validation set as a single file, so
    # it is buffered whole and written once at the end rather than sharded.
    val_writer = ShardWriter(os.path.join(planes_dir, 'val'),
                             samples_per_shard=2 ** 62,
                             shard_name='evaluation_shard.parquet')

    # 2200, not 2400, on purpose. The corpus is seeded at 2400 so every board
    # being analysed clears that bar, but this filter applies to all four
    # players -- raising it here would re-drop every pair whose partner board
    # sits between the two, which measurement showed is ~85% of them.
    game_gen = process_parquet_file(games_path, min_rating=2200)
    print(f"Starting plane generation... (time encoding {time_encoding.value}, "
          f"val fraction {val_fraction:.1%}, split on game)")
    print(f"Writing to {planes_dir}")

    n_train_games = n_val_games = 0

    for n_seen, reader in enumerate(tqdm(game_gen, desc='Processing games')):
        # A full pass is ~2 hours, so --max-games exists to make the encoding
        # arms comparable in seconds before committing to that. Applied before
        # the time_control filter so both arms consume the same prefix.
        if max_games is not None and n_seen >= max_games:
            break

        if reader.time_control == -1:
            continue

        if is_validation_pair(reader.pair_key, val_fraction):
            writer = val_writer
            n_val_games += 1
        else:
            writer = train_writer
            n_train_games += 1

        try:
            board = BughouseBoard(reader.time_control)

            # This holds the state and moves for the "current" team's turn
            current_action = {
                "team": None,  # 0 or 1
                "planes": None,  # Snapshot of board before any team moves
                "moves": [None, None]  # [board_0_move, board_1_move]
            }

            for board_num, move, time_left, move_time in reader.moves:
                board.update_time(board_num, time_left, move_time)
                side = board.boards[board_num].turn

                # Identify Team: Team 0 is (B0-White/B1-Black), Team 1 is (B0-Black/B1-White)
                # This works because partners always have opposite colors.
                moving_team = 0 if (board_num == 0 and side == chess.WHITE) or \
                                   (board_num == 1 and side == chess.BLACK) else 1

                # If the team changed, the previous team's opportunity to move is over. Flush it.
                if current_action["team"] is not None and moving_team != current_action["team"]:
                    save_team_action(writer, current_action, labels, reader.result)
                    current_action = {"team": None, "planes": None, "moves": [None, None]}

                # If this is the start of a team turn, snapshot the planes
                if current_action["planes"] is None:
                    # 'perspective_side' represents the perspective for board2planes
                    perspective_side = chess.WHITE if moving_team == 0 else chess.BLACK
                    current_action["planes"] = (
                        board2planes(board, perspective_side, time_encoding=time_encoding),
                        board2planes(board, not perspective_side, time_encoding=time_encoding))
                    current_action["team"] = moving_team

                # Canonicalize the move
                move_uci = move.uci()
                if side == chess.BLACK:
                    move_uci = mirrorMoveUCI(move_uci)
                if len(move_uci) == 5 and move_uci[-1] != 'n':
                    move_uci = move_uci[:-1]

                # Store the move in the buffer for the correct board
                current_action["moves"][board_num] = move_uci

                # Advance board state
                board.push(board_num, move)

            # Flush the final moves of the game
            if current_action["team"] is not None:
                save_team_action(writer, current_action, labels, reader.result)

        except Exception as e:
            print(f'Error processing game: {e}')

    train_writer.write_shard()
    val_writer.write_shard()
    print(f"Split on game: {n_train_games} train games, {n_val_games} val games "
          f"({n_val_games / max(n_train_games + n_val_games, 1):.2%})")


def save_team_action(writer, action, labels, game_result):
    """Utility to format and write the buffered team moves."""
    # Convert moves to labels, defaulting to 'pass' if a board didn't move
    m0 = action["moves"][0] if action["moves"][0] else 'pass'
    m1 = action["moves"][1] if action["moves"][1] else 'pass'

    assert m0 != "pass" or m1 != "pass", "Both boards didn't move!"

    policy_idx = (labels.index(m0), labels.index(m1))

    # Calculate value: If Team 0 moved, they represent the "Board 0 White" perspective
    # If Team 1 moved, they represent the "Board 0 Black" perspective
    value = game_result if action["team"] == 0 else -game_result

    # Time advantage is per board: each member races their *diagonal* opponent,
    # so board A's margin (31) says nothing about board B's (63 = 31 + 32).
    # Gating both boards on 31 was correct only while the two planes held the
    # same value, which stopped being true in 92774eb.
    #
    # The threshold is > 0.0, not > 0.5, so it reads the same under both
    # encodings: BINARY stores 1.0 exactly when CONTINUOUS stores a positive
    # squashed margin. That is what holds training-set composition fixed across
    # the two arms, leaving the encoding as the only difference between them.
    board_a_time_advantage = action["planes"][0][TIME_PLANE_CHANNEL_A, 0, 0] > 0.0
    board_b_time_advantage = action["planes"][0][TIME_PLANE_CHANNEL_B, 0, 0] > 0.0

    # Check if boards are on turn (channels 25 and 57)
    board_a_on_turn = action["planes"][0][25, 0, 0] > 0.5  # Board A turn plane
    board_b_on_turn = action["planes"][0][57, 0, 0] > 0.5  # Board B turn plane (channel 57 = 25 + 32)

    # Skip sample if a board passes while on turn without the time to afford it,
    # judged against that board's own margin.
    if (m0 == 'pass' and board_a_on_turn and not board_a_time_advantage) or \
       (m1 == 'pass' and board_b_on_turn and not board_b_time_advantage):
        return  # Don't add this sample

    writer.add_sample(action["planes"][0], policy_idx, value)

    # Create full pass move for other team to teach network how to sit and use time
    # From the other's team perspective it could be that both board are not on turn so both have to pass
    # Or the case we care more about: only one board is on turn and both boards still pass
    if 'pass' in [m0, m1]:
        # For the other team's sample, check their time advantage per board.
        # These are NOT duplicates of each other -- see the note above.
        other_board_a_time_advantage = action["planes"][1][TIME_PLANE_CHANNEL_A, 0, 0] > 0.0
        other_board_b_time_advantage = action["planes"][1][TIME_PLANE_CHANNEL_B, 0, 0] > 0.0

        # For other team's perspective, the turn channels are different
        other_board_a_on_turn = action["planes"][1][25, 0, 0] > 0.5
        other_board_b_on_turn = action["planes"][1][57, 0, 0] > 0.5

        # Skip other team's sample if either board would sit while on turn
        # without the time to afford it, judged per board.
        if (other_board_a_on_turn and not other_board_a_time_advantage) or \
           (other_board_b_on_turn and not other_board_b_time_advantage):
            return  # Don't add the other team's sample either
        # Skip if both boards are on turn since it doesn't make any sense to double sit even if up time
        if other_board_a_on_turn and other_board_b_on_turn:
            return

        writer.add_sample(action["planes"][1], (labels.index('pass'), labels.index('pass')), -value)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate training planes from data/games.parquet.")
    parser.add_argument(
        '--time-encoding', choices=[e.value for e in TimeEncoding],
        default=TimeEncoding.BINARY.value,
        help="Sit-margin representation in channels 31/63. 'binary' is arm A "
             "(the deployed network's encoding), 'continuous' is arm B.")
    parser.add_argument(
        '--planes-dir', default=None,
        help="Output directory, containing train/ and val/. Defaults to "
             "data/planes. Give each arm its own so they do not overwrite.")
    parser.add_argument('--val-fraction', type=float, default=0.02)
    parser.add_argument(
        '--max-games', type=int, default=None,
        help="Stop after this many game readers. For smoke-testing both arms "
             "cheaply before committing to a full ~2 h pass.")
    args = parser.parse_args()

    generate_planes(val_fraction=args.val_fraction,
                    time_encoding=TimeEncoding(args.time_encoding),
                    planes_dir=args.planes_dir,
                    max_games=args.max_games)
