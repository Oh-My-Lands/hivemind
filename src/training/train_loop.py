import argparse
import glob
import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import polars as pl
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from configs.train_config import TrainConfig, TrainObjects, rl_train_config
from configs.main_config import main_config
from src.training.data_loaders import load_parquet_shard, load_rl_data_from_directory, load_rl_parquet_shard, RLDataset, StreamingRLDataset, CombinedRLDataset

from src.training.lr_schedules.lr_schedules import *
from src.architectures.rise_mobile_v3 import get_rise_v33_model
from src.training.trainer_agent import TrainerAgentPytorch, save_torch_state,\
    load_torch_state, export_to_onnx, get_context, get_data_loader, evaluate_metrics
from src.training.train_util import get_metrics, value_to_wdl_label, prepare_plys_label


def prepare_export_dir(tc, export_dir):
    """Point `tc.export_dir` at `export_dir` and create what training writes into.

    Two hazards, both of which have bitten:

    - `weights/` is never created by anything downstream, so the first
      checkpoint dies on a missing directory -- an hour into a rented pod.
    - `trainer_agent` addresses that directory two different ways:
      `Path(export_dir) / "weights"` in some places and the string
      `export_dir + "weights/..."` in others. The string form needs a trailing
      separator, and `str(Path(...))` strips it, so the two forms silently
      disagree unless it is put back here.
    """
    export_dir = os.path.join(str(export_dir), '')  # guarantee a trailing sep
    tc.export_dir = export_dir
    for sub in ('weights', 'logs'):
        Path(export_dir, sub).mkdir(parents=True, exist_ok=True)
    return export_dir


def get_model_args():
    """Get model configuration arguments."""
    class Args:
        def __init__(self):
            self.model_type = "risev33"
            self.input_version = "1.0"
            self.export_dir = "../../checkpoints"
            self.device_id = 0
            self.context = "gpu"
            self.input_shape = (64, 8, 8)
            self.n_labels = 0
            self.channels_policy_head = 73
            self.select_policy_from_plane = True
            self.use_wdl = False
            self.use_plys_to_end = False
            self.use_mlp_wdl_ply = False
    return Args()


def select_planes(planes_dir):
    """Point the config at one arm's planes, train and val together.

    `main_config` is read directly in two places -- here and
    `TrainerAgentPytorch.__init__` -- so the override has to land on the dict
    rather than being threaded through as an argument.

    Both keys move together on purpose. Training arm A's shards against arm B's
    validation set would run to completion and report plausible numbers while
    measuring nothing, and it is two independent paths that have to agree for
    that not to happen.
    """
    planes_dir = Path(planes_dir).resolve()
    train_dir, val_dir = planes_dir / "train", planes_dir / "val"

    for name, d in (("train", train_dir), ("val", val_dir)):
        if not d.is_dir():
            raise ValueError(
                f"{planes_dir} has no {name}/ directory. Generate it with "
                f"src/preprocessing/generate_planes.py --planes-dir {planes_dir}")

    main_config['planes_train_dir'] = os.path.join(str(train_dir), '')
    main_config['planes_val_dir'] = os.path.join(str(val_dir), '')
    return planes_dir


def train_supervised(planes_dir=None, export_dir=None):
    """Run supervised learning training on human game data."""
    tc = TrainConfig()
    to = TrainObjects()
    to.metrics = get_metrics(tc)

    if planes_dir is not None:
        planes_dir = select_planes(planes_dir)

    # Derived from the planes directory rather than fixed, so the two Phase 2
    # arms cannot collide. They would otherwise share src/training/weights/, and
    # delete_previous_weights() removes *every* file there -- so training arm B
    # after arm A would destroy the baseline it is meant to be compared against,
    # silently and after the pod time had already been paid for.
    if export_dir is None:
        run_name = planes_dir.name if planes_dir is not None else "default"
        export_dir = project_root / "src" / "training" / "runs" / run_name
    prepare_export_dir(tc, export_dir)

    tc.nb_parts = len(glob.glob(main_config['planes_train_dir'] + '*.parquet'))
    if tc.nb_parts == 0:
        raise ValueError(
            f"No training shards in {main_config['planes_train_dir']}. "
            "Run src/preprocessing/generate_planes.py first.")

    print(f"train shards  {tc.nb_parts} from {main_config['planes_train_dir']}")
    print(f"val set       {main_config['planes_val_dir']}evaluation_shard.parquet")
    print(f"exporting to  {tc.export_dir}")

    # Load validation data
    x_val, y_val_value, y_val_policy = load_parquet_shard(
        main_config['planes_val_dir'] + 'evaluation_shard.parquet')
    dataset = TensorDataset(x_val, y_val_value, y_val_policy)
    val_data = DataLoader(dataset, batch_size=tc.batch_size, shuffle=False)

    nb_it_per_epoch = (2**16 * tc.nb_parts) // tc.batch_size
    tc.total_it = int(nb_it_per_epoch * tc.nb_training_epochs)

    to.lr_schedule = OneCycleSchedule(start_lr=tc.max_lr / 8, max_lr=tc.max_lr, cycle_length=tc.total_it * .3,
                                      cooldown_length=tc.total_it * .6, finish_lr=tc.min_lr)
    to.lr_schedule = LinearWarmUp(to.lr_schedule, start_lr=tc.min_lr, length=tc.total_it / 30)
    to.momentum_schedule = MomentumSchedule(to.lr_schedule, tc.min_lr, tc.max_lr, tc.min_momentum, tc.max_momentum)

    args = get_model_args()
    model = get_rise_v33_model(args)

    trainer = TrainerAgentPytorch(model, val_data, tc, to, use_rtpt=True, is_rl=False)
    trainer.train()


def train_rl(rl_data_dir: str, val_data_dir: str, checkpoint_path: str = None, augment_flip: bool = True):
    """
    Run RL training on self-play data.
    
    Args:
        rl_data_dir: Directory containing RL parquet files (converted from binary)
        val_data_dir: Directory containing validation parquet files
        checkpoint_path: Optional path to load model weights from
        augment_flip: If True, use board flip augmentation to double training data
    """
    import glob
    from pathlib import Path as PathLib
    
    tc = rl_train_config()
    to = TrainObjects()
    to.metrics = get_metrics(tc)
    
    # Set export directory and ensure it exists. Via the helper so the trailing
    # separator survives -- trainer_agent builds some of these paths by string
    # concatenation, and str(Path(...)) drops it.
    weights_dir = Path(prepare_export_dir(tc, project_root / "src" / "training"), "weights")
    
    # Ensure ONNX export is enabled
    tc.export_weights = True
    
    # Find all training parquet files
    parquet_files = sorted(glob.glob(str(PathLib(rl_data_dir) / "*.parquet")))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {rl_data_dir}")
    
    print(f"Found {len(parquet_files)} training parquet files in {rl_data_dir}")
    
    # Load validation data from separate directory
    val_parquet_files = sorted(glob.glob(str(PathLib(val_data_dir) / "*.parquet")))
    if not val_parquet_files:
        raise ValueError(f"No parquet files found in validation directory {val_data_dir}")
    
    print(f"Found {len(val_parquet_files)} validation parquet files in {val_data_dir}")
    print(f"Loading validation data...")
    
    val_samples = []
    for vf in val_parquet_files:
        x, y_val, pol_a, pol_b = load_rl_parquet_shard(vf)
        for i in range(len(x)):
            val_samples.append((x[i], y_val[i], pol_a[i], pol_b[i]))
    
    if not val_samples:
        raise ValueError(f"No validation samples found in {val_data_dir}")
    
    # Convert validation to tensors
    x_val = torch.stack([s[0] for s in val_samples])
    y_val = torch.stack([s[1] for s in val_samples])
    pol_a_val = torch.stack([s[2] for s in val_samples])
    pol_b_val = torch.stack([s[3] for s in val_samples])
    
    val_dataset = RLDataset(x_val, y_val, pol_a_val, pol_b_val, augment_flip=augment_flip)
    val_loader = DataLoader(val_dataset, batch_size=tc.batch_size, shuffle=False)
    print(f"Loaded {len(val_dataset)} validation samples")
    
    # Estimate training samples
    estimated_samples_per_shard = 16384  # Typical shard size
    estimated_training_samples = len(parquet_files) * estimated_samples_per_shard
    if augment_flip:
        estimated_training_samples *= 2  # Double with augmentation
    n_train = estimated_training_samples
    
    print(f"Board flip augmentation: {'ENABLED' if augment_flip else 'DISABLED'}")
    if augment_flip:
        print(f"Training data will be doubled through board flip augmentation")
    print(f"Estimated training samples: ~{n_train}")
    
    # Create streaming training dataset from all training files
    train_dataset = StreamingRLDataset(parquet_files, shuffle_files=True, shuffle_buffer_size=10000, augment_flip=augment_flip)
    train_loader = DataLoader(train_dataset, batch_size=tc.batch_size, num_workers=0)
    
    # Calculate iterations (approximate)
    nb_it_per_epoch = max(1, n_train // tc.batch_size)
    tc.total_it = int(nb_it_per_epoch * tc.nb_training_epochs)
    tc.nb_parts = len(parquet_files)
    
    print(f"Iterations per epoch: ~{nb_it_per_epoch}, Total iterations: ~{tc.total_it}")
    
    # LR schedule: Cosine Annealing with 25% warm-up (as per CrazyAra RL paper)
    # The warm-up helps with context drift in the training data
    cosine_schedule = CosineAnnealingSchedule(min_lr=tc.min_lr, max_lr=tc.max_lr, cycle_length=tc.total_it)
    to.lr_schedule = LinearWarmUp(cosine_schedule, start_lr=tc.min_lr, length=int(tc.total_it * 0.25))
    to.momentum_schedule = MomentumSchedule(to.lr_schedule, tc.min_lr, tc.max_lr, tc.min_momentum, tc.max_momentum)
    
    # Load model
    args = get_model_args()
    model = get_rise_v33_model(args)
    
    # Optionally load checkpoint
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Train with streaming RL data loader
    trainer = TrainerAgentPytorch(model, val_loader, tc, to, use_rtpt=True, 
                                  is_rl=True, rl_train_loader=train_loader)
    trainer.train()
    
    # Export final model to ONNX
    print("\nExporting final model to ONNX...")
    ctx = get_context(tc.context, tc.device_id)
    dummy_input = torch.zeros(1, 64, 8, 8).to(ctx)
    model_prefix = f"model-rl-final"
    export_to_onnx(model, 1, dummy_input, weights_dir, model_prefix, False, True)
    print(f"ONNX model exported to {weights_dir}/{model_prefix}.onnx")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Hivemind neural network')
    parser.add_argument('--mode', type=str, default='sl', choices=['sl', 'rl'],
                        help='Training mode: sl (supervised learning) or rl (reinforcement learning)')
    parser.add_argument('--rl-data-dir', type=str, default='../../engine/selfplay_games/training_data_parquet',
                        help='Directory containing RL training parquet files')
    parser.add_argument('--val-data-dir', type=str, default='/home/ben/hivemind/engine/selfplay_games/val_data_parquet',
                        help='Directory containing RL validation parquet files')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--planes-dir', type=str, default=None,
                        help="Which arm to train, e.g. data/planes-binary or "
                             "data/planes-continuous. Selects train/ and val/ "
                             "together, and gives the run its own export "
                             "directory so the two arms cannot overwrite each "
                             "other. Defaults to whatever main_config points at.")
    parser.add_argument('--export-dir', type=str, default=None,
                        help='Override where weights and logs are written.')

    args = parser.parse_args()

    if args.mode == 'sl':
        train_supervised(planes_dir=args.planes_dir, export_dir=args.export_dir)
    else:
        train_rl(args.rl_data_dir, args.val_data_dir, checkpoint_path=args.checkpoint)