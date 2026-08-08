#!/usr/bin/env bash
#
# Bring up a rented GPU pod and train both Phase 2 arms, sequentially.
#
# Run this ON the pod, after the planes are in place. It assumes nothing about
# the local machine except that the repo and data arrived.
#
#   ./scripts/train_pod_bootstrap.sh /workspace/data
#
# Why sequential: the two arms are a controlled comparison, and running them
# concurrently on one card halves each one's throughput while making the timing
# numbers useless for the next capacity estimate. There is no accuracy reason.
#
# Cost, at the rates measured 2026-08-07: 14,639,343 samples x 7 epochs at
# ~4,000 samples/s is ~7.1 h per arm, ~14.2 h for both. Secure 4090 is
# $0.74/hr, community $0.34/hr. CHECK THE BALANCE COVERS IT BEFORE STARTING --
# a pod that dies of an empty account mid-arm has spent the money and produced
# nothing for that arm.
set -euo pipefail

DATA_ROOT="${1:-/workspace/data}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ARMS=(planes-binary planes-continuous)

echo "=== preflight ==="
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; \
print('torch  ', torch.__version__); print('gpu    ', torch.cuda.get_device_name(0))"

# The export bug this branch fixed only shows up at the first checkpoint, which
# is an hour in. Prove the exporter works against a random-init net first --
# it needs no trained weights, so it costs seconds rather than an hour.
#
# "A file appeared" is not the check. The first version of this preflight
# passed on a model whose batch dimension onnxsim had frozen to 1, which loads
# and validates and gives correct single-position answers, and which the engine
# then cannot build a batched TensorRT profile against -- discovered after both
# arms had been trained and paid for. So the shape is asserted, not the file.
echo "=== onnx export smoke test ==="
python - <<'PY'
import tempfile, torch, onnx
from pathlib import Path
from src.training.train_loop import get_model_args
from src.architectures.rise_mobile_v3 import get_rise_v33_model
from src.training.trainer_agent import export_to_onnx
with tempfile.TemporaryDirectory() as d:
    export_to_onnx(get_rise_v33_model(get_model_args()), 1,
                   torch.zeros(1, 64, 8, 8), Path(d), "preflight", False, True)
    produced = sorted(p.name for p in Path(d).iterdir())
    onnx_files = [p for p in produced if p.endswith(".onnx")]
    assert onnx_files, produced

    model = onnx.load(str(Path(d) / onnx_files[0]))
    shape = model.graph.input[0].type.tensor_type.shape.dim
    dims = [d_.dim_param or d_.dim_value for d_ in shape]
    assert shape[0].WhichOneof("value") == "dim_param", \
        f"batch dimension is static {dims}; the engine cannot batch this model"
    print("onnx export OK:", onnx_files[0], dims)
PY

for arm in "${ARMS[@]}"; do
    dir="$DATA_ROOT/$arm"
    for sub in train val; do
        [ -d "$dir/$sub" ] || { echo "MISSING $dir/$sub" >&2; exit 1; }
    done
    n=$(find "$dir/train" -name '*.parquet' | wc -l)
    echo "$arm: $n train shards"
    [ "$n" -gt 0 ] || { echo "no shards in $dir/train" >&2; exit 1; }
done

# Both arms must see the same number of shards. Different counts mean the two
# datasets are not the matched pair the comparison assumes, and every Elo
# number that follows would be measuring the corpus as well as the encoding.
counts=$(for arm in "${ARMS[@]}"; do
             find "$DATA_ROOT/$arm/train" -name '*.parquet' | wc -l
         done | sort -u | wc -l)
[ "$counts" -eq 1 ] || { echo "arms have different shard counts -- not a matched pair" >&2; exit 1; }

for arm in "${ARMS[@]}"; do
    echo
    echo "=== training $arm ==="
    date -u +"start %Y-%m-%dT%H:%M:%SZ"
    python -m src.training.train_loop --mode sl --planes-dir "$DATA_ROOT/$arm" \
        2>&1 | tee "logs-$arm.txt"
    date -u +"end   %Y-%m-%dT%H:%M:%SZ"
done

echo
echo "=== artefacts ==="
find src/training/runs -name '*.onnx' -o -name '*.tar' | sort
echo
echo "Pull these before terminating the pod -- the volume goes with it."
