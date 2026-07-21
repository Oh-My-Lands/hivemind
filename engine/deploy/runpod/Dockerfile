# Hivemind Bughouse Engine - RunPod Serverless Deployment
# Based on NVIDIA TensorRT container for GPU inference
# Using 25.01 for TensorRT 10.8 (closest to local 10.14)

FROM nvcr.io/nvidia/tensorrt:25.01-py3

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for the handler
RUN pip install --no-cache-dir \
    runpod \
    requests

# Copy engine source code
COPY src/ /app/src/
COPY cmake/ /app/cmake/
COPY tests/ /app/tests/
COPY CMakeLists.txt /app/

# Build the engine.
#
# -DHIVEMIND_ARCH: CMakeLists defaults -march to native, which targets the build
# host's CPU and SIGILLs on any worker with a narrower instruction set. See the
# comment on HIVEMIND_ARCH in CMakeLists.txt.
RUN mkdir -p build && cd build && \
    cmake .. -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DHIVEMIND_ARCH=x86-64-v3 \
        -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" && \
    ninja

# Copy the network model, and any prebuilt TensorRT plan sitting beside it.
#
# Baking a plan is close to mandatory for serverless. Measured on an RTX 4090:
# starting from a usable plan takes 0.6s, building one from the ONNX takes 236s.
# A cold worker without a plan will blow the job timeout on its first request.
#
# Plans are tied to a GPU architecture and TensorRT major version, and are named
# accordingly (..._sm89_trt10_v1.engine). A plan for the wrong hardware simply
# misses the cache and is rebuilt, so shipping one is safe -- but only pays off
# if the endpoint is pinned to the GPU type it was built for. Build it by
# running the engine once on that GPU, then drop the .engine into networks/
# before docker build.
RUN mkdir -p /app/networks
COPY networks/ /app/networks/

# Copy the RunPod handler. uci_parse.py sits beside it as a flat module, so it
# must be copied too -- handler.py imports it at module scope and the container
# fails to start without it.
COPY deploy/runpod/handler.py /app/handler.py
COPY deploy/runpod/uci_parse.py /app/uci_parse.py

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ENGINE_PATH=/app/build/hivemind
ENV NETWORKS_DIR=/app/networks

# RunPod serverless entry point
CMD ["python", "-u", "handler.py"]
