#!/bin/bash
# Build and push the Hivemind engine image, then print the digest to deploy.
#
# Usage: ./build_and_push.sh [tag]
#
# The tag defaults to the git SHA (suffixed -dirty when the tree has uncommitted
# changes). Deploy the *digest* this prints, not the tag -- see below.

set -euo pipefail

REGISTRY="ghcr.io"
NAMESPACE="oh-my-lands"
IMAGE_NAME="hivemind-engine"
DOCKERFILE="deploy/runpod/Dockerfile.slim"

# Everything is relative to the engine root, which is the build context: the
# Dockerfile COPYs src/, cmake/, tests/, networks/ and deploy/runpod/ from here.
cd "$(dirname "$0")/../.."

# podman, not docker. The Dockerfile fully qualifies its base images because
# podman enforces short-name resolution and cannot prompt in a non-interactive
# build.
ENGINE_CMD="${ENGINE_CMD:-podman}"
command -v "$ENGINE_CMD" >/dev/null || {
    echo "error: $ENGINE_CMD not found. Set ENGINE_CMD=docker to override." >&2
    exit 1
}

# ---------------------------------------------------------------------------
# Tag
# ---------------------------------------------------------------------------
#
# Never reuse a tag. A previous :v1 was pushed over in place and RunPod workers
# went on serving a cached copy of the old one -- the endpoint kept failing with
# an error the new image could not produce, which read as "the fix didn't work"
# rather than "you are not running the fix". Unique tags make that impossible to
# misread, and deploying by digest makes it impossible to happen.
if [ $# -ge 1 ]; then
    TAG="$1"
elif git rev-parse --git-dir >/dev/null 2>&1; then
    TAG="$(git rev-parse --short HEAD)"
    [ -n "$(git status --porcelain)" ] && TAG="${TAG}-dirty"
else
    TAG="$(date -u +%Y%m%d-%H%M%S)"
fi

FULL_IMAGE="${REGISTRY}/${NAMESPACE}/${IMAGE_NAME}:${TAG}"

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
#
# networks/ is gitignored -- it holds a 61 MB ONNX and a 32 MB plan -- so a
# fresh clone builds an image that works in every local test and then times out
# on its first cold worker, because the plan it needs is not in it. Fail here
# instead, where the cause is obvious.
[ -d networks ] || { echo "error: networks/ missing (gitignored; fetch it)" >&2; exit 1; }

shopt -s nullglob
onnx=(networks/*.onnx)
plans=(networks/*.engine)
shopt -u nullglob

[ ${#onnx[@]} -gt 0 ] || { echo "error: no .onnx in networks/" >&2; exit 1; }
[ ${#plans[@]} -gt 0 ] || {
    echo "error: no prebuilt .engine plan in networks/." >&2
    echo "       Building one from ONNX takes ~236s and blows the job timeout" >&2
    echo "       on a cold worker. Bake the plan before shipping." >&2
    exit 1
}

# A plan is bound to one GPU architecture. Serving sm89 (4090 / L40S / L4 /
# RTX 2000 Ada) needs an sm89 plan; anything else misses the cache and tries to
# rebuild. Warn rather than fail -- the endpoint's GPU pinning is what decides
# this, and it is not knowable from here.
case "${plans[0]}" in
    *_sm89_*) ;;
    *) echo "warning: ${plans[0]} is not an sm89 plan; confirm it matches the endpoint's GPU" >&2 ;;
esac

echo "Building ${FULL_IMAGE}"
echo "  dockerfile: ${DOCKERFILE}"
echo "  plan:       ${plans[0]##*/}"
echo

"$ENGINE_CMD" build -f "$DOCKERFILE" -t "$FULL_IMAGE" .

# ---------------------------------------------------------------------------
# Verify before pushing
# ---------------------------------------------------------------------------
#
# These are cheap and each one corresponds to a failure that reached production.
echo
echo "Verifying image..."

# /usr/local/cuda/compat ahead of the host driver makes every CUDA call fail
# with error 803 on any non-data-center GPU. See the comment on LD_LIBRARY_PATH
# in the Dockerfile.
ld_path="$("$ENGINE_CMD" inspect "$FULL_IMAGE" --format '{{range .Config.Env}}{{println .}}{{end}}' | grep '^LD_LIBRARY_PATH=' || true)"
case "$ld_path" in
    *cuda/compat*)
        echo "error: LD_LIBRARY_PATH contains cuda/compat -- this image will fail" >&2
        echo "       with CUDA error 803 on any non-data-center GPU." >&2
        echo "       got: $ld_path" >&2
        exit 1
        ;;
esac
echo "  ok: ${ld_path:-LD_LIBRARY_PATH unset (inherits base)}"

# The plan and the ONNX must both be present: getEnginePath() derives the plan
# filename from the ONNX stem, so shipping the plan alone fails at startup with
# a confusing complaint about no .onnx file.
"$ENGINE_CMD" run --rm --entrypoint "" "$FULL_IMAGE" \
    sh -c 'ls /app/networks/*.engine /app/networks/*.onnx /app/build/hivemind >/dev/null' \
    || { echo "error: image is missing the plan, the ONNX, or the binary" >&2; exit 1; }
echo "  ok: plan, onnx and binary present"

# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------
echo
echo "Pushing ${FULL_IMAGE}"

digestfile="$(mktemp)"
trap 'rm -f "$digestfile"' EXIT
"$ENGINE_CMD" push --digestfile "$digestfile" "$FULL_IMAGE"
DIGEST="$(cat "$digestfile")"

cat <<EOF

Pushed.

  tag:    ${FULL_IMAGE}
  digest: ${DIGEST}

Deploy this, not the tag:

  ${REGISTRY}/${NAMESPACE}/${IMAGE_NAME}@${DIGEST}

RunPod console -> Serverless -> your endpoint -> Edit Endpoint ->
Container Image. Saving recycles the workers, which is what makes them pull.

A tag is mutable and workers cache it, so pushing over one does not reliably
redeploy anything; a digest names exactly these bytes. After it comes up, run a
real search -- a worker reporting "ready" says nothing about whether the engine
loaded, and RunPod will report a worker healthy with an unusable GPU.
EOF
