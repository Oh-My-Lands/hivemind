#!/bin/bash
# Fetch the network artifacts into engine/networks/.
#
# Usage:
#   ./fetch_networks.sh            download and verify
#   ./fetch_networks.sh verify     check what is already there, download nothing
#   ./fetch_networks.sh publish    upload the local files to create the release
#
# engine/networks/ is gitignored (engine/.gitignore:80) and correctly so -- it
# holds ~94 MB of binaries that git would store badly. The cost is that the
# image build depends on files no clone can obtain. This script is the missing
# half of that arrangement.

set -euo pipefail

REPO="${NETWORKS_REPO:-Oh-My-Lands/hivemind}"
RELEASE_TAG="${NETWORKS_RELEASE:-networks-v3.0}"

cd "$(dirname "$0")/.."
DEST="networks"

# ---------------------------------------------------------------------------
# What we expect, and why there are two kinds of file
# ---------------------------------------------------------------------------
#
# The ONNX is the model: portable, the same bytes everywhere, the source of
# truth. The .engine is a TensorRT *plan* compiled from it, and is not portable
# -- it is tied to a GPU architecture and a TensorRT version, both of which are
# encoded in the filename (..._sm89_trt10_v1.engine).
#
# Both must be present. getEnginePath() derives the plan's filename from the
# ONNX stem (src/onnx_utils.cc:29-30) and findLatestOnnxFile() scans this
# directory for a .onnx, throwing when there is none (neuralnetapi.cpp:65), so
# shipping the plan alone fails at startup complaining about a missing .onnx.
#
# Shipping the ONNX alone is worse, because it *works*: the engine rebuilds the
# plan from it, which takes ~236s. On a dev pod that is a slow first search and
# then it is cached. On a serverless worker it blows the job timeout, and does
# so again on every cold start, because nothing persists.
ARTIFACTS=(
    "model-0.97878-0.683-0224-v3.0.onnx 463f2fc8c2805bfb99456e756a76340b2dfbeddada4bbc2fb909516cb6d36d3b"
    "model-0.97878-0.683-0224-v3.0_fp16_b16_sm89_trt10_v1.engine f2eda17a89808019e6efcbcc652592708d864c9e5aec9665cb11a6903235c378"
)

command -v gh >/dev/null || { echo "error: gh CLI not found (needed; the repo may be private)" >&2; exit 1; }

verify_one() {
    # $1 filename, $2 expected sha256. Prints a status line, returns non-zero
    # when the file is missing or does not match.
    local name="$1" want="$2" path="$DEST/$1"
    [ -f "$path" ] || { echo "  missing  $name"; return 1; }
    local got
    got="$(sha256sum "$path" | cut -d' ' -f1)"
    if [ "$got" != "$want" ]; then
        # A truncated download is the common case and looks identical to
        # corruption from the engine's point of view: TensorRT rejects the plan
        # and falls back to a rebuild, i.e. the 236s timeout again.
        echo "  CORRUPT  $name (sha256 $got, expected $want)"
        return 1
    fi
    echo "  ok       $name"
    return 0
}

case "${1:-fetch}" in

verify)
    rc=0
    echo "Verifying $DEST/"
    for a in "${ARTIFACTS[@]}"; do verify_one ${a} || rc=1; done
    exit $rc
    ;;

publish)
    # Run once, from a machine that already has good files, to create the
    # release everyone else fetches from.
    echo "Publishing $DEST/* to ${REPO} release ${RELEASE_TAG}"
    for a in "${ARTIFACTS[@]}"; do
        verify_one ${a} || { echo "refusing to publish: local files are not the expected ones" >&2; exit 1; }
    done

    if ! gh release view "$RELEASE_TAG" --repo "$REPO" >/dev/null 2>&1; then
        gh release create "$RELEASE_TAG" --repo "$REPO" \
            --title "Networks $RELEASE_TAG" \
            --notes "ONNX model and prebuilt sm89 / TensorRT 10 plan for engine/networks/. See engine/deploy/fetch_networks.sh."
    fi

    for a in "${ARTIFACTS[@]}"; do
        set -- $a
        gh release upload "$RELEASE_TAG" "$DEST/$1" --repo "$REPO" --clobber
    done
    echo "Published."
    exit 0
    ;;

fetch) ;;
*) echo "usage: $0 [fetch|verify|publish]" >&2; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
mkdir -p "$DEST"

if ! gh release view "$RELEASE_TAG" --repo "$REPO" >/dev/null 2>&1; then
    cat >&2 <<EOF
error: release '${RELEASE_TAG}' does not exist on ${REPO}.

Nothing has published these artifacts yet -- this script is new and the release
is the half that still has to be created. From a machine that has a known-good
engine/networks/ (checksums are pinned in this script):

    $0 publish

Or set NETWORKS_REPO / NETWORKS_RELEASE to point somewhere else.
EOF
    exit 1
fi

need_any=0
for a in "${ARTIFACTS[@]}"; do
    set -- $a
    verify_one "$1" "$2" >/dev/null 2>&1 || need_any=1
done

if [ "$need_any" -eq 0 ]; then
    echo "networks/ is already complete and verified; nothing to do."
    exit 0
fi

for a in "${ARTIFACTS[@]}"; do
    set -- $a
    name="$1" want="$2"
    if verify_one "$name" "$want" >/dev/null 2>&1; then
        echo "  have     $name"
        continue
    fi
    echo "  fetching $name"
    gh release download "$RELEASE_TAG" --repo "$REPO" --pattern "$name" --dir "$DEST" --clobber
done

echo "Verifying"
rc=0
for a in "${ARTIFACTS[@]}"; do verify_one ${a} || rc=1; done
[ "$rc" -eq 0 ] || { echo "error: verification failed after download" >&2; exit 1; }

cat <<EOF

networks/ is ready.

The plan here is sm89 (4090 / L40S / L4 / RTX 2000 Ada) built against TensorRT
10. Running it on another architecture misses the cache and triggers a ~236s
rebuild rather than failing outright, so if you are targeting sm80 or sm90,
generate a plan for it instead of relying on this one.
EOF
