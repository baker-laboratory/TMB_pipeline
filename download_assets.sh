#!/bin/bash
# Download the model weights (and optionally the Apptainer containers) that this
# pipeline needs. Safe to re-run: files that already exist are left alone, so an
# existing checkout only fetches what it is missing.
#
#   ./download_assets.sh              # weights only (~1.9 GB)
#   ./download_assets.sh --containers # weights and containers (~30 GB)

set -euo pipefail

TMB_PIPELINE_DIR="${TMB_PIPELINE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$TMB_PIPELINE_DIR/weights}"
ENV_DIR="${ENV_DIR:-$TMB_PIPELINE_DIR/environments}"

WANT_CONTAINERS=0
for arg in "$@"; do
    case "$arg" in
        --containers) WANT_CONTAINERS=1 ;;
        -h|--help) sed -n '2,8p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

TMB_BASE="https://files.ipd.uw.edu/pub/TMB_pipeline"
LIGANDMPNN_BASE="https://files.ipd.uw.edu/pub/ligandmpnn"

# The side-chain packer is the unmodified side-chain packing model from the
# public LigandMPNN release, so it comes from that project's own directory.
WEIGHT_URLS=(
    "$TMB_BASE/TMB_RFD2/TMB_RFD2.pt"
    "$TMB_BASE/TMB_RFD2/TMB_RFD2_motif_scaffolding.pt"
    "$TMB_BASE/TMB_MPNN/TMB_MPNN.pth"
    "$TMB_BASE/TMB_MPNN/TMB_MPNN_tuning_hydrophobic_thickness.pth"
    "$LIGANDMPNN_BASE/ligandmpnn_sc_v_32_002_16.pt"
)

CONTAINER_URLS=(
    "$TMB_BASE/sif_file/modulus_latest.sif"
    "$TMB_BASE/sif_file/TMB_RFD2.sif"
    "$TMB_BASE/sif_file/TMB_MPNN.sif"
)

fetch() {
    local url="$1" dest_dir="$2"
    local name dest
    name="$(basename "$url")"
    dest="$dest_dir/$name"

    if [[ -s "$dest" ]]; then
        echo "  have    $name"
        return 0
    fi

    echo "  fetch   $name"
    # Download to a temporary name so an interrupted transfer is never mistaken
    # for a complete file on the next run.
    if curl -fSL --retry 3 --retry-delay 5 -o "$dest.partial" "$url"; then
        mv "$dest.partial" "$dest"
    else
        rm -f "$dest.partial"
        echo "  FAILED  $name ($url)" >&2
        return 1
    fi
}

mkdir -p "$WEIGHTS_DIR" "$ENV_DIR"

failed=0

echo "Weights -> $WEIGHTS_DIR"
for url in "${WEIGHT_URLS[@]}"; do
    fetch "$url" "$WEIGHTS_DIR" || failed=1
done

if [[ "$WANT_CONTAINERS" -eq 1 ]]; then
    echo "Containers -> $ENV_DIR"
    for url in "${CONTAINER_URLS[@]}"; do
        fetch "$url" "$ENV_DIR" || failed=1
    done
else
    echo "Containers skipped (pass --containers to download them)."
fi

if [[ "$failed" -ne 0 ]]; then
    echo
    echo "Some downloads failed; re-run to retry just those." >&2
    exit 1
fi

echo
echo "Done. Now export the paths as described in the README, section 2 and 3."
