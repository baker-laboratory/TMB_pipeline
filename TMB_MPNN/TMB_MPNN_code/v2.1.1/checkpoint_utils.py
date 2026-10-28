"""Portable checkpoint paths for the supported TMB_MPNN runner."""

import os
from pathlib import Path


SC_CHECKPOINT_NAME = "ligandmpnn_sc_v_32_002_16.pt"
PIPELINE_ROOT = Path(__file__).resolve().parents[3]


def default_sc_checkpoint():
    """Prefer an explicit environment override, then the local weights directory."""
    override = os.environ.get("TMB_MPNN_SC_CKPT_PATH")
    if override:
        return str(Path(override).expanduser())
    weights_dir = Path(os.environ.get("WEIGHTS_DIR") or PIPELINE_ROOT / "weights")
    return str(weights_dir.expanduser() / SC_CHECKPOINT_NAME)


def require_sc_checkpoint(path):
    """Fail before loading the sequence model if packing weights are missing."""
    checkpoint = Path(path).expanduser()
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError(
            f"Side-chain packing checkpoint is missing or empty: {checkpoint}. "
            f"Run {PIPELINE_ROOT / 'download_assets.sh'} to download the public "
            "LigandMPNN weights, or set --checkpoint_path_sc (or "
            "TMB_MPNN_SC_CKPT_PATH) to an existing checkpoint. "
            "Set WEIGHTS_DIR when using a custom download directory."
        )
    return str(checkpoint)
