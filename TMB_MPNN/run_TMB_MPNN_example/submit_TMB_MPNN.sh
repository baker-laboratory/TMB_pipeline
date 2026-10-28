#!/bin/bash
set -e

TMB_PIPELINE_DIR="${TMB_PIPELINE_DIR:-$HOME/TMB_pipeline}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$TMB_PIPELINE_DIR/weights}"
ENV_DIR="${ENV_DIR:-$TMB_PIPELINE_DIR/environments}"

TMB_MPNN_APPTAINER_PATH="${TMB_MPNN_APPTAINER_PATH:-$ENV_DIR/TMB_MPNN.sif}"
TMB_MPNN_CKPT_PATH="${TMB_MPNN_CKPT_PATH:-$WEIGHTS_DIR/TMB_MPNN.pth}"
TMB_MPNN_SC_CKPT_PATH="${TMB_MPNN_SC_CKPT_PATH:-$WEIGHTS_DIR/ligandmpnn_sc_v_32_002_16.pt}"

echo "# dummy cv2 for headless TMB_MPNN run" > cv2.py

apptainer exec --nv --env PYTHONPATH="$PWD" "$TMB_MPNN_APPTAINER_PATH" \
python "$TMB_PIPELINE_DIR/TMB_MPNN/TMB_MPNN_code/v2.1.1/run.py" \
    --checkpoint "$TMB_MPNN_CKPT_PATH" \
    --model_type "$TMB_PIPELINE_DIR/TMB_MPNN/TMB_MPNN_code/v2.1.1/mpnn/configs/big_tmbb_80p_surf_condition.py" \
    --pdb_path_multi ./pdbs_multi_json.jsonl \
    --out_folder ./TMB_MPNN_sequences \
    --batch_size 5 \
    --number_of_batches 2 \
    --pack_side_chains 1 \
    --checkpoint_path_sc "$TMB_MPNN_SC_CKPT_PATH" \
    --threonine_serine_threshold 0.25 \
    --serthre_bias -0.5 \
    --tmbb_surface_condition_multi ./intf_buried.jsonl
