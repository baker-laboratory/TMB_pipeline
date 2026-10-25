# TMB_pipeline

This repository contains code for running a generative pipeline for transmembrane β-barrel (TMB) design. It includes scripts demonstrating how to use **TMB_RFD2** for backbone generation and **TMB_MPNN** for sequence design, as well as notebooks describing the curation and filtering of the distillation set used for training.

---

# Installation and setup

## 1. Clone the repository

```bash
git clone https://github.com/baker-laboratory/TMB_pipeline.git
cd TMB_pipeline
```

---

## 2. Set environment variables and create directories

The example commands in this repository use the following environment variables.

```bash
mkdir -p weights environments

export TMB_PIPELINE_DIR="$PWD"
export WEIGHTS_DIR="$TMB_PIPELINE_DIR/weights"
export ENV_DIR="$TMB_PIPELINE_DIR/environments"
```

You may change the directory names if desired, but make sure the exported paths match your chosen locations.

---

## 3. Download model weights

Navigate to the weights directory and download the checkpoints.

```bash
cd "$WEIGHTS_DIR"

# TMB_RFD2 checkpoint for standard backbone generation
curl -O https://files.ipd.uw.edu/pub/TMB_pipeline/TMB_RFD2/TMB_RFD2.pt

# TMB_RFD2 checkpoint for motif scaffolding
curl -O https://files.ipd.uw.edu/pub/TMB_pipeline/TMB_RFD2/TMB_RFD2_motif_scaffolding.pt

# TMB_MPNN checkpoint for standard sequence design
curl -O https://files.ipd.uw.edu/pub/TMB_pipeline/TMB_MPNN/TMB_MPNN.pth

# TMB_MPNN checkpoint for tuning hydrophobic thickness
curl -O https://files.ipd.uw.edu/pub/TMB_pipeline/TMB_MPNN/TMB_MPNN_tuning_hydrophobic_thickness.pth
```

Set checkpoint paths.

```bash
export TMB_RFD2_CKPT_PATH="$WEIGHTS_DIR/TMB_RFD2.pt"
export TMB_RFD2_MOTIF_SCAFFOLDING_CKPT_PATH="$WEIGHTS_DIR/TMB_RFD2_motif_scaffolding.pt"
export TMB_MPNN_CKPT_PATH="$WEIGHTS_DIR/TMB_MPNN.pth"
export TMB_MPNN_COND_CKPT_PATH="$WEIGHTS_DIR/TMB_MPNN_tuning_hydrophobic_thickness.pth"
```

---

## 4. Set up the environment: Singularity / Apptainer

Navigate to the environments directory and download the containers.

```bash
cd "$ENV_DIR"

curl -O https://files.ipd.uw.edu/pub/TMB_pipeline/sif_file/modulus_latest.sif
curl -O https://files.ipd.uw.edu/pub/TMB_pipeline/sif_file/TMB_RFD2.sif
curl -O https://files.ipd.uw.edu/pub/TMB_pipeline/sif_file/TMB_MPNN.sif
```

Set container paths.

```bash
export MODULUS_APPTAINER_PATH="$ENV_DIR/modulus_latest.sif"
export TMB_RFD2_APPTAINER_PATH="$ENV_DIR/TMB_RFD2.sif"
export TMB_MPNN_APPTAINER_PATH="$ENV_DIR/TMB_MPNN.sif"
```

The container variables are used as follows:

```bash
# General utilities / examples
apptainer exec "$MODULUS_APPTAINER_PATH" python --version

# TMB_RFD2 backbone generation
apptainer exec "$TMB_RFD2_APPTAINER_PATH" python --version

# TMB_MPNN sequence design
apptainer exec "$TMB_MPNN_APPTAINER_PATH" python --version
```

---

## 5. Verify setup

Check that all paths are set correctly.

```bash
echo "$TMB_PIPELINE_DIR"
echo "$WEIGHTS_DIR"
echo "$ENV_DIR"

echo "$TMB_RFD2_CKPT_PATH"
echo "$TMB_RFD2_MOTIF_SCAFFOLDING_CKPT_PATH"
echo "$TMB_MPNN_CKPT_PATH"
echo "$TMB_MPNN_COND_CKPT_PATH"

echo "$MODULUS_APPTAINER_PATH"
echo "$TMB_RFD2_APPTAINER_PATH"
echo "$TMB_MPNN_APPTAINER_PATH"
```

Check that the files exist.

```bash
ls "$TMB_RFD2_CKPT_PATH"
ls "$TMB_RFD2_MOTIF_SCAFFOLDING_CKPT_PATH"
ls "$TMB_MPNN_CKPT_PATH"
ls "$TMB_MPNN_COND_CKPT_PATH"

ls "$MODULUS_APPTAINER_PATH"
ls "$TMB_RFD2_APPTAINER_PATH"
ls "$TMB_MPNN_APPTAINER_PATH"
```

## 6. Re-export paths in a new terminal session

Environment variables are not permanent unless added to your shell startup file. If you open a new terminal session, rerun:

```bash
cd /path/to/TMB_pipeline

export TMB_PIPELINE_DIR="$PWD"
export WEIGHTS_DIR="$TMB_PIPELINE_DIR/weights"
export ENV_DIR="$TMB_PIPELINE_DIR/environments"

export TMB_RFD2_CKPT_PATH="$WEIGHTS_DIR/TMB_RFD2.pt"
export TMB_RFD2_MOTIF_SCAFFOLDING_CKPT_PATH="$WEIGHTS_DIR/TMB_RFD2_motif_scaffolding.pt"
export TMB_MPNN_CKPT_PATH="$WEIGHTS_DIR/TMB_MPNN.pth"
export TMB_MPNN_COND_CKPT_PATH="$WEIGHTS_DIR/TMB_MPNN_tuning_hydrophobic_thickness.pth"

export MODULUS_APPTAINER_PATH="$ENV_DIR/modulus_latest.sif"
export TMB_RFD2_APPTAINER_PATH="$ENV_DIR/TMB_RFD2.sif"
export TMB_MPNN_APPTAINER_PATH="$ENV_DIR/TMB_MPNN.sif"
```

---

---

## Distillation set notebook

The notebook at [Distillation_set_curation/distillation_set.ipynb](Distillation_set_curation/distillation_set.ipynb) loads the distillation-set silent file from:

- [Distillation_set_curation/distillation_set.silent](Distillation_set_curation/distillation_set.silent) by default, or
- the path provided by the `DISTILLATION_SILENT_PATH` environment variable.

The silent file is large, so it is intentionally excluded from git. Keep a local copy in the path above, or point the environment variable at your own copy before running the notebook.

This notebook also expects a PyRosetta-enabled environment. It adds [TMB_RFD2/TMB_RFD2_code](TMB_RFD2/TMB_RFD2_code) to `sys.path` automatically so it can reuse the repository's shared silent-file reader.

---

## Notes

All assets are hosted at:

```text
https://files.ipd.uw.edu/pub/TMB_pipeline/
```

This pipeline assumes access to:

* a SLURM cluster
* GPU resources for full-scale runs
* Singularity or Apptainer

---

## License

This repository is released under the BSD 3-Clause License ([LICENSE](LICENSE)).

Third-party code is vendored into `TMB_MPNN/TMB_MPNN_code/` and stays under its
own license — notably MMPretrain and OpenFold, both Apache License 2.0. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the full list and the
locations of the license texts.
