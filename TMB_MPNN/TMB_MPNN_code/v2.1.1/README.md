# MPNN inferencer V2.0

The new MPNN inferencer unifies the training and inference processes, making model delivery easier.

- Currently, only support the TMBB_MPNN model.
- The default checkpoint is stored inside the config file, you can also use `--checkpoint` to specify another checkpoint file.
- All design arguments remain unchanged.

```Bash
/software/containers/users/faxue/modelhub_wMM_20250502.sif \
  python run.py \
    --model_type "mpnn/configs/tmbb_mpnn.py" \
    --pdb_path tests/data/1BC8.pdb \
    --out_folder output \
    --batch_size 8 \
    --number_of_batches 1 \
    --pack_side_chains 0
```

## Recent changes
- v2.1.0:
    - Added interface condition for TMBB_MPNN ([`mpnn/configs/big_tmbb_80p_surf_condition.py`](mpnn/configs/big_tmbb_80p_surf_condition.py)).
    - Updated the cifutils version to `455c1d`, which is NOT compatible with the previous apptainer image. Using `/software/containers/users/faxue/modelhub_wMM_20250502.sif` instead.


# pLDDT enhancer model

We have enhanced ligandMPNN to generate sequences with higher pLDDT scores, which could improve the success rate. To use the enhanced model, simply add the `--enhance` argument to the command, like this:

```Bash
apptainer exec /software/containers/mlfold.sif python run.py \
    --model_type "ligand_mpnn" \
    --pdb_path tests/data/1BC8.pdb \
    --out_folder output \
    --batch_size 1 \
    --number_of_batches 1 \
    --pack_side_chains 0 \
    --enhance plddt_16_20240910-b65a33eb
```

> This model is still under avtive development. Please ask Fanglei Xue for the appropriate value for the `--enhance` argument.
