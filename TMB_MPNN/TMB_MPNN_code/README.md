> **Note:** this is internal development documentation, kept for reference. The
> commands below point at IPD-internal container and checkpoint paths and will
> not resolve outside the cluster. For a supported, portable workflow see the
> [top-level README](../../README.md) and the runnable example in
> [run_TMB_MPNN_example/](../run_TMB_MPNN_example/).

# MPNN inferencer V2.0

The new MPNN inferencer unifies the training and inference processes, making model delivery easier.

- Currently, only support the TMBB_MPNN model.
- The default checkpoint is stored inside the config file, you can also use `--checkpoint` to specify another checkpoint file.
- All design arguments remain unchanged.

```Bash
/net/scratch/faxue/share/envs/mpnnv2.sif python run.py \
    --model_type "mpnn/configs/tmbb_mpnn.py" \
    --pdb_path tests/data/1BC8.pdb \
    --out_folder output \
    --batch_size 8 \
    --number_of_batches 1 \
    --pack_side_chains 0
```


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


# Sequence design example:
```
apptainer exec /software/containers/mlfold.sif python run.py \
--model_type "ligand_mpnn" \
--pdb_path $pdb_path \
--out_folder $out_folder_path \
--batch_size 1 \
--number_of_batches 1 \
--pack_side_chains 0 #optionally pack side chains
```

# Side chain only packing example:
```
apptainer exec /software/containers/mlfold.sif python sc_run.py \
--pdb_path $pdb_path \
--out_folder $out_folder_path \
--pack_side_chains 1 \
--batch_size 1 \
--number_of_batches 1
```

# Design only specified residues:
```
apptainer exec /software/containers/mlfold.sif python run.py \
--model_type "ligand_mpnn" \
--pdb_path $pdb_path \
--out_folder $out_folder_path \
--batch_size 1 \
--number_of_batches 1 \
--redesigned_residues "A12 A13 A14 B2 B25" \
--omit_AA "X" \
--pack_side_chains 1 \
--repack_everything 1
```



# Model types:

- protein_mpnn - original ProteinMPNN trained on the whole PDB exluding non-protein atoms
- ligand_mpnn - atomic context aware model trained with small molecules, nucleotides, metals etc on the whole PDB
- per_residue_label_membrane_mpnn - ProteinMPNN model trained with addition label per residue specifying if that residue is buried or exposed
- global_label_membrane_mpnn - ProteinMPNN model trained with global label per PDB id to specify if protein is transmembrane
- soluble_mpnn - ProteinMPNN trained only on soluble PDB ids
- pssm_mpnn - ProteinMPNN with additional PSSM like inputs
- antibody_mpnn - ProteinMPNN trained with bias towards antibody PDBs



--
# Training

please see the `training` branch.

 All the codes we need to focus on is located in the `/mpnn` folder.

