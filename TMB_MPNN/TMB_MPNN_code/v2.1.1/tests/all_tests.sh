#!/bin/bash
#SBATCH --mem=16g
#SBATCH -c 1
#SBATCH --output=submit_new.out

pdb_path="./1BC8.pdb"

apptainer exec /software/containers/mlfold.sif python ./run.py \
        --model_type "protein_mpnn" \
        --seed 111 \
        --pdb_path $pdb_path \
        --batch_size 3 \
        --number_of_batches 2 \
        --out_folder "./ytest_protein_mpnn_vX"


apptainer exec /software/containers/mlfold.sif python ./run.py \
        --model_type "ligand_mpnn" \
        --seed 111 \
        --pdb_path $pdb_path \
        --batch_size 3 \
        --number_of_batches 2 \
        --out_folder "./ytest_ligand_mpnn_vX"

apptainer exec /software/containers/mlfold.sif python ./run.py \
        --model_type "per_residue_label_membrane_mpnn" \
        --seed 111 \
        --pdb_path $pdb_path \
        --batch_size 3 \
        --number_of_batches 2 \
        --out_folder "./ytest_per_residue_label_membrane_mpnn_vX" \
        --transmembrane_buried "C14 C15 C16 C17 C18" \
        --transmembrane_interface "C19 C20"

apptainer exec /software/containers/mlfold.sif python ./run.py \
        --model_type "global_label_membrane_mpnn" \
        --seed 111 \
        --pdb_path $pdb_path \
        --batch_size 3 \
        --number_of_batches 2 \
        --out_folder "./ytest_global_label_membrane_mpnn_vX" \
        --global_transmembrane_label 0

apptainer exec /software/containers/mlfold.sif python ./run.py \
        --model_type "soluble_mpnn" \
        --seed 111 \
        --pdb_path $pdb_path \
        --batch_size 3 \
        --number_of_batches 2 \
        --out_folder "./ytest_soluble_mpnn_vX"

apptainer exec /software/containers/mlfold.sif python ./run.py \
        --model_type "antibody_mpnn" \
        --seed 111 \
        --pdb_path $pdb_path \
        --batch_size 3 \
        --number_of_batches 2 \
        --out_folder "./ytest_antibody_mpnn_vX"

apptainer exec /software/containers/mlfold.sif python ./run.py \
        --model_type "pssm_mpnn" \
        --seed 111 \
        --pdb_path $pdb_path \
        --batch_size 3 \
        --number_of_batches 2 \
        --out_folder "./ytest_pssm_mpnn_xV" \
        --pssm_input "./pssm_test.json"

pdb_path="/home/justas/MPNN_tests/fused_mpnn/make_msa/6tht.pdb"

apptainer exec /software/containers/mlfold.sif python ./run.py \
        --model_type "msa_mpnn" \
        --seed 111 \
        --pdb_path $pdb_path \
        --batch_size 3 \
        --number_of_batches 2 \
        --out_folder "./ytest_msa_mpnn_xV" \
        --msa_path "/home/justas/MPNN_tests/fused_mpnn/make_msa/out_4/hhblits/t000_.1e-10.id90cov75.a3m"
