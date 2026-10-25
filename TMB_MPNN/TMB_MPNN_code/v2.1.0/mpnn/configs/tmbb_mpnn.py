from mmengine.config import read_base

from mpnn.datasets.transforms import (
    BuildInferenceInput,
    BuildMPNNInput,
    ExcludeByResName,
    GetChainInfo,
    GetStructure,
    LoadFile,
    PackInputs,
    SeperateProteinLigand,
)

with read_base():
    from .ligand_mpnn import model


test_pipeline = [
    dict(type=LoadFile),
    dict(type=GetStructure, random_select_model=False),
    dict(type=ExcludeByResName, exclude_res_list=["HOH", "NA", "CL", "K", "BR"]),
    dict(type=GetChainInfo),
    dict(
        type=SeperateProteinLigand,
        protein_backbone_occ_cutoff=0.0,
        protein_side_chain_occ_cutoff=0.0,
        ligand_occ_cutoff=0.0
    ),
    dict(type=BuildMPNNInput, lig_nn_num=25),
    dict(
        type=PackInputs,
        keys=[
            "X",
            "X_m",
            "Y",
            "Y_t",
            "Y_coords",
            "S",
            "R_idx",
            "chain_labels",
            "ins_codes",  # TODO: add to inference
            "chain_letters",
            "mask",
            "Y_m",
            "mask_XY",
            "other_atoms",
            "atom_array"
        ]
    ),
    dict(type=BuildInferenceInput),
]


load_from = '/net/scratch/faxue/share/mpnn_enhancer/tmbb_mpnn_bf_rep12_20241128-71f23636.pth'
