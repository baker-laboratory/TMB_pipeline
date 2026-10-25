from mpnn.datasets.transforms import (
    BuildMPNNInput,
    Compose,
    ExcludeByResName,
    GetAssembly,
    GetChainInfo,
    LoadFile,
    SeperateProteinLigand,
)

# TODO: use config system to define this

test_pipeline = Compose(
    [
        LoadFile(),
        GetAssembly(),
        ExcludeByResName(exclude_res_list=["HOH", "NA", "CL", "K", "BR", "UNX"]),
        GetChainInfo(),
        SeperateProteinLigand(
            protein_backbone_occ_cutoff=0.0,
            protein_side_chain_occ_cutoff=0.0,
            ligand_occ_cutoff=0.0,
        ),
        BuildMPNNInput(lig_nn_num=25),
    ]
)
