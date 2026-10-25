import copy
import glob
from unittest import TestCase
import json
import numpy as np
import torch
from biotite.structure.io.pdbx import list_assemblies
from tqdm import tqdm

from mpnn.datasets.transforms import (
    BuildMPNNInput,
    Compose,
    ExcludeByResName,
    GetAssembly,
    LoadFile,
    MPNNLoadCif,
    GetChainInfo,
    MPNNLoadPDB,
    SeperateProteinLigand,
)

original_train_loader = MPNNLoadCif(
    parser_exclude_res_list=["HOH", "NA", "CL", "K", "BR"],
    atoms_to_load="all",
    parse_only_DNA_RNA=False,
    protein_backbone_occ_cutoff=0.8,
    protein_side_chain_occ_cutoff=0.5,
    excluded_elements=[1],
    ligand_occ_cutoff=0.01,
    num_nn=25,  # params["NUM_CONTEXT_ATOMS"],
    parse_only_proteins=False,
)

new_train_loader = Compose(
    [
        LoadFile(),
        GetAssembly(),
        ExcludeByResName(exclude_res_list=["HOH", "NA", "CL", "K", "BR"]),
        GetChainInfo(),
        SeperateProteinLigand(),
        BuildMPNNInput(lig_nn_num=25),
    ]
)


def test_one_file(file_path):
    results = LoadFile()(dict(cif_path=file_path))
    assemblys = list_assemblies(results["file_obj"])
    compare_keys = (
        # 'S',
        'seq',
        "X_m",
        "X",
        "Y_t",
        "Y",
        "R_idx_offset",  # check the offset inst
    )
    for assembly in assemblys:
        input_data = dict(cif_path=file_path, assembly_id=assembly)
        input_data.update(results)

        original_output = original_train_loader(copy.deepcopy(input_data))
        new_output = new_train_loader(copy.deepcopy(input_data))

        original_output["R_idx_offset"] = (
            original_output["R_idx"][:, None] - original_output["R_idx"][None, :]
        )
        new_output["R_idx_offset"] = new_output["R_idx"][:, None] - new_output["R_idx"][None, :]

        for key in compare_keys:
            if key.startswith("X"):
                # revert O and Cb, 3rd and 4th columns
                original_output[key][:, [3, 4]] = original_output[key][:, [4, 3]]

            if key == 'X':
                # the side chain will be rotated in the original output, such as PDB ID: 11bg
                original_output[key] *= original_output['X_m'][:, :, None]
                # new_output[key] *= new_output['X_m'][:, :, None]
            
            if key == "seq":
                np.testing.assert_array_equal(
                    original_output[key], new_output[key], err_msg=f"{file_path} {assembly} {key}"
                )
            else:
                np.testing.assert_array_almost_equal(
                    original_output[key], new_output[key], err_msg=f"{file_path} {assembly} {key}",
                    decimal=5
                )


if __name__ == "__main__":
    # files = glob.glob("/databases/TrRosetta/cif/*/*.cif.gz")
    with open("/home/faxue/data/TrRosetta/all_3.5A_v0.jsonl", "r") as f:
        data = [json.loads(line) for line in f]
    data = [
        i for i in data if i["method"].startswith("X") and len(i["protein"])
    ]  # only use X-ray data

    print(f'Total {len(data)} files')
    failed_ids = []
    for i, item in enumerate(tqdm(data)):
        pdb_id = item["label"]
        if pdb_id in {
            '148l',  # chain B should be ligand
        }:
            continue
        # pdb_id = '11bg'
        file_path = f"/databases/rcsb/cif/{pdb_id[1:3]}/{pdb_id}.cif.gz"
        # try:
        test_one_file(file_path)
        # except Exception as e:
        #     print(f"Failed {pdb_id}: {e}")
        #     failed_ids.append(dict(pdb_id=pdb_id, error=str(e)))
        # exit()
    print(f"{len(failed_ids)} / {len(data)} failed")
    with open("failed_ids.jsonl", "w") as f:
        for item in failed_ids:
            f.write(json.dumps(item) + "\n")
