import os
import warnings

import biotite
import numpy as np
import torch

from mpnn.datasets.transforms import (
    BuildMPNNInput,
    Compose,
    ExcludeByResName,
    GetAssembly,
    GetChainInfo,
    LoadFile,
    SeperateProteinLigand,
)
from mpnn.models.train_data_utils import padding_batch


class PDBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cif_path,
        cache_folder=None,
        exclude_res_list=["HOH", "NA", "CL", "K", "BR", "UNX"],
        protein_backbone_occ_cutoff=0.8,
        protein_side_chain_occ_cutoff=0.5,
        ligand_occ_cutoff=0.01,
        lig_nn_num=25,
    ):
        self.cif_path = cif_path
        self.cache_folder = cache_folder
        if cache_folder is not None:
            os.makedirs(cache_folder, exist_ok=True)

        self.train_pipeline = Compose(
            [
                LoadFile(),
                GetAssembly(),
                ExcludeByResName(exclude_res_list=exclude_res_list),
                GetChainInfo(),
                SeperateProteinLigand(
                    protein_backbone_occ_cutoff=protein_backbone_occ_cutoff,
                    protein_side_chain_occ_cutoff=protein_side_chain_occ_cutoff,
                    ligand_occ_cutoff=ligand_occ_cutoff,
                ),
                BuildMPNNInput(lig_nn_num=lig_nn_num),
            ]
        )

    def __getitem__(self, index):
        """
        index = [[("pdb_id", "biounit_id"), ("pdb_id", "biounit_id")]]
        """
        batch = [self.loader(pdb_tuple[0], pdb_tuple[1]) for pdb_tuple in index[0]]
        batch = [x for x in batch if x is not None]
        batch = padding_batch(batch, mode="all")
        return batch

    def __len__(self):
        return self.sample_num

    def loader(self, pdb_id, assembly_id):
        assert len(pdb_id) == 4
        if self.cache_folder is not None:
            cache_path = f"{self.cache_folder}/{pdb_id}_{assembly_id}.pth"
            if os.path.exists(cache_path):
                try:
                    out_dict = torch.load(cache_path)
                    return out_dict
                except:
                    os.remove(cache_path)

        # processing
        cif_path = os.path.join(self.cif_path, f"{pdb_id[1:3]}/{pdb_id}.cif.gz")
        try:
            out_dict = self.train_pipeline((dict(cif_path=cif_path, assembly_id=assembly_id)))
        except biotite.InvalidFileError:
            warnings.warn(f"Invalid file: {pdb_id}")
            return None

        protein_len = out_dict["X"].shape[0]
        out_dict = {
            "X": out_dict["X"],
            "X_m": out_dict["X_m"],
            "Y": out_dict["Y"],
            "Y_t": out_dict["Y_t"],
            "S": out_dict["S"],
            "R_idx": out_dict["R_idx"],
            "chain_labels": out_dict["chain_labels"],
            "protein_L": protein_len,
        }

        # save cache
        if self.cache_folder is not None:
            torch.save(out_dict, cache_path)

        return out_dict
