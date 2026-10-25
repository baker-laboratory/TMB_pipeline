import copy
from unittest import TestCase

import numpy as np
import torch

from mpnn.datasets.transforms import (
    BuildMPNNInput,
    Compose,
    ExcludeByResName,
    GetAssembly,
    GetChainInfo,
    LoadFile,
    MPNNLoadCif,
    MPNNLoadPDB,
    SeperateProteinLigand,
)


class TestTransforms(TestCase):

    init_args = dict(
        parser_exclude_res_list=["HOH", "NA", "CL", "K", "BR"],
        atoms_to_load="backbone",
        parse_only_DNA_RNA=False,
        protein_backbone_occ_cutoff=0.8,
        protein_side_chain_occ_cutoff=0.5,
        excluded_elements=[1],
        ligand_occ_cutoff=0.01,
        num_nn=25,  # params["NUM_CONTEXT_ATOMS"],
        parse_only_proteins=False,
    )

    @classmethod
    def setUpClass(self) -> None:
        self.cif_loader = MPNNLoadCif(**self.init_args)
        self.pdb_loader = MPNNLoadPDB(**self.init_args)
        self.new_loader = Compose(
            [
                LoadFile(),
                GetAssembly(),
                ExcludeByResName(exclude_res_list=["HOH", "NA", "CL", "K", "BR"]),
                GetChainInfo(),
                SeperateProteinLigand(),
                BuildMPNNInput(lig_nn_num=25),
            ]
        )

    def test_pdb_loader_train(self):
        """PDB should return the same value as the CIF loader"""

        cif_output = self.cif_loader(dict(cif_path="tests/data/1BC8.cif", assembly_id="1"))
        pdb_output = self.pdb_loader(dict(cif_path="tests/data/1BC8.pdb"))

        pdb_output = {k: v.numpy() for k, v in pdb_output.items() if isinstance(v, torch.Tensor)}

        assert np.allclose(cif_output["X"], pdb_output["X"])

        compare_keys = (
            "X",
            "X_m",
            "Y",
            "Y_t",
            "R_idx",
        )
        for key in compare_keys:
            assert np.allclose(cif_output[key], pdb_output[key])

        # check O coordinates
        all_args = copy.deepcopy(self.init_args)
        all_args["atoms_to_load"] = "all"
        all_loader = MPNNLoadCif(**all_args)
        all_output = all_loader(dict(cif_path="tests/data/1BC8.cif", assembly_id="1"))
        self.assertTrue(np.allclose(all_output["X"][:, 4], pdb_output["X"][:, 3]))

    def test_cifutils_loader(self):
        train_cfg = copy.deepcopy(self.init_args)
        train_cfg["atoms_to_load"] = "all"
        cif_loader = MPNNLoadCif(**train_cfg)
        input_data = dict(cif_path="tests/data/1BC8.cif", assembly_id="1")
        cif_output = cif_loader(input_data)

        new_loader_output = self.new_loader(input_data)

        compare_keys = (
            "X",
            "X_m",
            "Y_coords",
            "Y",
            "Y_t",
            "R_idx",
        )
        for key in compare_keys:
            if key.startswith("X"):
                # revert O and Cb, 3rd and 4th columns
                cif_output[key][:, [3, 4]] = cif_output[key][:, [4, 3]]

            np.testing.assert_allclose(cif_output[key], new_loader_output[key], rtol=1e-5, err_msg=key)

    def test_cifutils_loader_test(self):
        pdb_loader = MPNNLoadPDB(**self.init_args)
        input_data = dict(cif_path="tests/data/1BC8.pdb", assembly_id="1")
        pdb_output = pdb_loader(copy.deepcopy(input_data))


        cifutils_output = self.new_loader(input_data)

        compare_keys = (
            "xyz_37",
            "xyz_37_m",
            "X",
            "X_m",
            "Y",
            "Y_t",
            "R_idx",
            "mask",
            "Y_m",
        )
        for key in compare_keys:
            # xyz_37 is the same as X (reverse O and Cb)
            if key.startswith("xyz"):
                pdb_output[key][:, [3, 4]] = pdb_output[key][:, [4, 3]]
                cifutils_output[key] = cifutils_output[key.replace("xyz_37", "X")]
            # X only has backbone atoms during inference
            if key.startswith("X"):
                cifutils_output[key] = cifutils_output[key][:, :4]
            np.testing.assert_allclose(pdb_output[key], cifutils_output[key], rtol=1e-5, err_msg=key)


if __name__ == "__main__":
    import unittest

    unittest.main()
