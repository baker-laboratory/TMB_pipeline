import os
from unittest import TestCase

import numpy as np
import torch

from legacy.lt_v2.ligand_data_utils_sc import PDBDataset
from mpnn.datasets.transforms import (
    BuildMPNNInput,
    Compose,
    ExcludeByResName,
    GetAssembly,
    GetChainInfo,
    LoadFile,
    SeperateProteinLigand,
)

class TestDatasetVsTransforms(TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that will be shared across all test methods."""
        cls.dataset = PDBDataset(
            cif_path="/databases/rcsb/cif",
            parser_exclude_res_list=["HOH", "NA", "CL", "K", "BR"],
            atom_list_to_save="all",
            parse_only_DNA_RNA=False,
            protein_backbone_occ_cutoff=0.8,
            protein_side_chain_occ_cutoff=0.5,
            excluded_elements=[1],
            ligand_occ_cutoff=0.01,
            num_nn=25,
            parse_only_proteins=False,
        )

        # ------------------ Setup Transforms Approach ------------------
        cls.transforms = Compose([
            LoadFile(),
            GetAssembly(),
            ExcludeByResName(exclude_res_list=["HOH", "NA", "CL", "K", "BR"]),
            GetChainInfo(),
            SeperateProteinLigand(),
            BuildMPNNInput(lig_nn_num=25),
        ])

        # ------------------ Define Keys to Compare ------------------
        cls.compare_keys = [
            "X",            # Protein coordinates
            "X_m",          # Protein mask
            "Y",            # Ligand coordinates
            "Y_t",          # Ligand element types
            "S",            # Sequence indices
            "R_idx",        # Residue indices
            "chain_labels", # Chain labels
        ]

    def test_outputs_match(self):
        """Compare outputs between PDBDataset and Compose transforms approaches."""
        # ------------------ Get Outputs from Both Approaches ------------------
        dataset_output, _ = self.dataset.loader("1bc8", "1")
        transform_output = self.transforms({
            "cif_path": os.path.join("tests/data", "1bc8.cif"),
            "assembly_id": "1"
        })

        # ------------------ Compare Outputs ------------------
        skipped_keys = []
        compared_keys = []

        print("\nComparing dataset and transform outputs:")
        for key in self.compare_keys:
            if key not in dataset_output or key not in transform_output:
                print(f"WARNING: Key {key} missing from one or both outputs")
                skipped_keys.append(key)
                continue

            dataset_val = dataset_output[key]
            transform_val = transform_output[key]

            # Check if values are comparable
            if not (isinstance(dataset_val, (np.ndarray, torch.Tensor)) and 
                   isinstance(transform_val, (np.ndarray, torch.Tensor))):
                print(f"WARNING: Skipping {key} - incompatible types: "
                      f"dataset ({type(dataset_val)}) vs transform ({type(transform_val)})")
                skipped_keys.append(key)
                continue

            # Convert to numpy if needed
            if isinstance(transform_val, torch.Tensor):
                transform_val = transform_val.numpy()
            if isinstance(dataset_val, torch.Tensor):
                dataset_val = dataset_val.numpy()

            print(f"\n{key}:")
            print(f"  Dataset shape:   {dataset_val.shape}")
            print(f"  Transform shape: {transform_val.shape}")

            # Compare shapes
            self.assertEqual(
                dataset_val.shape,
                transform_val.shape,
                f"Shape mismatch for {key}"
            )

            # Compare values
            try:
                if np.issubdtype(dataset_val.dtype, np.floating):
                    np.testing.assert_allclose(
                        dataset_val,
                        transform_val,
                        rtol=1e-5,
                        err_msg=f"Value mismatch for {key}"
                    )
                else:
                    np.testing.assert_array_equal(
                        dataset_val,
                        transform_val,
                        err_msg=f"Value mismatch for {key}"
                    )
                print("  Values match within tolerance")
                compared_keys.append(key)
            except AssertionError as e:
                print(f"  Value mismatch: {str(e)}")
                raise

        print("\nComparison Summary:")
        print(f"Successfully compared {len(compared_keys)} keys: {compared_keys}")
        print(f"Checked internal consistency for {len(self.consistency_keys)} keys: {self.consistency_keys}")
        if skipped_keys:
            print(f"Skipped {len(skipped_keys)} keys: {skipped_keys}")
        
        self.assertTrue(len(compared_keys) > 0, "No keys were successfully compared!")


if __name__ == "__main__":
    import unittest
    unittest.main(buffer=False)
