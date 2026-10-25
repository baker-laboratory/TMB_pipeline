import copy
from unittest import TestCase

import numpy as np
import torch

from legacy.fused_mpnn.data_utils import (
    featurize,
    parse_PDB
)
from mpnn.datasets.transforms import (
    BuildMPNNInput,
    Compose,
    ExcludeByResName,
    GetAssembly,
    GetChainInfo,
    LoadFile,
    SeperateProteinLigand,
)

class TestTransforms(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.params = dict(
            number_of_ligand_atoms = 25,
            ligand_mpnn_use_atom_context = True,
            model_type = "ligand_mpnn",
        )

        # ------------------ Setup Transforms Approach ------------------
        cls.transforms = Compose(
            [
                LoadFile(),
                GetAssembly(),
                ExcludeByResName(exclude_res_list=["HOH", "NA", "CL", "K", "BR", "UNX"]),
                GetChainInfo(),
                SeperateProteinLigand(
                    protein_backbone_occ_cutoff=0.,
                    protein_side_chain_occ_cutoff=0.,
                    ligand_occ_cutoff=0.,
                ),
                BuildMPNNInput(lig_nn_num=25),
            ]
        )

        # ------------------ Define Keys to Compare ------------------
        cls.compare_keys = [
            "X",            # Protein coordinates
            "X_m",          # Protein mask
            "Y",            # Ligand coordinates
            "Y_t",          # Ligand element types
            "S",            # Sequence indices
            "R_idx",        # Residue indices
            "chain_labels", # Chain labels
            "mask_c",       # Chain mask
            "xyz_37",        # All atoms coordinates
            "xyz_37_m"       # All atoms mask
        ]

    def process_PDB_file(self, pdb_file):
        protein_dict, backbone, other_atoms, icodes, water_atoms = parse_PDB(pdb_file,
                                                                             atom_context_num=self.params["number_of_ligand_atoms"], 
                                                                             parse_all_atoms=True)

        protein_dict["chain_mask"] = torch.tensor(np.ones_like(protein_dict["chain_letters"], dtype=np.int32))
        protein_dict["side_chain_mask"] = protein_dict["chain_mask"]

        pdb_output = featurize(protein_dict,
                               use_atom_context=self.params["ligand_mpnn_use_atom_context"],
                               number_of_ligand_atoms=self.params["number_of_ligand_atoms"],
                               model_type=self.params["model_type"])

        X_m = torch.zeros(pdb_output["X"].squeeze().shape[:2]).bool()
        X_m[pdb_output["mask"].squeeze().bool()] = True
        pdb_output["X_m"] = X_m[None,]

        pdb_output["mask_c"] = protein_dict["mask_c"][0]

        return pdb_output

    def process_cifutils_file(self, cif_file):
        cifutils_output = self.transforms({
            "cif_path": cif_file,
            "assembly_id": "1"
        })

        feature_dict = dict()
        for k, v in cifutils_output.items():
            if isinstance(v, np.ndarray):
                if not np.issubdtype(v.dtype, np.str_):
                    v = torch.from_numpy(v)[None, ]
            elif isinstance(v, torch.Tensor):
                v = v[None, ]
            feature_dict[k] = v
        
        # move to device
        for k, v in feature_dict.items():
            if isinstance(v, torch.Tensor):
                feature_dict[k] = v
        feature_dict['xyz_37'] = feature_dict['X']
        feature_dict['xyz_37_m'] = feature_dict['X_m']
        feature_dict['X'] = feature_dict['X'][:, :, :4]
        feature_dict['X_m'] = feature_dict['X_m'][:, :, :4]

        # Looks like the mask_c is only for save fasta file
        sorted_chain_letters = sorted(set(cifutils_output["chain_letters"]))
        mask_c = np.array([cifutils_output['chain_letters'] == c for c in sorted_chain_letters])
        mask_c = torch.tensor(mask_c, dtype=bool)[:, ]
        feature_dict['mask_c'] = mask_c.squeeze()

        return feature_dict


    def test_cifutils_loader_test(self):

        # ------------------ Get Outputs from Both Approaches ------------------    

        #parse PDB file
        pdb_output = self.process_PDB_file("tests/data/1BC8.pdb")
        
        # ------------------ Transform with Compose ------------------
        cifutils_output = self.process_cifutils_file("tests/data/1BC8.cif")

        # ------------------ Compare Outputs ------------------
        skipped_keys = []
        compared_keys = []

        print("\nComparing parse_PDB and transform outputs:")
        for key in self.compare_keys:
            if key not in pdb_output or key not in cifutils_output:
                print(f"WARNING: Key {key} missing from one or both outputs")
                skipped_keys.append(key)
                continue  
            
            if key.startswith("xyz"):
                pdb_output[key][:, :, [3, 4]] = pdb_output[key][:, :, [4, 3]]

            pdb_val = pdb_output[key]
            transform_val = cifutils_output[key]

            # Check if values are comparable
            if not (isinstance(pdb_val, (np.ndarray, torch.Tensor)) and 
                   isinstance(transform_val, (np.ndarray, torch.Tensor))):
                print(f"WARNING: Skipping {key} - incompatible types: "
                      f"dataset ({type(pdb_val)}) vs transform ({type(transform_val)})")
                skipped_keys.append(key)
                continue

            # Convert to numpy if needed
            if isinstance(transform_val, torch.Tensor):
                transform_val = transform_val.numpy()
            if isinstance(pdb_val, torch.Tensor):
                pdb_val = pdb_val.numpy()

            print(f"\n{key}:")
            print(f"  Dataset shape:   {pdb_val.shape}")
            print(f"  Transform shape: {transform_val.shape}")

            # Compare shapes
            self.assertEqual(
                pdb_val.shape,
                transform_val.shape,
                f"Shape mismatch for {key}"
            )

            # Compare values
            try:
                if np.issubdtype(pdb_val.dtype, np.floating):
                    np.testing.assert_allclose(
                        pdb_val,
                        transform_val,
                        rtol=1e-5,
                        err_msg=f"Value mismatch for {key}"
                    )
                else:
                    np.testing.assert_array_equal(
                        pdb_val,
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
        if skipped_keys:
            print(f"Skipped {len(skipped_keys)} keys: {skipped_keys}")
        
        self.assertTrue(len(compared_keys) > 0, "No keys were successfully compared!")
        

if __name__ == "__main__":
    import unittest

    unittest.main()
