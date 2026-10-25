import os
import tempfile
from unittest import TestCase

import numpy as np
from biotite.structure.io import load_structure

from data_utils import save_output_structure
from mpnn.configs.inference import test_pipeline


class TestPDBWriter(TestCase):

    def test_save_output_structure(self):
        """
        Load a PDB, save it to a CIF, and then load it back in
        """
        results = test_pipeline(dict(cif_path="tests/data/1BC8.cif", assembly_id="1"))
        fake_b_factor = np.random.rand(len(results["X"]))
        fake_S = results["S"].copy() + 1
        # Save the CIF
        with tempfile.NamedTemporaryFile(suffix=".cif", delete=True) as tmp_file:
            cif_save_path = tmp_file.name
            save_output_structure(
                cif_save_path,
                X=results["X"],
                X_m=results["X_m"],
                b_factors=fake_b_factor,
                R_idx=results["R_idx"],
                chain_letters=results["chain_letters"],
                S=fake_S,
                other_atoms=results["other_atoms"],
                icodes=results["ins_codes"],
            )
            new_results = test_pipeline(dict(cif_path=cif_save_path, assembly_id="1"))
            # Compare the original and new results
            self.assertTrue(np.allclose(results["X"], new_results["X"]))
            self.assertTrue(np.allclose(results["X_m"], new_results["X_m"]))

            new_atom_array = load_structure(cif_save_path, extra_fields=["b_factor"])
            new_load_b_factor = new_atom_array.b_factor
            self.assertTrue(np.allclose(fake_b_factor, new_load_b_factor))

            self.assertTrue(np.allclose(fake_S, new_results["S"]))
            self.assertTrue(np.allclose(results["R_idx"], new_results["R_idx"]))
            self.assertTrue(np.allclose(results["chain_letters"], new_results["chain_letters"]))
            self.assertTrue(np.allclose(results["other_atoms"], new_results["other_atoms"]))
            self.assertTrue(np.allclose(results["ins_codes"], new_results["ins_codes"]))


if __name__ == "__main__":
    import unittest

    unittest.main()
