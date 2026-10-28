"""CPU tests for barrel conditioning and rotations; no RFD2 weights required."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TMB_RFD2"))
sys.path.insert(0, str(ROOT / "TMB_RFD2/TMB_RFD2_code"))

import adj_ss_wrapper_glytwist as conditioning
from rf_diffusion.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_quaternion,
    quaternion_to_matrix,
)


class BarrelConditioningTests(unittest.TestCase):
    def test_adjacency_is_symmetric_and_closes_the_barrel(self):
        for residues_per_strand in [6, 7, 16, 25]:
            with self.subTest(residues_per_strand=residues_per_strand):
                adj = conditioning.generate_paired_and_masked_boxes(4, residues_per_strand)
                self.assertEqual(tuple(adj.shape), (4 * residues_per_strand,) * 2)
                torch.testing.assert_close(adj, adj.T)
                self.assertEqual(set(adj.unique().tolist()), {0, 2, 3})
                blocks = adj.reshape(4, residues_per_strand, 4, residues_per_strand)
                for strand in range(4):
                    self.assertTrue(torch.all(blocks[strand, :, strand, :] == 0))
                    self.assertTrue(torch.all(blocks[strand, :, (strand + 2) % 4, :] == 0))
                    for neighbor in [(strand - 1) % 4, (strand + 1) % 4]:
                        block = blocks[strand, :, neighbor, :]
                        self.assertTrue(torch.all(block >= 2))
                        self.assertEqual(int((block == 3).sum()), 4)

    def test_secondary_structure_marks_only_two_central_strand_residues(self):
        ss = conditioning.generate_flat_masked_ss_tensor(4, 6)
        expected = torch.tensor([3, 3, 1, 1, 3, 3] * 4, dtype=ss.dtype)
        torch.testing.assert_close(ss, expected)

    def test_twists_use_one_based_strand_indices_and_five_residue_windows(self):
        for residues_per_strand in [6, 7]:
            with self.subTest(residues_per_strand=residues_per_strand):
                twist = conditioning.generate_glycine_twist_positions(4, residues_per_strand, [1, 4])
                # The first and last strands are selected, with no leakage to neighbors.
                self.assertEqual(int(twist.sum()), 10)
                rows = twist.reshape(4, residues_per_strand)
                torch.testing.assert_close(rows[0], torch.tensor(
                    [0, 1, 1, 1, 1, 1] + ([0] if residues_per_strand == 7 else []),
                    dtype=twist.dtype,
                ))
                torch.testing.assert_close(rows[0], rows[3])
                self.assertTrue(torch.all(rows[1:3] == 0))

    def test_no_twists_produces_zero_mask(self):
        twist = conditioning.generate_glycine_twist_positions(4, 6, [])
        self.assertEqual(tuple(twist.shape), (24,))
        self.assertEqual(int(twist.count_nonzero()), 0)

    def test_duplicate_twist_indices_are_idempotent(self):
        torch.testing.assert_close(
            conditioning.generate_glycine_twist_positions(4, 6, [2, 2]),
            conditioning.generate_glycine_twist_positions(4, 6, [2]),
        )

    def test_cli_writes_loadable_conditioning_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new directory"
            argv = ["adj_ss_wrapper_glytwist.py", "--n", "4", "--residues_per_strand", "6",
                    "--twist_indices", "1", "4", "--output_dir", str(output)]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                conditioning.main()
            self.assertEqual(len(list(output.glob("*.pt"))), 3)
            adj = torch.load(output / "n4_nres6_adj.pt", weights_only=True)
            ss = torch.load(output / "n4_nres6_ss.pt", weights_only=True)
            twist = torch.load(output / "n4_nres6_twist.pt", weights_only=True)
            self.assertEqual(tuple(adj.shape), (24, 24))
            self.assertEqual(tuple(ss.shape), (24,))
            self.assertEqual(tuple(twist.shape), (24,))
            self.assertEqual(int((ss == 1).sum()), 8)
            self.assertEqual(int(twist.sum()), 10)

    def test_cli_without_twists_writes_only_adjacency_and_ss(self):
        with tempfile.TemporaryDirectory() as directory:
            argv = ["adj_ss_wrapper_glytwist.py", "--n", "4", "--residues_per_strand", "6",
                    "--output_dir", directory]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                conditioning.main()
            self.assertEqual(
                {p.name for p in Path(directory).glob("*.pt")},
                {"n4_nres6_adj.pt", "n4_nres6_ss.pt"},
            )


class RotationTests(unittest.TestCase):
    def test_zero_angle_is_identity_without_nans(self):
        result = axis_angle_to_matrix(torch.zeros(2, 3))
        torch.testing.assert_close(result, torch.eye(3).repeat(2, 1, 1))

    def test_quarter_turn_about_z_maps_x_to_y(self):
        rotation = axis_angle_to_matrix(torch.tensor([0.0, 0.0, torch.pi / 2]))
        torch.testing.assert_close(
            rotation @ torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]),
            atol=1e-6, rtol=1e-6,
        )

    def test_half_turns_and_small_rotations_round_trip(self):
        axes = torch.tensor([[torch.pi, 0, 0], [0, torch.pi, 0], [0, 0, torch.pi],
                             [1e-7, 0, 0], [0.2, -0.4, 0.8]], dtype=torch.float64)
        rotations = axis_angle_to_matrix(axes)
        torch.testing.assert_close(
            quaternion_to_matrix(matrix_to_quaternion(rotations)), rotations,
        )
        torch.testing.assert_close(
            rotations @ rotations.transpose(-1, -2),
            torch.eye(3, dtype=torch.float64).expand(5, 3, 3),
        )
        torch.testing.assert_close(torch.linalg.det(rotations), torch.ones(5, dtype=torch.float64))

    def test_invalid_rotation_matrix_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            matrix_to_quaternion(torch.eye(4))


if __name__ == "__main__":
    unittest.main()
