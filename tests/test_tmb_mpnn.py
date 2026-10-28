"""CPU tests for sequence scoring and side-chain packing, without model weights."""

from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TMB_MPNN/TMB_MPNN_code/v2.1.1"))

from mpnn.datasets.fused_data_utils import get_score, get_seq_rec, make_pair_bias
from sc_utils import Packer, make_torsion_features, pack_side_chains


class SequenceScoringTests(unittest.TestCase):
    def test_recovery_ignores_masked_positions_and_handles_batches(self):
        native = torch.tensor([[0, 1, 2, 3]])
        designed = torch.tensor([[0, 9, 2, 8], [0, 1, 9, 3]])
        mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        torch.testing.assert_close(get_seq_rec(native, designed, mask), torch.tensor([1.0, 0.5]))

    def test_score_is_masked_mean_negative_log_probability(self):
        sequence = torch.tensor([[0, 1, 2], [2, 1, 0]])
        logits = torch.arange(126, dtype=torch.float32).reshape(2, 3, 21) / 20
        log_probs = torch.log_softmax(logits, dim=-1)
        mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        mean, per_residue = get_score(sequence, log_probs, mask)
        expected = -log_probs.gather(-1, sequence.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(per_residue, expected)
        torch.testing.assert_close(mean, torch.stack([
            (expected[0, 0] + expected[0, 2]) / 2, expected[1, 1],
        ]))

    def test_score_for_fully_masked_sequence_is_finite_zero(self):
        sequence = torch.zeros(1, 3, dtype=torch.long)
        log_probs = torch.log_softmax(torch.zeros(1, 3, 21), dim=-1)
        mean, _ = get_score(sequence, log_probs, torch.zeros(1, 3))
        torch.testing.assert_close(mean, torch.zeros(1))

    def test_pair_bias_respects_chain_breaks_numbering_gaps_and_direction(self):
        # 1->2 is a real neighbor; 2->3 crosses chains; 3->5 is a numbering gap.
        chains = torch.tensor([0, 0, 1, 1, 1])
        indices = torch.tensor([1, 2, 3, 5, 6])
        aa_bias = torch.zeros(21, 21)
        aa_bias[0, 1] = 2.5
        result = make_pair_bias(chains, indices, aa_bias)
        self.assertEqual(tuple(result.shape), (1, 5, 21, 5, 21))
        self.assertEqual(result[0, 0, 0, 1, 1].item(), 2.5)
        self.assertEqual(result[0, 1, 1, 0, 0].item(), 2.5)
        self.assertEqual(result[0, 3, 0, 4, 1].item(), 2.5)
        self.assertEqual(result[0, 4, 1, 3, 0].item(), 2.5)
        self.assertEqual(int(result.count_nonzero()), 4)


class SideChainPackingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        # Non-collinear N/CA/C/O coordinates, translated for four residues.
        backbone = torch.tensor([[-0.525, 1.363, 0], [0, 0, 0],
                                 [1.526, 0, 0], [2.153, 1.062, 0]])
        xyz = backbone.repeat(1, 4, 1, 1)
        xyz[0, :, :, 0] += 3.8 * torch.arange(4)[:, None]
        self.features = {
            "X": xyz,
            "S": torch.tensor([[5, 0, 18, 8]]),  # Gly, Ala, Trp, Lys in MPNN order
            "mask": torch.tensor([[True, True, True, False]]),
            "chain_mask": torch.ones(1, 4),
            "chain_labels": torch.zeros(1, 4, dtype=torch.long),
            "R_idx": torch.arange(1, 5)[None, :],
        }

    def test_torsion_features_map_residue_types_and_mask_padding(self):
        features = make_torsion_features(self.features)
        torch.testing.assert_close(features["S_af2"], torch.tensor([[7, 0, 17, 11]]))
        # Gly has no side chain; Ala has CB; Trp has all 14 heavy atoms.
        torch.testing.assert_close(features["xyz14_m"].sum(-1), torch.tensor([[4.0, 5.0, 14.0, 0.0]]))
        self.assertTrue(torch.all(features["xyz14_noised"][0, 3] == 0))
        self.assertTrue(torch.isfinite(features["xyz14_noised"]).all())
        torch.testing.assert_close(features["xyz14_noised"][0, :3, 1], self.features["X"][0, :3, 1])

    def test_fixed_residue_torsions_are_not_randomized(self):
        xyz37 = torch.randn(1, 4, 37, 3)
        xyz37[:, :, :3] = self.features["X"][:, :, :3]
        xyz37[:, :, 4] = self.features["X"][:, :, 3]
        self.features["xyz_37"] = xyz37
        self.features["chain_mask"] = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
        features = make_torsion_features(self.features, repack_everything=False)
        torch.testing.assert_close(
            features["torsions_noised"][:, [0, 2], 3:],
            features["torsions_true"][:, [0, 2]],
        )

    def test_packer_runs_denoising_without_checkpoint_or_ligand(self):
        # Random weights test computation and geometry, not model quality.
        model = Packer(top_k=32, dropout=0.0, device=torch.device("cpu")).eval()
        original_ca = self.features["X"][0, :3, 1].clone()
        with torch.no_grad():
            packed = pack_side_chains(self.features, model, num_denoising_steps=2, num_samples=2)
        self.assertEqual(tuple(packed["X"].shape), (1, 4, 14, 3))
        self.assertEqual(tuple(packed["b_factors"].shape), (1, 4, 14))
        self.assertEqual(tuple(packed["sample"].shape), (1, 4, 4))
        for key in ["X", "b_factors", "log_prob", "concentration"]:
            self.assertTrue(torch.isfinite(packed[key]).all(), key)
        self.assertTrue(torch.all(packed["concentration"] > 0))
        self.assertTrue(torch.all(packed["X"][0, 3] == 0))
        torch.testing.assert_close(packed["X"][0, :3, 1], original_ca)
        torch.testing.assert_close(packed["X_m"].sum(-1), torch.tensor([[4.0, 5.0, 14.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()
