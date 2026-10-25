
import random
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from subprocess import PIPE, Popen
from unittest import TestCase

import mmengine
import numpy as np
import torch
from mmengine.config import Config

from mmpretrain import get_model
from mpnn.datasets.fused_data_utils import featurize, parse_PDB
from mpnn.datasets.residue_mapping import ALPHABET, restype_INTtoSTR

seed = 0

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


class TestLigandMPNN(TestCase):

    @classmethod
    def setUpClass(self):
        print('start setup')
        config = 'mpnn/configs/ligand_mpnn.py'
        ckpt = 'output/s25_r010_t300_p.pt'
        self.device = 'cpu'
        self.model = get_model(config, ckpt, device=self.device)
        self.model.eval()

    def _load_data(self):
        device = self.device
        pdb_path = ['tests/data/1BC8.pdb']

        fixed_residues = []
        redesigned_residues = []

        protein_dict, backbone, other_atoms, icodes, water_atoms = parse_PDB(
            pdb_path,
            device=self.device,
            atom_context_num=25,  # 25
            chains='',    # ''
            parse_all_atoms=False
        )

        # make chain_letter + residue_idx + insertion_code mapping to integers
        R_idx_list = list(protein_dict["R_idx"].cpu().numpy())
        chain_letters_list = list(protein_dict["chain_letters"])
        encoded_residues = []
        for i in range(len(R_idx_list)):
            tmp = str(chain_letters_list[i]) + str(R_idx_list[i]) + icodes[i]
            encoded_residues.append(tmp)

        omit_AA_per_residue = torch.zeros(
            [len(encoded_residues), 21], device=self.device, dtype=torch.float32
        )

        fixed_positions = torch.tensor(
            [int(item not in fixed_residues) for item in encoded_residues],
            device=device,
        )

        buried_positions = torch.zeros_like(fixed_positions)
        interface_positions = torch.zeros_like(fixed_positions)
        protein_dict["membrane_per_residue_labels"] = 2 * buried_positions * (
            1 - interface_positions
        ) + 1 * interface_positions * (1 - buried_positions)
        protein_dict["pssm"] = torch.zeros(
            [fixed_positions.shape[0], 20], device=device
        )
        chains_to_design_list = protein_dict["chain_letters"]
        chain_mask = torch.tensor(
            np.array(
                [
                    item in chains_to_design_list
                    for item in protein_dict["chain_letters"]
                ],
                dtype=np.int32,
            ),
            device=device,
        )
        protein_dict["chain_mask"] = chain_mask
        protein_dict["side_chain_mask"] = protein_dict["chain_mask"]

        # if other_atoms:
        #     other_bfactors = other_atoms.getBetas()
        #     other_atoms.setBetas(other_bfactors * 0.0)

        feature_dict = featurize(
            protein_dict,
            cutoff_for_score=8.,
            use_atom_context=1,
            number_of_ligand_atoms=25,
            model_type='ligand_mpnn',
        )
        feature_dict["batch_size"] = 1
        B, L, _, _ = feature_dict["X"].shape  # batch size should be 1 for now.
        # ----


        omit_AA_list = ''
        omit_AA = torch.tensor(
            np.array([AA in omit_AA_list for AA in ALPHABET]).astype(np.float32),
            device=device,
        )
        omit_AA_per_residue = torch.zeros(
            [len(encoded_residues), 21], device=device, dtype=torch.float32
        )
        bias_AA = torch.zeros([21], device=device, dtype=torch.float32)
        bias_AA_per_residue = torch.zeros(
            [len(encoded_residues), 21], device=device, dtype=torch.float32
        )

        # add additional keys to the feature dictionary
        feature_dict["temperature"] = 0.1
        feature_dict["bias"] = (
            (-1e8 * omit_AA[None, None, :] + bias_AA).repeat([1, L, 1])
            + bias_AA_per_residue[None]
            - 1e8 * omit_AA_per_residue[None]
        )
        remapped_symmetry_residues = [[]]
        symmetry_weights = [[]]
        feature_dict["symmetry_residues"] = remapped_symmetry_residues
        feature_dict["symmetry_weights"] = symmetry_weights

        return feature_dict


    @torch.no_grad()
    def _predict(self):
        feature_dict = self._load_data()
        # for _ in range(args.number_of_batches):
        torch.random.manual_seed(seed)
        feature_dict["randn"] = torch.randn(
            [feature_dict["batch_size"], feature_dict["mask"].shape[1]],
            device=self.device,
        )

        output_dict = self.model(feature_dict, mode='predict')
        pred_seq_idx = output_dict['S'][0].cpu().tolist()
        pred_seq = ''.join([restype_INTtoSTR[AA] for AA in pred_seq_idx])
        return pred_seq

    def test_predict_result(self):
        pred_seq = self._predict()
        gt_seq = 'AKSPISLHEFLLELLSDPRYDGLIRWTSDNGEFQLVDPEGVAKLWGEVKGKPKMNWKNLHKALRGYKKKKIIRTVKGKPYQYQFVNYPEFLHP'
        print('  gt_seq:\t', gt_seq)
        print('pred_seq:\t', pred_seq)
        acc_rate = sum([1 for i, j in zip(pred_seq, gt_seq) if i == j]) / len(gt_seq)
        print('acc_rate:\t', acc_rate)
        self.assertGreater(acc_rate, 0.45)


if __name__ == '__main__':
    unittest.main()
