import os
import sys
import unittest
import numpy as np
import torch

from mpnn.datasets.pdb_dataset import PDBDataset
from mpnn.datasets.transforms import (
    Compose,
    LoadFile,
    GetAssembly,
    ExcludeByResName,
    GetChainInfo,
    SeperateProteinLigand,
    BuildMPNNInput,
)
from mpnn.models.train_data_utils import train_featurize

# Import the full models from their respective modules using unique names.
from mpnn.models.protein_mpnn import ProteinMPNN as TrainingProteinMPNN
from model_utils import ProteinMPNN as InferenceProteinMPNN


class TestFullModelsFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ------------------ Setup Dataset and Transforms ------------------
        # Use the same dataset for the training samples.
        cls.dataset = PDBDataset(
            cif_path="tests/data",
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
        # Transforms for inference that produce a feature dictionary from a CIF file.
        cls.transforms = Compose([
            LoadFile(),
            GetAssembly(),
            ExcludeByResName(exclude_res_list=["HOH", "NA", "CL", "K", "BR"]),
            GetChainInfo(),
            SeperateProteinLigand(),
            BuildMPNNInput(lig_nn_num=25),
        ])

        # ------------------ Initialize the Full Models ------------------
        # Training model
        cls.training_model = TrainingProteinMPNN(
            model_type="ligand_mpnn",
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=32,
            dropout=0.0,
            augment_eps=0.0,
            atom_context_num=16,
            ligand_mpnn_use_side_chain_context=False
        )
        # Inference model
        cls.inference_model = InferenceProteinMPNN(
            model_type="ligand_mpnn",
            num_letters=21,
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            vocab=21,
            k_neighbors=32,
            augment_eps=0.0,
            dropout=0.0,
            device=cls.device,
            atom_context_num=16,
            ligand_mpnn_use_side_chain_context=False
        )

        # ------------------ Load Trained Weights ------------------
        # Provide a generic file path; update this path to your actual checkpoint file.
        checkpoint_path = "/home/cfmelend/projects/rotamer_aware/mpnn_checkpoints/refactor_test/model_epoch_150_step_215009.pt"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}. Please update the file path to load your trained weights.")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        # The checkpoint is expected to have a key 'model_state_dict'; if not, assume checkpoint itself is the state_dict.
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        cls.inference_model.load_state_dict(state_dict, strict=False)
        cls.training_model.load_state_dict(state_dict, strict=False)
        cls.inference_model.eval()
        cls.training_model.eval()

    def test_features_consistency(self):
        device = self.device
        self.inference_model.to(device)
        self.training_model.to(device)
        ##########################################
        # 1. Inference Pipeline
        ##########################################
        inf_input = {
            "cif_path": os.path.join("tests", "data", "1bc8.cif"),
            "assembly_id": "1"
        }
        # Get feature dictionary from inference
        feature_dict = self.transforms(inf_input)
        # Convert any NumPy arrays to torch tensors on the current device.
        for k, v in feature_dict.items():
            if isinstance(v, np.ndarray):
                if not np.issubdtype(v.dtype, np.str_):
                    v = torch.from_numpy(v)[None, ].to(device)
            elif isinstance(v, torch.Tensor):
                v = v[None, ].to(device)
            feature_dict[k] = v

        feature_dict['xyz_37'] = feature_dict['X']
        feature_dict['xyz_37_m'] = feature_dict['X_m']
        feature_dict['X'] = feature_dict['X'][:, :, :4]
        feature_dict['X_m'] = feature_dict['X_m'][:, :, :4]
        # Ensure chain letter info exists; if absent, assume single chain "A".
        if "chain_letters" not in feature_dict:
            L = feature_dict["X"].shape[0]
            feature_dict["chain_letters"] = ["A"] * L
        # Build chain and side-chain masks as in run.py
        chain_letters = feature_dict["chain_letters"]
        chain_mask = torch.tensor(
            np.array([item in chain_letters for item in chain_letters], dtype=np.int32),
            device=device
        )
        feature_dict["chain_mask"] = chain_mask
        feature_dict["side_chain_mask"] = chain_mask

        # Get features from inference
        with torch.no_grad():
            features_inf = self.inference_model.features(feature_dict)

        ##########################################
        # 2. Training Pipeline
        ##########################################
        # Define a simple dataset that always returns the same sample.
        class ConstantSampleDataset(torch.utils.data.Dataset):
            def __init__(self, loader_func):
                self.loader_func = loader_func

            def __len__(self):
                return 1

            def __getitem__(self, idx):
                sample_dict, seq_len = self.loader_func("1bc8", "1")
                # Convert numpy arrays to tensors
                processed_dict = {
                    k: torch.from_numpy(v) if isinstance(v, np.ndarray) else torch.tensor(v)
                    for k, v in sample_dict.items()
                }
                return [[processed_dict, seq_len]]

        # Create the dataset and DataLoader.
        constant_dataset = ConstantSampleDataset(self.dataset.loader)
        train_loader = torch.utils.data.DataLoader(
            dataset=constant_dataset,
            batch_size=1,
            collate_fn=torch.utils.data.default_collate,
            num_workers=1
        )

        batch = next(iter(train_loader))
        X_train, X_m_train, mask_train, S_train, R_idx_train, chain_labels_train, \
            Y_train, Y_t_train, Y_m_train = train_featurize(batch, mode="all", num_nn=25)

        # Get features from train
        features_train = self.training_model.features(
            X_train, Y_train, Y_m_train, Y_t_train, mask_train, R_idx_train,
            chain_labels_train,
            None,
            X_m=X_m_train,
            use_sc=False
        )

        ##########################################
        # 3. Compare the Graph Features
        ##########################################
        expected_keys = ["V", "E", "E_idx", "Y_nodes", "Y_edges", "Y_m"]
        
        print("\nComparing inference and training features:")
        self.assertEqual(len(features_inf), len(features_train),
                         f"Mismatch in number of graph feature outputs: {len(features_inf)} vs {len(features_train)}")
        
        for idx, key in enumerate(expected_keys):
            inf_shape = features_inf[idx][0].shape
            train_shape = features_train[idx][0].shape
            print(f"\n{key}:")
            print(f"  Inference shape: {inf_shape}")
            print(f"  Training shape:  {train_shape}")
            
            self.assertEqual(inf_shape, train_shape,
                           f"Shape mismatch for {key}: {inf_shape} vs {train_shape}")
            
            # Compare actual values
            inf_values = features_inf[idx][0].cpu().detach().numpy()
            train_values = features_train[idx][0].cpu().detach().numpy()
            
            try:
                if np.issubdtype(inf_values.dtype, np.floating):
                    np.testing.assert_allclose(
                        inf_values,
                        train_values,
                        atol=1e-5,
                        err_msg=f"Value mismatch for {key}"
                    )
                else:
                    np.testing.assert_array_equal(
                        inf_values,
                        train_values,
                        err_msg=f"Value mismatch for {key}"
                    )
                # if key == "V" or key == "E" or key == "Y_nodes":
                #     print(f"  Inference values: {inf_values[:2]}")
                #     print(f"  Training values: {train_values[:2]}")
                print(f"  Values match within tolerance")
            except AssertionError as e:
                print(f"  Value mismatch: {str(e)}")
                raise


if __name__ == "__main__":
    unittest.main()
