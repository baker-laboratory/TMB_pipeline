import torch
import numpy as np
import random
from data_utils import featurize
from model_utils import ProteinMPNN

#Amino acid alphabet
restype_1to3 = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}
restype_STRtoINT = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20}
restype_INTtoSTR = {0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P', 13: 'Q', 14: 'R', 15: 'S', 16: 'T', 17: 'V', 18: 'W', 19: 'Y', 20: 'X'}
alphabet = list(restype_STRtoINT)
#Chemical element type alphabet
element_list = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mb', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Uut', 'Fl', 'Uup', 'Lv', 'Uus', 'Uuo']
element_list = [item.upper() for item in element_list]
element_dict = dict(zip(element_list, range(1,len(element_list))))

#Choose your device GPU/CPU
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

#Choose your MPNN model
#model_type="protein_mpnn"
#model_type="ligand_mpnn"
#model_type="per_residue_label_membrane_mpnn"
#model_type="global_label_membrane_mpnn"
#model_type="soluble_mpnn"
#model_type="pssm_mpnn"
model_type="antibody_mpnn"

#To use protein side chains and fake ligands or not when using ligand_mpnn
ligand_mpnn_use_side_chain_context = False

#number of samples to batch (only if using GPU)
batch_size = 2
#number of times to call the model
number_of_batches = 2
#decoding temperature
temperature=0.1
seed = 1

#fix seeds
if seed:
    seed=seed
else:
    seed=int(np.random.randint(0, high=99999, size=1, dtype=int)[0])

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

#to use or not ligand context
ligand_mpnn_use_atom_context = True #to use ligand atoms or not

#provide path to model weights
checkpoint_protein_mpnn="/databases/mpnn/vanilla_model_weights/v_48_020.pt"
checkpoint_ligand_mpnn="/databases/mpnn/ligand_mpnn_model_weights/v_32_010.pt"
checkpoint_per_residue_label_membrane_mpnn="/databases/mpnn/tmd_per_residue_weights/tmd_v_48_020.pt"
checkpoint_global_label_membrane_mpnn="/databases/mpnn/tmd_weights/v_48_020.pt"
checkpoint_soluble_mpnn="/databases/mpnn/no_transmembrane/v_48_020.pt"
checkpoint_pssm_mpnn="/databases/mpnn/pssm_model_weights/v_48_020.pt"
checkpoint_antibody_mpnn="/databases/mpnn/antibody_mpnn_model_weights/v_48_020_bias_005.pt"

if model_type == "protein_mpnn":
    checkpoint_path = checkpoint_protein_mpnn
elif model_type == "ligand_mpnn":
    checkpoint_path = checkpoint_ligand_mpnn
elif model_type == "per_residue_label_membrane_mpnn":
    checkpoint_path = checkpoint_per_residue_label_membrane_mpnn
elif model_type == "global_label_membrane_mpnn":
    checkpoint_path = checkpoint_global_label_membrane_mpnn
elif model_type == "soluble_mpnn":
    checkpoint_path = checkpoint_soluble_mpnn
elif model_type == "pssm_mpnn":
    checkpoint_path = checkpoint_pssm_mpnn
elif model_type == "antibody_mpnn":
    checkpoint_path = checkpoint_antibody_mpnn
checkpoint = torch.load(checkpoint_path, map_location=device)

if model_type == "ligand_mpnn":
    atom_context_num = 16 #TODO: load from weights
    k_neighbors=32
    ligand_mpnn_use_side_chain_context = ligand_mpnn_use_side_chain_context
elif model_type == "antibody_mpnn":
    atom_context_num = 1
    ligand_mpnn_use_side_chain_context = 0
    k_neighbors=48
else:
    atom_context_num = 1
    ligand_mpnn_use_side_chain_context = 0
    k_neighbors=checkpoint["num_edges"]

model = ProteinMPNN(node_features=128,
                edge_features=128,
                hidden_dim=128,
                num_encoder_layers=3,
                num_decoder_layers=3,
                k_neighbors=k_neighbors,
                device=device,
                atom_context_num=atom_context_num,
                model_type=model_type,
                ligand_mpnn_use_side_chain_context=ligand_mpnn_use_side_chain_context)

model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

#Load protein coordinates etc to populate protein_dict

#random example
L = 66 #protein residues parsed
M = 10 #number of ligand atoms parsed
protein_dict = {"X": torch.randn([L,4,3], device=device, dtype=torch.float32), #N, Ca, C, O coordinates, xyz
                "R_idx": torch.arange(start=0, end=L, device=device, dtype=torch.int64), #residue index
                "chain_labels": torch.zeros([L], device=device, dtype=torch.int64), #label for chain [0,0,0,1,1,1,1,1,] for chain A and B
                "S": torch.zeros([L], device=device, dtype=torch.int64), #input sequence encoded using restype_STRtoINT
                "mask": torch.ones([L], device=device, dtype=torch.float32), #mask out specific residues, e.g. missing backbone
                "chain_mask": torch.ones([L], device=device, dtype=torch.float32), #specify which residues need to be redesigned
                "side_chain_mask": torch.ones([L], device=device, dtype=torch.float32), #which residue side chains to use
                "membrane_per_residue_labels": torch.ones([L], device=device, dtype=torch.int64), #global_label_membrane_mpnn -  1 - transmembrane, 0 - soluble; per_residue_label_membrane_mpnn - 2 buried, 1 interface, 0 otherwise
                "pssm": torch.ones([L,20], device=device, dtype=torch.float32), #pssm log odds with alphabet ARNDCQEGHILKMFPSTWYV (different from ProteinMPNN alpghabet!)
                "xyz_37": torch.zeros([L,37,3], device=device, dtype=torch.float32), #if using side chains as context for ligandMPNN
                "xyz_37_m": torch.zeros([L,37], device=device, dtype=torch.float32), #if using side chains as context for ligandMPNN
                "Y": torch.zeros([M,3], device=device, dtype=torch.float32), #ligand atom xyz coords
                "Y_t": torch.zeros([M], device=device, dtype=torch.float32), #ligand atom chemical element types; element_list alphabet
                "Y_m": torch.zeros([M], device=device, dtype=torch.float32), #ligand atom mask
                "bias": torch.zeros([L,21], device=device), #AA bias per position; real number
                "pair_bias": torch.zeros([L,21,L,21], device=device), #pair AA bias for position i and j; real number
                "symmetry_residues": [[]], #specify which residues need to be decoded together [[0,66], [1,67], [2,68], [3,4,5,69,70,71]] 
                "symmetry_weights": [[]], #specify weights when combining logits [[1.0,1.0], [1.0,1.0], [1.0,1.0], [0.5,0.3,-0.8,1.4,4.5,1.1]] 
                }

with torch.no_grad():
    feature_dict = featurize(protein_dict,
                             cutoff_for_score=8.0, 
                             use_atom_context=ligand_mpnn_use_atom_context)
    feature_dict["batch_size"] = batch_size
    B, L, _, _ = feature_dict["X"].shape
    feature_dict["temperature"] = temperature
    feature_dict["bias"] = protein_dict["bias"][None,]
    feature_dict["pair_bias"] = protein_dict["pair_bias"][None,]
    feature_dict["symmetry_residues"] = protein_dict["symmetry_residues"]
    feature_dict["symmetry_weights"] = protein_dict["symmetry_weights"]

    output_list = []
    for idx in range(number_of_batches):
        feature_dict["randn"] = torch.randn([feature_dict["batch_size"], feature_dict["mask"].shape[1]], device=device)
        output_dict = model.sample(feature_dict)
        output_list.append(output_dict)
        #output_dict = {"S": S, "sampling_probs": all_probs, "log_probs": all_log_probs, "decoding_order": decoding_order[0]}
        print(seed, model_type, _, output_dict["S"])
