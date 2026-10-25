from __future__ import print_function

import gzip
import random
import string
import sys

import biotite.structure as struc
import biotite.structure.io.pdb as pdbb
import biotite.structure.io.pdbx as pdbx
import numpy as np
import torch
import torch.nn as nn
import torch.utils
from cifutils.utils.io_utils import to_cif_file, to_pdb_buffer

from mpnn.datasets.residue_mapping import BACKBONE_ATOMS, PROTEIN_ALL_ATOMS


def make_pair_bias(chain_labels, R_idx, pair_bias_AA):
    d_chains = ((chain_labels[:, None] - chain_labels[None,:])==0).long()
    upper_diag = (R_idx[1:]-R_idx[:-1]==1).long()
    lower_diag = (R_idx[:-1]-R_idx[1:]==-1).long()
    u_mask = torch.diag(upper_diag, 1)
    u_mask = u_mask*d_chains
    l_mask = torch.diag(lower_diag, -1)
    l_mask = l_mask*d_chains
    pair_bias = (u_mask[None,:,None,:,None]*pair_bias_AA[None,None,:,None,:]+l_mask[None,:,None,:,None]*torch.transpose(pair_bias_AA, 0, 1)[None,None,:,None,:]) #[1,L,21,L,21]
    return pair_bias

def get_seq_rec(S: torch.Tensor, 
                S_pred: torch.Tensor, 
                mask: torch.Tensor):
    """
    S : true sequence shape=[batch, length]
    S_pred : predicted sequence shape=[batch, length]
    mask : mask to compute average over the region shape=[batch, length]

    average : averaged sequence recovery shape=[batch]
    """
    match = (S == S_pred)
    average = torch.sum(match*mask, dim=-1)/torch.sum(mask, dim=-1)
    return average

def get_score(S: torch.Tensor, 
              log_probs: torch.Tensor, 
              mask: torch.Tensor):
    """
    S : true sequence shape=[batch, length]
    log_probs : predicted sequence shape=[batch, length]
    mask : mask to compute average over the region shape=[batch, length]

    average_loss : averaged categorical cross entropy (CCE) [batch]
    loss_per_resdue : per position CCE [batch, length]
    """
    S_one_hot = torch.nn.functional.one_hot(S, 21)
    loss_per_residue = -(S_one_hot*log_probs).sum(-1) #[B, L]
    average_loss = torch.sum(loss_per_residue*mask, dim=-1)/(torch.sum(mask, dim=-1)+1e-8)
    return average_loss, loss_per_residue


def save_output_structure(save_path: str, 
                   X: np.ndarray, 
                   X_m: np.ndarray, 
                   b_factors: np.ndarray,
                   R_idx: np.ndarray, 
                   chain_letters: np.ndarray, 
                   S: np.ndarray, 
                   other_atoms=None, 
                   icodes=None,
                   **kwargs,
                   ) -> None:
    """
    save_path : path where the output file will be written to
    X : protein atom xyz coordinates shape=[length, 14, 3]
    X_m : protein atom mask shape=[length, 14]
    b_factors: shape=[length, 14]
    R_idx: protein residue indices shape=[length]
    chain_letters: protein chain letters shape=[length]
    S : protein amino acid sequence shape=[length]
    other_atoms: AtomArray containing any atoms outside of canonical amino acids
    icodes: a list of insertion codes for the PDB; e.g. antibody loops
    """

    restype_1to3 = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL', 'X': 'UNK'}
    restype_INTtoSTR = {0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P', 13: 'Q', 14: 'R', 15: 'S', 16: 'T', 17: 'V', 18: 'W', 19: 'Y', 20: 'X'}
    restype_name_to_atom14_names = {'ALA': ['N', 'CA', 'C', 'O', 'CB', '', '', '', '', '', '', '', '', ''], 
                                    'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2', '', '', ''], 
                                    'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2', '', '', '', '', '', ''], 
                                    'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2', '', '', '', '', '', ''], 
                                    'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG', '', '', '', '', '', '', '', ''], 
                                    'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2', '', '', '', '', ''], 
                                    'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2', '', '', '', '', ''], 
                                    'GLY': ['N', 'CA', 'C', 'O', '', '', '', '', '', '', '', '', '', ''], 
                                    'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2', '', '', '', ''], 
                                    'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '', '', '', '', '', ''], 
                                    'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', '', '', '', '', '', ''], 
                                    'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', '', '', '', '', ''], 
                                    'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE', '', '', '', '', '', ''], 
                                    'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', '', '', ''], 
                                    'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', '', '', '', '', '', '', ''], 
                                    'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG', '', '', '', '', '', '', '', ''], 
                                    'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '', '', '', '', '', '', ''], 
                                    'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'NE1', 'CZ2', 'CZ3', 'CH2'], 
                                    'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', '', ''], 
                                    'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '', '', '', '', '', '', ''], 
                                    'UNK': ['', '', '', '', '', '', '', '', '', '', '', '', '', '']}

    S_str = [restype_1to3[AA] for AA in [restype_INTtoSTR[AA] for AA in S]]
    
    X_list = []
    b_factor_list = []
    atom_name_list = []
    element_name_list = []
    residue_name_list = []
    residue_number_list = []
    chain_id_list = []
    icodes_list = []
    for i, AA in enumerate(S_str):
        sel = (X_m[i].astype(np.int32) == 1)
        total = np.sum(sel)
        tmp = np.array(restype_name_to_atom14_names[AA])[sel]
        X_list.append(X[i][sel])
        b_factor_list.append(b_factors[i][sel])
        atom_name_list.append(tmp)
        element_name_list += [AA[:1] for AA in list(tmp)]
        residue_name_list += total*[AA]
        residue_number_list += total*[R_idx[i]]
        chain_id_list += total*[chain_letters[i]]
        icodes_list += total*[icodes[i]]

    X_stack = np.concatenate(X_list, 0)
    b_factor_stack = np.concatenate(b_factor_list, 0)
    atom_name_stack = np.concatenate(atom_name_list, 0)
    
    n_atoms = X_stack.shape[0]
    atom_array = struc.AtomArray(n_atoms)
    atom_array.coord = X_stack
    atom_array.atom_name = atom_name_stack
    atom_array.res_id = residue_number_list
    atom_array.res_name = residue_name_list
    atom_array.chain_id = chain_id_list
    atom_array.element = element_name_list
    atom_array.b_factor = b_factor_stack
    atom_array.occcupancy = np.ones([X_stack.shape[0]])
    atom_array.ins_code = icodes_list
    atom_array.hetero = np.zeros([X_stack.shape[0]])

    if "force_hetatm" in kwargs and not kwargs["force_hetatm"]:
        print(
            "WARNING: The force_hetatm argument is deprecated and no longer has an effect. All atoms will be"
            "written correctly according to their true heteroatom status."
        )

    if other_atoms:
        atom_array = atom_array + other_atoms

    if ".cif" in save_path:
        to_cif_file(atom_array, path=save_path, file_type="cif", include_entity_poly=False)
    elif ".pdb" in save_path:
        pdb_buffer = to_pdb_buffer(atom_array)
        with open(save_path, "w") as f:
            f.write(pdb_buffer.getvalue())
    else:
        raise ValueError("Unrecognized file type. Please specify a .cif or a .pdb file.")


def parse_a3m(filename, maxseq=101):
    msa = []
    ins = []

    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    if filename.split('.')[-1] == 'gz':
        fp = gzip.open(filename, 'rt')
    else:
        fp = open(filename, 'r')

    for line in fp:
        # skip labels
        if line[0] == '>':
            continue

        # remove right whitespaces
        line = line.rstrip()

        if len(line) == 0:
            continue

        # remove lowercase letters and append to MSA
        msa.append(line.translate(table))

        # sequence length
        L = len(msa[-1])

        # 0 - match or gap; 1 - insertion
        a = np.array([0 if c.isupper() or c=='-' else 1 for c in line])
        i = np.zeros((L))

        if np.sum(a) > 0:
            # positions of insertions
            pos = np.where(a==1)[0]

            # shift by occurrence
            a = pos - np.arange(pos.shape[0])

            # position of insertions in cleaned sequence
            # and their length
            pos,num = np.unique(a, return_counts=True)

            # append to the matrix of insetions
            i[pos] = num

        ins.append(i)

        if (len(msa) >= maxseq):
            break

    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in msa], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    # treat all unknown characters as gaps
    msa[msa > 20] = 20

    ins = np.array(ins, dtype=np.uint8)

    return msa,ins


def subsample_msa(msa, ins, maxseq, sub_type="UNI"):

    nr, nc = msa.shape

    # trim if too many sequences
    if nr > maxseq:
        nr = maxseq

    # do not subsample shallow MSAs
    if nr < 3:
        return msa,ins

    # number of sequences to subsample
    if sub_type == 'LOG':
        n = np.maximum(int(10.0**np.random.uniform(np.log10(nr))-10.0), 0)
    elif sub_type == 'UNI':
        n = np.maximum(int(np.random.uniform(9, nr-1))-1, 0)
    else:
        sys.exit('Error: wrong MSA subsampling mode:', sub_type)

    if n <= 0:
        return np.reshape(msa[0],[1,nc]),np.reshape(ins[0],[1,nc])

    # n unique indices
    sample = sorted(random.sample(range(1, nr-1), n))

    # stack first sequence with n randomly selected ones
    msa_new = np.vstack([msa[0][None,:], msa[1:][sample]])
    ins_new = np.vstack([ins[0][None,:], ins[1:][sample]])

    return msa_new.astype(np.uint8),ins_new.astype(np.uint8)

def get_aligned_coordinates(protein_atoms, 
                            CA_dict: dict, 
                            atom_name: str):
    """
    protein_atoms: atom group
    CA_dict: mapping between chain_residue_idx_icodes and integers
    atom_name: atom to be parsed; e.g. CA
    """
    #atom_atoms = protein_atoms.select(f'name {atom_name}')
    atom_atoms = protein_atoms[protein_atoms.atom_name == atom_name]

    if atom_atoms != None:
        atom_coords = atom_atoms.coord
        atom_resnums = atom_atoms.res_id
        atom_chain_ids = atom_atoms.chain_id
        atom_icodes = atom_atoms.ins_code
    
    atom_coords_= np.zeros([len(CA_dict),3], np.float32)
    atom_coords_m= np.zeros([len(CA_dict)], np.int32)
    if atom_atoms != None:
        for i in range(len(atom_resnums)):
            code=atom_chain_ids[i]+"_"+str(atom_resnums[i])+"_"+atom_icodes[i]
            if code in list(CA_dict):
                atom_coords_[CA_dict[code],:] = atom_coords[i]
                atom_coords_m[CA_dict[code]] = 1
    return atom_coords_, atom_coords_m


def get_nearest_neighbours(CB, mask, Y, Y_t, Y_m, number_of_ligand_atoms):
    device=CB.device
    mask_CBY = mask[:,None]*Y_m[None,:] #[A,B]
    L2_AB = torch.sum((CB[:,None,:]-Y[None,:,:])**2,-1)
    L2_AB = L2_AB*mask_CBY + (1-mask_CBY)*1000.0

    nn_idx = torch.argsort(L2_AB, -1)[:,:number_of_ligand_atoms]
    L2_AB_nn = torch.gather(L2_AB, 1, nn_idx)
    D_AB_closest = torch.sqrt(L2_AB_nn[:,0])

    Y_r = Y[None,:,:].repeat(CB.shape[0],1,1)
    Y_t_r = Y_t[None,:].repeat(CB.shape[0],1)
    Y_m_r = Y_m[None,:].repeat(CB.shape[0],1)

    Y_tmp = torch.gather(Y_r, 1, nn_idx[:,:,None].repeat(1,1,3))
    Y_t_tmp = torch.gather(Y_t_r, 1, nn_idx)
    Y_m_tmp = torch.gather(Y_m_r, 1, nn_idx)

    Y = torch.zeros([CB.shape[0], number_of_ligand_atoms, 3], dtype=torch.float32, device=device)
    Y_t = torch.zeros([CB.shape[0], number_of_ligand_atoms], dtype=torch.int32, device=device)
    Y_m = torch.zeros([CB.shape[0], number_of_ligand_atoms], dtype=torch.int32, device=device)

    num_nn_update = Y_tmp.shape[1]
    Y[:,:num_nn_update] = Y_tmp
    Y_t[:,:num_nn_update] = Y_t_tmp
    Y_m[:,:num_nn_update] = Y_m_tmp

    return Y, Y_t, Y_m, D_AB_closest


def featurize(input_dict, cutoff_for_score=8.0, use_atom_context=True, number_of_ligand_atoms=16, model_type="protein_mpnn"):
    output_dict = {}
    if model_type == "msa_mpnn":
        output_dict["MSA"] = input_dict["MSA"][None,]
        output_dict["MSA_confidence"] = input_dict["MSA_confidence"][None,]
    elif model_type == "ligand_mpnn":
        mask=input_dict["mask"]
        Y=input_dict["Y"]
        Y_t=input_dict["Y_t"]
        Y_m=input_dict["Y_m"]
        N=input_dict["X"][:,0,:]
        CA=input_dict["X"][:,1,:]
        C=input_dict["X"][:,2,:]
        b = CA - N
        c = C - CA
        a = torch.cross(b, c, axis=-1)
        CB = -0.58273431*a + 0.56802827*b - 0.54067466*c + CA
        Y, Y_t, Y_m, D_XY  = get_nearest_neighbours(CB, mask, Y, Y_t, Y_m, number_of_ligand_atoms)
        mask_XY = (D_XY < cutoff_for_score)*mask*Y_m[:,0]
        output_dict["mask_XY"] = mask_XY[None,]
        output_dict["side_chain_mask"] = input_dict["side_chain_mask"][None,]
        output_dict["Y"] = Y[None,]
        output_dict["Y_t"] = Y_t[None,]
        output_dict["Y_m"] = Y_m[None,]
        if not use_atom_context:
            output_dict["Y_m"] = 0.0*output_dict["Y_m"]   
    elif model_type == "per_residue_label_membrane_mpnn" or model_type == "global_label_membrane_mpnn":
        output_dict["membrane_per_residue_labels"] = input_dict["membrane_per_residue_labels"][None,]
    elif model_type == "pssm_mpnn":
        output_dict["pssm"] = input_dict["pssm"][None,] 

    R_idx_list = []
    count=0
    R_idx_prev=-100000
    for R_idx in list(input_dict["R_idx"]):
        if R_idx_prev==R_idx:
            count+=1
        R_idx_list.append(R_idx+count)
        R_idx_prev=R_idx
    R_idx_renumbered = torch.tensor(R_idx_list, device=R_idx.device)
    output_dict["R_idx"] = R_idx_renumbered[None,]
    output_dict["R_idx_original"] = input_dict["R_idx"][None,]
    output_dict["chain_labels"] = input_dict["chain_labels"][None,]
    output_dict["S"] = input_dict["S"][None,]
    output_dict["chain_mask"] = input_dict["chain_mask"][None,]
    output_dict["mask"] = input_dict["mask"][None,]

    output_dict["X"] = input_dict["X"][None,]

    output_dict["xyz_37"] = input_dict["xyz_37"][None,]
    output_dict["xyz_37_m"] = input_dict["xyz_37_m"][None,]

    return output_dict
