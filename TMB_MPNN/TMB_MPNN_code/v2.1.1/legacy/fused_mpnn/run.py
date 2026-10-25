import argparse
import copy
import json
import os.path
import random
import sys

import numpy as np
import torch
from prody import writePDB

from data_utils import (
    featurize,
    get_score,
    get_seq_rec,
    make_pair_bias,
    parse_a3m,
    parse_PDB,
    subsample_msa,
    write_full_PDB,
)
from model_utils import ProteinMPNN
from sc_utils import Packer, pack_side_chains


def parse_enhance_args(args):
    """Update the checkpoint path based on the enhancer."""
    if not args.enhance:
        return args
    enhancer = args.enhance.lower()
    if not enhancer.startswith("plddt"):
        raise ValueError("Currently only support enhance pLDDT, reach out if you need more.")
    assert args.model_type == "ligand_mpnn", "Currently only support ligand_mpnn."
    args.checkpoint_ligand_mpnn = f'/net/scratch/faxue/share/mpnn_enhancer/{enhancer}.pth'
    return args


def main(args):

    restype_1to3 = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL', 'X': 'UNK'}
    restype_STRtoINT = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20}
    restype_INTtoSTR = {0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P', 13: 'Q', 14: 'R', 15: 'S', 16: 'T', 17: 'V', 18: 'W', 19: 'Y', 20: 'X'}
    alphabet = list(restype_STRtoINT)

    #fix seeds
    if args.seed:
        seed=args.seed
    else:
        seed=int(np.random.randint(0, high=99999, size=1, dtype=int)[0])

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    #----

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    
    #make folders for outputs
    folder_for_outputs = args.out_folder
    base_folder = folder_for_outputs
    if base_folder[-1] != '/':
        base_folder = base_folder + '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder, exist_ok=True)
    if not os.path.exists(base_folder + 'seqs'):
        os.makedirs(base_folder + 'seqs', exist_ok=True)
    if not os.path.exists(base_folder + 'backbones'):
        os.makedirs(base_folder + 'backbones', exist_ok=True)
    if args.pack_side_chains:
        if not os.path.exists(base_folder + 'packed'):
            os.makedirs(base_folder + 'packed', exist_ok=True)
    if args.save_stats:
        if not os.path.exists(base_folder + 'stats'):
            os.makedirs(base_folder + 'stats', exist_ok=True)
    #----

     
    if args.model_type == "protein_mpnn":
        checkpoint_path = args.checkpoint_protein_mpnn
    elif args.model_type == "ligand_mpnn":
        checkpoint_path = args.checkpoint_ligand_mpnn
    elif args.model_type == "per_residue_label_membrane_mpnn":
        checkpoint_path = args.checkpoint_per_residue_label_membrane_mpnn
    elif args.model_type == "global_label_membrane_mpnn":
        checkpoint_path = args.checkpoint_global_label_membrane_mpnn
    elif args.model_type == "soluble_mpnn":
        checkpoint_path = args.checkpoint_soluble_mpnn
    elif args.model_type == "pssm_mpnn":
        checkpoint_path = args.checkpoint_pssm_mpnn
    elif args.model_type == "antibody_mpnn":
        checkpoint_path = args.checkpoint_antibody_mpnn
    elif args.model_type == "msa_mpnn":
        checkpoint_path = args.checkpoint_msa_mpnn
    else:
        print("Choose one of the available models")
        sys.exit()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if args.model_type == "ligand_mpnn":
        atom_context_num = 25 #TODO: load from weights
        k_neighbors=32
        ligand_mpnn_use_side_chain_context = args.ligand_mpnn_use_side_chain_context
    elif args.model_type == "antibody_mpnn" or args.model_type == "msa_mpnn":
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
                    model_type=args.model_type,
                    ligand_mpnn_use_side_chain_context=ligand_mpnn_use_side_chain_context)

    #load pretrained parameters

    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"
    model.to(device)
    model.eval()
    #----




    #load side chain packing model if needed
    if args.pack_side_chains:
        model_sc = Packer(node_features=128,
                        edge_features=128,
                        num_positional_embeddings=16,
                        num_chain_embeddings=16,
                        num_rbf=16,
                        hidden_dim=128,
                        num_encoder_layers=3,
                        num_decoder_layers=3,
                        atom_context_num=16,
                        lower_bound=0.0,
                        upper_bound=20.0,
                        top_k=32,
                        dropout=0.0,
                        augment_eps=0.0,
                        atom37_order=False,
                        device=device,
                        num_mix=3)

        checkpoint_sc = torch.load(args.checkpoint_path_sc, map_location=device)
        model_sc.load_state_dict(checkpoint_sc['model_state_dict'])
        model_sc.to(device)
        model_sc.eval()
    #----
    if args.pdb_path_multi:
        with open(args.pdb_path_multi, 'r') as fh:
            pdb_paths = list(json.load(fh))
    else:
        pdb_paths = [args.pdb_path]


    if args.fixed_residues_multi:
        with open(args.fixed_residues_multi, 'r') as fh:
            fixed_residues_multi = json.load(fh)
            for k, v in fixed_residues_multi.items():
                if isinstance(v, str):
                    fixed_residues_multi[k] = v.split()
    else:
        fixed_residues = [item for item in args.fixed_residues.split()]
        fixed_residues_multi = {}
        for pdb in pdb_paths:
            fixed_residues_multi[pdb] = fixed_residues

    if args.redesigned_residues_multi:
        with open(args.redesigned_residues_multi, 'r') as fh:
            redesigned_residues_multi = json.load(fh)
            for k, v in redesigned_residues_multi.items():
                if isinstance(v, str):
                    redesigned_residues_multi[k] = v.split()
    else:
        redesigned_residues = [item for item in args.redesigned_residues.split()]
        redesigned_residues_multi = {}
        for pdb in pdb_paths:
            redesigned_residues_multi[pdb] = redesigned_residues

    if args.fixed_residues_multi:
        for k, v in fixed_residues_multi.items():
            redesigned_residues_multi[k] = ""
    
    if args.redesigned_residues_multi:
        for k, v in redesigned_residues_multi.items():
            fixed_residues_multi[k] = ""
   

    #make amino acid bias array [21]
    bias_AA = torch.zeros([21], device=device, dtype=torch.float32)
    if args.bias_AA:
        tmp = [item.split(":") for item in args.bias_AA.split(",")]
        a1 = [b[0] for b in tmp]
        a2 = [float(b[1]) for b in tmp]
        for i, AA in enumerate(a1):
            bias_AA[restype_STRtoINT[AA]] = a2[i]

    #----
    if args.bias_AA_per_residue_multi:
        with open(args.bias_AA_per_residue_multi, 'r') as fh:
            bias_AA_per_residue_multi = json.load(fh) #{"pdb_path" : {"A12": {"G": 1.1}}}
    else:
        if args.bias_AA_per_residue:
            with open(args.bias_AA_per_residue, 'r') as fh:
                bias_AA_per_residue = json.load(fh) #{"A12": {"G": 1.1}}
            bias_AA_per_residue_multi={}
            for pdb in pdb_paths:
                bias_AA_per_residue_multi[pdb] = bias_AA_per_residue


    if args.omit_AA_per_residue_multi:
        with open(args.omit_AA_per_residue_multi, 'r') as fh:
            omit_AA_per_residue_multi = json.load(fh) #{"pdb_path" : {"A12": "PQR", "A13": "QS"}}
    else:
        if args.omit_AA_per_residue:
            with open(args.omit_AA_per_residue, 'r') as fh:
                omit_AA_per_residue = json.load(fh) #{"A12": "PG"}
            omit_AA_per_residue_multi={}
            for pdb in pdb_paths:
                omit_AA_per_residue_multi[pdb] = omit_AA_per_residue

    #make amino acid pair bias array [21, 21]
    pair_bias_AA = torch.zeros([21,21], dtype=torch.float32, device=device)
    if args.pair_bias_AA:
        tmp = [item.split(":") for item in args.pair_bias_AA.split(",")]
        a1 = [b[0][0] for b in tmp]
        a2 = [b[0][1] for b in tmp]
        a3 = [float(b[1]) for b in tmp]
        for i, AA in enumerate(a1):
            pair_bias_AA[restype_STRtoINT[AA], restype_STRtoINT[a2[i]]] = a3[i]
    #----
      
    #make array to indicate which amino acids need to be omitted [21]
    omit_AA_list = args.omit_AA
    omit_AA = torch.tensor(np.array([AA in omit_AA_list for AA in alphabet]).astype(np.float32), device=device)
    #----


    for pdb in pdb_paths:
        if args.verbose:
            print("Designing this PDB:", pdb)
        fixed_residues = fixed_residues_multi[pdb]
        redesigned_residues = redesigned_residues_multi[pdb]

        #parse PDB file
        protein_dict, backbone, other_atoms, icodes, water_atoms = parse_PDB(pdb,
                                                                            device=device, 
                                                                            atom_context_num=atom_context_num, 
                                                                            chains=args.parse_these_chains_only,
                                                                            parse_all_atoms=args.ligand_mpnn_use_side_chain_context or not args.repack_everything)
        #----
        #make chain_letter + residue_idx + insertion_code mapping to integers
        R_idx_list = list(protein_dict["R_idx"].cpu().numpy())
        chain_letters_list = list(protein_dict["chain_letters"])
        encoded_residues = []
        for i in range(len(R_idx_list)):
            tmp = str(chain_letters_list[i]) + str(R_idx_list[i]) + icodes[i]
            encoded_residues.append(tmp)
        encoded_residue_dict = dict(zip(encoded_residues, range(len(encoded_residues))))
        encoded_residue_dict_rev = dict(zip(list(range(len(encoded_residues))), encoded_residues))

        bias_AA_per_residue = torch.zeros([len(encoded_residues),21], device=device, dtype=torch.float32)
        if args.bias_AA_per_residue_multi or args.bias_AA_per_residue:    
            bias_dict = bias_AA_per_residue_multi[pdb]
            bias_hit_count = 0
            for residue_name, v1 in bias_dict.items():
                if residue_name in encoded_residues:
                    i1 = encoded_residue_dict[residue_name]
                    for amino_acid, v2 in v1.items():
                        if amino_acid in alphabet:
                            j1 = restype_STRtoINT[amino_acid]
                            bias_AA_per_residue[i1, j1] = v2
                            bias_hit_count += 1
            assert bias_hit_count > 0, f"--bias_AA_per_residue_multi or --bias_AA_per_residue does not match any residues in the PDB file"
        #----

        omit_AA_per_residue = torch.zeros([len(encoded_residues),21], device=device, dtype=torch.float32)
        if args.omit_AA_per_residue_multi or args.omit_AA_per_residue:    
            omit_dict = omit_AA_per_residue_multi[pdb]
            omit_hit_count = 0
            for residue_name, v1 in omit_dict.items():
                if residue_name in encoded_residues:
                    i1 = encoded_residue_dict[residue_name]
                    for amino_acid in v1:
                        if amino_acid in alphabet:
                            j1 = restype_STRtoINT[amino_acid]
                            omit_AA_per_residue[i1, j1] = 1.0
                            omit_hit_count += 1
            assert omit_hit_count > 0, f"--omit_AA_per_residue_multi or --omit_AA_per_residue does not match any residues in the PDB file"
        #----


        fixed_positions = torch.tensor([int(item not in fixed_residues) for item in encoded_residues], device=device)
        redesigned_positions = torch.tensor([int(item not in redesigned_residues) for item in encoded_residues], device=device)
        #----

        #specify which residues are buried for checkpoint_per_residue_label_membrane_mpnn model
        if args.transmembrane_buried:
            buried_residues = [item for item in args.transmembrane_buried.split()]
            buried_positions = torch.tensor([int(item in buried_residues) for item in encoded_residues], device=device)
        else:
            buried_positions = torch.zeros_like(fixed_positions)
        #----

        if args.transmembrane_interface:
            interface_residues = [item for item in args.transmembrane_interface.split()]
            interface_positions = torch.tensor([int(item in interface_residues) for item in encoded_residues], device=device)
        else:
            interface_positions = torch.zeros_like(fixed_positions)
        #----
        protein_dict["membrane_per_residue_labels"] = 2*buried_positions*(1-interface_positions) + 1*interface_positions*(1-buried_positions)

        if args.model_type == "global_label_membrane_mpnn":
            protein_dict["membrane_per_residue_labels"] = args.global_transmembrane_label + 0*fixed_positions
        
        #ARNDCQEGHILKMFPSTWYV - for 20 values - log odds ratios
        protein_dict["pssm"] = torch.zeros([fixed_positions.shape[0], 20], device=device)
        if args.model_type == "pssm_mpnn":
            if args.pssm_input:
                with open(args.pssm_input, 'r') as file:
                    pssm_dict = json.load(file)
                pssm_hit_count = 0
                for k, v in pssm_dict.items():
                    if k in encoded_residue_dict:
                        idx = encoded_residue_dict[k]
                        protein_dict["pssm"][idx,:] = torch.tensor(v, device=device, dtype=torch.float32)
                        pssm_hit_count += 1
                assert pssm_hit_count > 0, f"--pssm_input does not match any residues in the PDB file"

        #specify which chains need to be redesigned
        if type(args.chains_to_design) == str:
            chains_to_design_list = args.chains_to_design.split(",")
        else:
            chains_to_design_list = protein_dict["chain_letters"]
        chain_mask = torch.tensor(np.array([item in chains_to_design_list for item in protein_dict["chain_letters"]],dtype=np.int32), device=device)
        #----

        #create chain_mask to notify which residues are fixed (0) and which need to be designed (1)
        if redesigned_residues:
            protein_dict["chain_mask"] = chain_mask*(1-redesigned_positions)
        elif fixed_residues:
            protein_dict["chain_mask"] = chain_mask*fixed_positions
        else:
            protein_dict["chain_mask"] = chain_mask
        
        if args.verbose:
            PDB_residues_to_be_redesigned = [encoded_residue_dict_rev[item] for item in range(protein_dict["chain_mask"].shape[0]) if protein_dict["chain_mask"][item]==1]
            PDB_residues_to_be_fixed = [encoded_residue_dict_rev[item] for item in range(protein_dict["chain_mask"].shape[0]) if protein_dict["chain_mask"][item]==0]
            print("These residues will be redesigned: ", PDB_residues_to_be_redesigned)
            print("These residues will be fixed: ", PDB_residues_to_be_fixed)
        #----


        #create mask to specify for which residues side chain context can be used/do not repack those side chains
        protein_dict["side_chain_mask"] = protein_dict["chain_mask"]
        #----
    
        if args.msa_path and (args.model_type=="msa_mpnn"):
            msa, ins = parse_a3m(args.msa_path, maxseq=args.num_MSA_from_top)
            if args.num_MSA_to_use:
                msa, ins = subsample_msa(msa, ins, maxseq=args.num_MSA_to_use, sub_type="UNI")
            msa[0,:] = 0*msa[0,:]+20 #remove original sequence info
            msa = np.transpose(msa) #[L,N]

            protein_dict["MSA"] = torch.tensor(msa, device=device, dtype=torch.int64)
            protein_dict["MSA_confidence"] = torch.tensor(args.msa_confidence, device=device, dtype=torch.float32)

        #specify which residues are linked
        if args.symmetry_residues:
            symmetry_residues_list_of_lists = [x.split(',') for x in args.symmetry_residues.split('|')]
            remapped_symmetry_residues=[]
            for t_list in symmetry_residues_list_of_lists:
                tmp_list=[]
                for t in t_list:
                    tmp_list.append(encoded_residue_dict[t])
                remapped_symmetry_residues.append(tmp_list) 
        else:
            remapped_symmetry_residues=[[]]
        #----

        #specify linking weights
        if args.symmetry_weights:
            symmetry_weights = [[float(item) for item in x.split(',')] for x in args.symmetry_weights.split('|')]
        else:
            symmetry_weights = [[]]
        #----

        if args.homo_oligomer:
            if args.verbose:
                print("Designing HOMO-OLIGOMER")
            chain_letters_set = list(set(chain_letters_list))
            reference_chain = chain_letters_set[0]
            lc = len(reference_chain)
            per_chain_residue_indices_dict = {}
            for c in chain_letters_set:
                residue_indices = [item[lc:] for item in encoded_residues if item[:lc]==c]
                per_chain_residue_indices_dict[c] = residue_indices
            remapped_symmetry_residues=[]
            symmetry_weights = []
            for i, _ in enumerate(per_chain_residue_indices_dict[reference_chain]):
                tmp_list=[]
                tmp_w_list=[]
                for chain in chain_letters_set:
                    res = per_chain_residue_indices_dict[chain][i]
                    name = chain+res
                    tmp_list.append(encoded_residue_dict[name])
                    tmp_w_list.append(1.0)
                remapped_symmetry_residues.append(tmp_list)
                symmetry_weights.append(tmp_w_list)
                

        #set other atom bfactors to 0.0
        if other_atoms:
            other_bfactors = other_atoms.getBetas()
            other_atoms.setBetas(other_bfactors*0.0)
        #----

        #adjust input PDB name by dropping .pdb if it does exist
        name = pdb[pdb.rfind("/")+1:]
        if name[-4:] == ".pdb":
            name = name[:-4]
        #----

        with torch.no_grad():
            #run featurize to remap R_idx and add batch dimension
            feature_dict = featurize(protein_dict,
                                    cutoff_for_score=args.ligand_mpnn_cutoff_for_score, 
                                    use_atom_context=args.ligand_mpnn_use_atom_context,
                                    number_of_ligand_atoms=atom_context_num,
                                    model_type=args.model_type)
            feature_dict["batch_size"] = args.batch_size
            B, L, _, _ = feature_dict["X"].shape #batch size should be 1 for now.
            #----

            #add additional keys to the feature dictionary
            feature_dict["temperature"] = args.temperature
            feature_dict["bias"] = (-1e8*omit_AA[None,None,:]+bias_AA).repeat([1,L,1])+bias_AA_per_residue[None]-1e8*omit_AA_per_residue[None]
            if args.pair_bias_AA:
                feature_dict["pair_bias"] = make_pair_bias(feature_dict["chain_labels"][0], feature_dict["R_idx"][0], pair_bias_AA)
            feature_dict["symmetry_residues"] = remapped_symmetry_residues
            feature_dict["symmetry_weights"] = symmetry_weights
            #----

            sampling_probs_list = []
            log_probs_list = []
            decoding_order_list = []
            S_list = []
            loss_list = []
            loss_per_residue_list = []
            loss_XY_list = []
            for _ in range(args.number_of_batches):
                feature_dict["randn"] = torch.randn([feature_dict["batch_size"], feature_dict["mask"].shape[1]], device=device)
                #main step-----
                output_dict = model.sample(feature_dict)

                #compute confidence scores
                loss, loss_per_residue = get_score(output_dict["S"], output_dict["log_probs"], feature_dict["mask"]*feature_dict["chain_mask"])
                if args.model_type == "ligand_mpnn":
                    combined_mask = feature_dict["mask"]*feature_dict["mask_XY"]*feature_dict["chain_mask"]
                else:
                    combined_mask = feature_dict["mask"]*feature_dict["chain_mask"]
                loss_XY, _ = get_score(output_dict["S"], output_dict["log_probs"], combined_mask)
                #-----
                S_list.append(output_dict["S"])
                log_probs_list.append(output_dict["log_probs"])
                sampling_probs_list.append(output_dict["sampling_probs"])
                decoding_order_list.append(output_dict["decoding_order"])
                loss_list.append(loss)
                loss_per_residue_list.append(loss_per_residue)
                loss_XY_list.append(loss_XY)
            S_stack = torch.cat(S_list, 0)
            log_probs_stack = torch.cat(log_probs_list, 0)
            sampling_probs_stack = torch.cat(sampling_probs_list, 0)
            decoding_order_stack = torch.cat(decoding_order_list, 0)
            loss_stack = torch.cat(loss_list, 0)
            loss_per_residue_stack = torch.cat(loss_per_residue_list, 0)
            loss_XY_stack = torch.cat(loss_XY_list, 0)
            rec_mask = feature_dict["mask"][:1]*feature_dict["chain_mask"][:1]
            rec_stack = get_seq_rec(feature_dict["S"][:1], S_stack, rec_mask)


            #side chain packing
            #---------------
            #---------------
            if args.pack_side_chains:
                sc_feature_dict = copy.deepcopy(feature_dict)
                B = args.batch_size
                for k,v in sc_feature_dict.items():
                    if k!="S":
                        try:
                            num_dim = len(v.shape)
                            if num_dim == 2:
                                sc_feature_dict[k] = v.repeat(B,1)
                            elif num_dim == 3:
                                sc_feature_dict[k] = v.repeat(B,1,1)
                            elif num_dim == 4:
                                sc_feature_dict[k] = v.repeat(B,1,1,1)
                            elif num_dim == 5:
                                sc_feature_dict[k] = v.repeat(B,1,1,1,1)
                        except:
                            pass
                X_stack_list = []
                X_m_stack_list = []
                b_factor_stack_list = []
                for _ in range(args.number_of_packs_per_design):
                    X_list = []
                    X_m_list = []
                    b_factor_list = []
                    for c in range(args.number_of_batches):
                        sc_feature_dict["S"] = S_list[c]
                        sc_dict = pack_side_chains(sc_feature_dict, model_sc, args.sc_num_denoising_steps, args.sc_num_samples, args.repack_everything)
                        X_list.append(sc_dict["X"])
                        X_m_list.append(sc_dict["X_m"])
                        b_factor_list.append(sc_dict["b_factors"])

                    X_stack = torch.cat(X_list, 0)
                    X_m_stack = torch.cat(X_m_list, 0)
                    b_factor_stack = torch.cat(b_factor_list, 0)

                    X_stack_list.append(X_stack)
                    X_m_stack_list.append(X_m_stack)
                    b_factor_stack_list.append(b_factor_stack)

            #---------------
            #---------------
            
            #make input sequence string separated by / between different chains
            native_seq = "".join([restype_INTtoSTR[AA] for AA in feature_dict["S"][0].cpu().numpy()])
            seq_np = np.array(list(native_seq))
            seq_out_str = []
            for mask in protein_dict['mask_c']:
                seq_out_str += list(seq_np[mask.cpu().numpy()])
                seq_out_str += ['/']
            seq_out_str = "".join(seq_out_str)[:-1]
            #------

            output_fasta = base_folder + '/seqs/' + name + args.file_ending + '.fa'
            output_backbones = base_folder + '/backbones/'
            output_packed = base_folder + '/packed/'
            output_stats_path = base_folder + 'stats/' + name + ".pt"

            out_dict = {}
            out_dict["generated_sequences"] = S_stack.cpu()
            out_dict["sampling_probs"] = sampling_probs_stack.cpu()
            out_dict["log_probs"] = log_probs_stack.cpu()
            out_dict["decoding_order"] = decoding_order_stack.cpu()
            out_dict["native_sequence"] = feature_dict["S"][0].cpu()
            out_dict["mask"] = feature_dict["mask"][0].cpu()
            out_dict["chain_mask"] = feature_dict["chain_mask"][0].cpu()
            out_dict["seed"] = seed
            out_dict["temperature"] = args.temperature
            if args.save_stats:
                torch.save(out_dict, output_stats_path)



            with open(output_fasta, 'w') as f:
                f.write('>{}, T={}, seed={}, num_res={}, num_ligand_res={}, use_ligand_context={}, batch_size={}, number_of_batches={}, model_path={}\n{}\n'.format(name, args.temperature, seed, torch.sum(rec_mask).cpu().numpy(), torch.sum(combined_mask[:1]).cpu().numpy(), bool(args.ligand_mpnn_use_atom_context), args.batch_size, args.number_of_batches, checkpoint_path, seq_out_str))
                for ix in range(S_stack.shape[0]):
                    ix_suffix = ix
                    if not args.zero_indexed:
                        ix_suffix += 1
                    seq_rec_print = np.format_float_positional(rec_stack[ix].cpu().numpy(), unique=False, precision=4)
                    loss_np = np.format_float_positional(np.exp(-loss_stack[ix].cpu().numpy()), unique=False, precision=4)
                    loss_XY_np = np.format_float_positional(np.exp(-loss_XY_stack[ix].cpu().numpy()), unique=False, precision=4)
                    seq = "".join([restype_INTtoSTR[AA] for AA in S_stack[ix].cpu().numpy()])

                    #write new sequences into PDB with backbone coordinates
                    seq_prody = np.array([restype_1to3[AA] for AA in list(seq)])[None,].repeat(4,1)
                    bfactor_prody = loss_per_residue_stack[ix].cpu().numpy()[None,:].repeat(4,1)
                    backbone.setResnames(seq_prody)
                    backbone.setBetas(np.exp(-bfactor_prody)*(bfactor_prody>0.01).astype(np.float32))
                    if other_atoms:
                        writePDB(output_backbones+name+'_'+str(ix_suffix)+args.file_ending+".pdb", backbone+other_atoms)
                    else:
                        writePDB(output_backbones+name+'_'+str(ix_suffix)+args.file_ending+".pdb", backbone)
                    #-----

                    #write fasta lines
                    seq_np = np.array(list(seq))
                    seq_out_str = []
                    for mask in protein_dict['mask_c']:
                        seq_out_str += list(seq_np[mask.cpu().numpy()])
                        seq_out_str += ['/']
                    seq_out_str = "".join(seq_out_str)[:-1]
                    f.write('>{}, id={}, T={}, seed={}, overall_confidence={}, ligand_confidence={}, seq_rec={}\n{}\n'.format(name, ix_suffix,args.temperature, seed, loss_np,loss_XY_np,seq_rec_print,seq_out_str))
                    #-----

                    #write full PDB files
                    if args.pack_side_chains:
                        for c_pack in range(args.number_of_packs_per_design):
                            X_stack = X_stack_list[c_pack]
                            X_m_stack = X_m_stack_list[c_pack]
                            b_factor_stack = b_factor_stack_list[c_pack]
                            write_full_PDB(output_packed+name+args.packed_suffix+"_"+str(ix_suffix)+"_"+str(c_pack+1)+args.file_ending+".pdb", X_stack[ix].cpu().numpy(), X_m_stack[ix].cpu().numpy(), b_factor_stack[ix].cpu().numpy(), feature_dict["R_idx_original"][0].cpu().numpy(), protein_dict["chain_letters"], S_stack[ix].cpu().numpy(), other_atoms=other_atoms, icodes=icodes, force_hetatm=args.force_hetatm)
                    #-----
   
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--model_type", type=str, default="protein_mpnn", help="Choose your model: protein_mpnn, ligand_mpnn, per_residue_label_membrane_mpnn, global_label_membrane_mpnn, soluble_mpnn")
    #protein_mpnn - original ProteinMPNN trained on the whole PDB exluding non-protein atoms
    #ligand_mpnn - atomic context aware model trained with small molecules, nucleotides, metals etc on the whole PDB
    #per_residue_label_membrane_mpnn - ProteinMPNN model trained with addition label per residue specifying if that residue is buried or exposed
    #global_label_membrane_mpnn - ProteinMPNN model trained with global label per PDB id to specify if protein is transmembrane
    #soluble_mpnn - ProteinMPNN trained only on soluble PDB ids
    #pssm_mpnn - ProteinMPNN with additional PSSM like inputs
    #antibody_mpnn - ProteinMPNN trained with bias towards antibody PDBs
    argparser.add_argument("--checkpoint_protein_mpnn", type=str, default="/databases/mpnn/vanilla_model_weights/v_48_020.pt", help="Path to model weights.")
    argparser.add_argument("--checkpoint_ligand_mpnn", type=str, default="/databases/mpnn/ligand_mpnn_model_weights/s25_r010_t300_p.pt", help="Path to model weights.")
    argparser.add_argument("--checkpoint_per_residue_label_membrane_mpnn", type=str, default="/databases/mpnn/tmd_per_residue_weights/tmd_v_48_020.pt", help="Path to model weights.")
    argparser.add_argument("--checkpoint_global_label_membrane_mpnn", type=str, default="/databases/mpnn/tmd_weights/v_48_020.pt", help="Path to model weights.")
    argparser.add_argument("--checkpoint_soluble_mpnn", type=str, default="/databases/mpnn/no_transmembrane/v_48_020.pt", help="Path to model weights.")
    argparser.add_argument("--checkpoint_pssm_mpnn", type=str, default="/databases/mpnn/pssm_model_weights/v_48_020.pt", help="Path to model weights.")
    argparser.add_argument("--checkpoint_antibody_mpnn", type=str, default="/databases/mpnn/antibody_mpnn_model_weights/v_48_020_bias_005.pt", help="Path to model weights.")
    argparser.add_argument("--checkpoint_msa_mpnn", type=str, default="/projects/ml/struc2seq/msa_mpnn_models/dropout_v1/last.pt", help="Path to model weights.")
    argparser.add_argument("--enhance", type=str, default=None, help="Property to enhance, e.g. 'pLDDT'.")

    argparser.add_argument("--verbose", type=int, default=1, help="Print stuff")

    argparser.add_argument("--pdb_path", type=str, default="", help="Path to the input PDB.")
    argparser.add_argument("--pdb_path_multi", type=str, default="", help="Path to json listing PDB paths. {'/path/to/pdb': ''} - only keys will be used.")

    argparser.add_argument("--fixed_residues", type=str, default="", help="Provide fixed residues, A12 A13 A14 B2 B25")
    argparser.add_argument("--fixed_residues_multi", type=str, default="", help="Path to json mapping of fixed residues for each pdb i.e., {'/path/to/pdb': 'A12 A13 A14 B2 B25'} or {'/path/to/pdb': ['A12', 'A13', 'A14', 'B2', 'B25']}")

    argparser.add_argument("--redesigned_residues", type=str, default="", help="Provide to be redesigned residues, everything else will be fixed, A12 A13 A14 B2 B25")
    argparser.add_argument("--redesigned_residues_multi", type=str, default="", help="Path to json mapping of redesigned residues for each pdb i.e., {'/path/to/pdb': 'A12 A13 A14 B2 B25'} or {'/path/to/pdb': ['A12', 'A13', 'A14', 'B2', 'B25']}")

    argparser.add_argument("--bias_AA", type=str, default="", help="Bias generation of amino acids, e.g. 'A:-1.024,P:2.34,C:-12.34'")
    argparser.add_argument("--bias_AA_per_residue", type=str, default="", help="Path to json mapping of bias {'A12': {'G': -0.3, 'C': -2.0, 'H': 0.8}, 'A13': {'G': -1.3}}") 
    argparser.add_argument("--bias_AA_per_residue_multi", type=str, default="", help="Path to json mapping of bias {'pdb_path': {'A12': {'G': -0.3, 'C': -2.0, 'H': 0.8}, 'A13': {'G': -1.3}}}") 

    argparser.add_argument("--omit_AA", type=str, default="", help="Bias generation of amino acids, e.g. 'ACG'")
    argparser.add_argument("--omit_AA_per_residue", type=str, default="", help="Path to json mapping of bias {'A12': 'APQ', 'A13': 'QST'}") 
    argparser.add_argument("--omit_AA_per_residue_multi", type=str, default="", help="Path to json mapping of bias {'pdb_path': {'A12': 'QSPC', 'A13': 'AGE'}}") 

    argparser.add_argument("--symmetry_residues", type=str, default="", help="Add list of lists for which residues need to be symmetric, e.g. 'A12,A13,A14|C2,C3|A5,B6'")
    argparser.add_argument("--symmetry_weights", type=str, default="", help="Add weights that match symmetry_residues, e.g. '1.01,1.0,1.0|-1.0,2.0|2.0,2.3'")
    argparser.add_argument("--homo_oligomer", type=int, default=0, help="Setting this to 1 will automatically set --symmetry_residues and --symmetry_weights to do homooligomer design with equal weighting.")

    argparser.add_argument("--out_folder", type=str, help="Path to a folder to output sequences, e.g. /home/out/")
    argparser.add_argument("--file_ending", type=str, default="", help="adding_string_to_the_end")
    argparser.add_argument("--checkpoint_path_sc", type=str, default="/projects/ml/struc2seq/ligandMPNN_models/b_v1/s_300756.pt", help="Path to model weights.")
    argparser.add_argument("--packed_suffix", type=str, default="_packed", help="Suffix for packed PDB paths")
    argparser.add_argument("--force_hetatm", type=int, default=0, help="To force ligand atoms to be written as HETATM to PDB file after packing.")
    argparser.add_argument("--zero_indexed", type=str, default=0, help="1 - to start output PDB numbering with 0")
    argparser.add_argument("--seed", type=int, default=0, help="Set seed for torch, numpy, and python random.")
    argparser.add_argument("--batch_size", type=int, default=1, help="Number of sequence to generate per one pass.")
    argparser.add_argument("--number_of_batches", type=int, default=1, help="Number of times to design sequence using a chosen batch size.")
    argparser.add_argument("--temperature", type=float, default=0.1, help="Temperature to sample sequences.")
    argparser.add_argument("--save_stats", type=int, default=0, help="Save output statistics")

    argparser.add_argument("--ligand_mpnn_use_atom_context", type=int, default=1, help="1 - use atom context, 0 - do not use atom context.")
    argparser.add_argument("--ligand_mpnn_cutoff_for_score", type=float, default=8.0, help="Cutoff in angstroms between protein and context atoms to select residues for reporting score.")
    argparser.add_argument("--ligand_mpnn_use_side_chain_context", type=int, default=0, help="Flag to use side chain atoms as ligand context for the fixed residues")  


    argparser.add_argument("--pack_side_chains", type=int, default=0, help="1 - to pack side chains, 0 - do not")
    argparser.add_argument("--number_of_packs_per_design", type=int, default=1, help="Define the number of side chain packings per design")
    argparser.add_argument("--sc_num_denoising_steps", type=int, default=3, help="Number of denoising steps for side-chain packing.")
    argparser.add_argument("--sc_num_samples", type=int, default=16, help="Number of sc samples")
    argparser.add_argument("--repack_everything", type=int, default=1, help="Flag to repack everything, otherwise only newly designed residues will be repacked")

    argparser.add_argument("--chains_to_design", type=str, default=None, help="Specify which chains to redesign, all others will be kept fixed.")

    argparser.add_argument("--parse_these_chains_only", type=str, default="", help="Provide chains letters for parsing backbones, 'ABCF'")

    argparser.add_argument("--pair_bias_AA", type=str, default="", help="Add pair bias for neighboring positions, e.g. 'KK:-10.0,KE:-10.0,EK:-10.0'")

    argparser.add_argument("--transmembrane_buried", type=str, default="", help="Provide buried residues when using checkpoint_per_residue_label_membrane_mpnn model, A12 A13 A14 B2 B25")
    argparser.add_argument("--transmembrane_interface", type=str, default="", help="Provide interface residues when using checkpoint_per_residue_label_membrane_mpnn model, A12 A13 A14 B2 B25")

    argparser.add_argument("--global_transmembrane_label", type=int, default=0, help="Provide global label for global_label_membrane_mpnn model. 1 - transmembrane, 0 - soluble")

    argparser.add_argument("--pssm_input", type=str, default="", help="Path to json file with pssm log odds [20] real numbers, alphabet - ARNDCQEGHILKMFPSTWYV")

    argparser.add_argument("--msa_path", type=str, default="", help="Path to the a3m file")
    argparser.add_argument("--msa_confidence", type=float, default=1.0, help="MSA confidence from 0.0 to 1.0")
    argparser.add_argument("--num_MSA_to_use", type=int, default=0, help="Maximum number of MSAs to use when subsampling MSA if 0 then no subsampling")
    argparser.add_argument("--num_MSA_from_top", type=int, default=1000, help="Maximum number of MSAs to use when subsampling MSA if 0 then no subsampling")

    args = argparser.parse_args()
    args = parse_enhance_args(args)
    main(args)
