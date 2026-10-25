import argparse
def main(args):
    import sys
    import copy
    import torch
    import random
    import json
    import numpy as np
    import os.path
    from data_utils import parse_PDB, featurize, write_full_PDB
    from sc_utils import Packer, pack_side_chains

    import time
    t0=time.time()
    restype_STRtoINT = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20}
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
    if args.pack_side_chains:
        if not os.path.exists(base_folder + 'packed'):
            os.makedirs(base_folder + 'packed', exist_ok=True)
    if args.save_stats:
        if not os.path.exists(base_folder + 'stats'):
            os.makedirs(base_folder + 'stats', exist_ok=True)


    model_sc = Packer(node_features=128,
                    edge_features=128,
                    num_positional_embeddings=16,
                    num_chain_embeddings=16,
                    num_rbf=16,
                    hidden_dim=128,
                    num_encoder_layers=3,
                    num_decoder_layers=3,
                    atom_context_num=args.number_of_ligand_atoms,
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

    #parse PDB file
    protein_dict, _, other_atoms, icodes, _ = parse_PDB(args.pdb_path, 
                                                        device=device, 
                                                        atom_context_num=args.number_of_ligand_atoms, 
                                                        chains=args.parse_these_chains_only,
                                                        parse_all_atoms=args.ligand_mpnn_use_side_chain_context)
    
    #make chain_letter + residue_idx + insertion_code mapping to integers
    R_idx_list = list(protein_dict["R_idx"].cpu().numpy())
    chain_letters_list = list(protein_dict["chain_letters"])
    encoded_residues = []
    for i in range(len(R_idx_list)):
        tmp = str(chain_letters_list[i]) + str(R_idx_list[i]) + icodes[i]
        encoded_residues.append(tmp)
    encoded_residue_dict = dict(zip(encoded_residues, range(len(encoded_residues))))
    #----

    #make fixed positions array; those residues will be kept fixed
    fixed_residues = [item for item in args.fixed_residues.split()]
    fixed_positions = torch.tensor([int(item not in fixed_residues) for item in encoded_residues], device=device)
    #----

    #specify which residues need to be designed; everything else will be fixed
    if args.redesigned_residues:
        redesigned_residues = [item for item in args.redesigned_residues.split()]
        redesigned_positions = torch.tensor([int(item not in redesigned_residues) for item in encoded_residues], device=device)
    else:
        redesigned_positions = torch.zeros_like(fixed_positions)

    #ARNDCQEGHILKMFPSTWYV - for 20 values - log odds ratios
    protein_dict["pssm"] = torch.zeros([fixed_positions.shape[0], 20], device=device)
    if args.model_type == "pssm_mpnn":
        if args.pssm_input:
            with open(args.pssm_input, 'r') as file:
                pssm_dict = json.load(file)
                for k, v in pssm_dict.items():
                    if k in list(encoded_residue_dict):
                        idx = encoded_residue_dict[k]
                        protein_dict["pssm"][idx,:] = torch.tensor(v, device=device, dtype=torch.float32)

    #specify which chains need to be redesigned
    if type(args.chains_to_design) == str:
        chains_to_design_list = args.chains_to_design.split(",")
    else:
        chains_to_design_list = protein_dict["chain_letters"]
    chain_mask = torch.tensor(np.array([item in chains_to_design_list for item in protein_dict["chain_letters"]],dtype=np.int32), device=device)
    #----

    #create chain_mask to notify which residues are fixed (0) and which need to be designed (1)
    protein_dict["chain_mask"] = chain_mask*fixed_positions*(1-redesigned_positions)
    #---

    #create mask to specify for which residues side chain context can be used/do not repack those side chains
    protein_dict["side_chain_mask"] = protein_dict["chain_mask"]

    #set other atom bfactors to 0.0
    if other_atoms:
        other_bfactors = other_atoms.getBetas()
        other_atoms.setBetas(other_bfactors*0.0)
    #----

    #adjust input PDB name by dropping .pdb if it does exist
    name = args.pdb_path[args.pdb_path.rfind("/")+1:]
    if name[-4:] == ".pdb":
        name = name[:-4]
    #----

    with torch.no_grad():
        #run featurize to remap R_idx and add batch dimension
        feature_dict = featurize(protein_dict,
                                 cutoff_for_score=8.0, 
                                 use_atom_context=args.ligand_mpnn_use_atom_context,
                                 number_of_ligand_atoms=args.number_of_ligand_atoms,
                                 model_type=args.model_type)
        feature_dict["batch_size"] = args.batch_size
        B, L, _, _ = feature_dict["X"].shape #batch size should be 1 for now.

        S_list = []
        S_true = feature_dict["S"].long().repeat(args.batch_size,1)
        for _ in range(args.number_of_batches):
            S_list.append(S_true)
        S_stack = torch.cat(S_list, 0)
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
            X_list = []
            X_m_list = []
            b_factor_list = []
            mean_list = []
            concentration_list = []
            mix_logits_list = []
            log_prob_list = []
            sample_list = []
            true_torsion_sin_cos_list = []
            for c in range(args.number_of_batches):
                sc_feature_dict["S"] = S_list[c]
                sc_dict = pack_side_chains(sc_feature_dict, model_sc, args.sc_num_denoising_steps, args.sc_num_samples, args.repack_everything)
                X_list.append(sc_dict["X"])
                X_m_list.append(sc_dict["X_m"])
                b_factor_list.append(sc_dict["b_factors"])
                mean_list.append(sc_dict["mean"])
                concentration_list.append(sc_dict["concentration"])
                mix_logits_list.append(sc_dict["mix_logits"])
                log_prob_list.append(sc_dict["log_prob"])
                sample_list.append(sc_dict["sample"])
                true_torsion_sin_cos_list.append(sc_dict["true_torsion_sin_cos"])

            X_stack = torch.cat(X_list, 0)
            X_m_stack = torch.cat(X_m_list, 0)
            b_factor_stack = torch.cat(b_factor_list, 0)
            S_stack = torch.cat(S_list, 0)
            mean_stack = torch.cat(mean_list, 0)
            concentration_stack= torch.cat(concentration_list, 0)
            mix_logits_stack = torch.cat(mix_logits_list, 0)
            log_prob_stack = torch.cat(log_prob_list, 0)
            sample_stack = torch.cat(sample_list, 0)
            true_torsion_sin_cos_stack = torch.cat(true_torsion_sin_cos_list, 0)

        output_packed = base_folder + '/packed/'
        output_stats_path = base_folder + 'stats/' + name + ".pt"

        if args.save_stats:
            out_dict = {}
            out_dict["X"] = X_stack.cpu()
            out_dict["X_m"] = X_m_stack.cpu()
            out_dict["b_factor"] =b_factor_stack.cpu()
            out_dict["S"] = S_stack.cpu()
            out_dict["mean"] = mean_stack.cpu()
            out_dict["concentration"] = concentration_stack.cpu()
            out_dict["mix_logits"] = mix_logits_stack.cpu()
            out_dict["log_prob"] = log_prob_stack.cpu()
            out_dict["sample"] = sample_stack.cpu()
            out_dict["true_torsion_sin_cos"] = true_torsion_sin_cos_stack.cpu()


            torch.save(out_dict, output_stats_path)


        for ix in range(S_stack.shape[0]):
            if args.pack_side_chains:
                write_full_PDB(output_packed+name+"_"+"packed"+"_"+str(ix+1)+".pdb"+ args.file_ending, X_stack[ix].cpu().numpy(), X_m_stack[ix].cpu().numpy(), b_factor_stack[ix].cpu().numpy(), feature_dict["R_idx_original"][0].cpu().numpy(), protein_dict["chain_letters"], S_stack[ix].cpu().numpy(), other_atoms=other_atoms, icodes=icodes)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--model_type", type=str, default="protein_mpnn", help="Choose your model: protein_mpnn, ligand_mpnn, per_residue_label_membrane_mpnn, global_label_membrane_mpnn, soluble_mpnn")
    argparser.add_argument("--out_folder", type=str, help="Path to a folder to output sequences, e.g. /home/out/")
    argparser.add_argument("--file_ending", type=str, default="", help="adding_string_to_the_end")
    argparser.add_argument("--checkpoint_path_sc", type=str, default="/projects/ml/struc2seq/ligandMPNN_models/b_v1/s_300756.pt", help="Path to model weights.")
    argparser.add_argument("--pdb_path", type=str, default="", help="Path to the input PDB.")
    argparser.add_argument("--seed", type=int, default=0, help="Set seed for torch, numpy, and python random.")
    argparser.add_argument("--batch_size", type=int, default=1, help="Number of sequence to generate per one pass.")
    argparser.add_argument("--number_of_batches", type=int, default=1, help="Number of times to design sequence using a chosen batch size.")
    argparser.add_argument("--save_stats", type=int, default=0, help="Save output statistics")

    argparser.add_argument("--number_of_ligand_atoms", type=int, default=16, help="Depends on the model weights, 16 or 25")
    argparser.add_argument("--ligand_mpnn_use_atom_context", type=int, default=1, help="1 - use atom context, 0 - do not use atom context.")
    argparser.add_argument("--ligand_mpnn_use_side_chain_context", type=int, default=0, help="Flag to use side chain atoms as ligand context for the fixed residues")  

    argparser.add_argument("--pack_side_chains", type=int, default=0, help="1 - to pack side chains, 0 - do not")
    argparser.add_argument("--sc_num_denoising_steps", type=int, default=3, help="Number of denoising steps for side-chain packing.")
    argparser.add_argument("--sc_num_samples", type=int, default=16, help="Number of sc samples")
    argparser.add_argument("--repack_everything", type=int, default=1, help="Flag to repack everything, otherwise only newly designed residues will be repacked")

    argparser.add_argument("--chains_to_design", type=str, default=None, help="Specify which chains to redesign, all others will be kept fixed.")
    argparser.add_argument("--fixed_residues", type=str, default="", help="Provide fixed residues, A12 A13 A14 B2 B25")
    argparser.add_argument("--redesigned_residues", type=str, default="", help="Provide to be redesigned residues, everything else will be fixed, A12 A13 A14 B2 B25")
    argparser.add_argument("--parse_these_chains_only", type=str, default="", help="Provide chains letters for parsing backbones, 'ABCF'")

    args = argparser.parse_args()    
    main(args)   
