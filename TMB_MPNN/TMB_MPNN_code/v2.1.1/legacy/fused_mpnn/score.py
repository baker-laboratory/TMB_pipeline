import argparse
def main(args):
    import sys
    import copy
    from prody import writePDB
    import torch
    import random
    import json
    import numpy as np
    import os.path
    from data_utils import make_pair_bias, get_seq_rec, get_score, parse_PDB, featurize, write_full_PDB, parse_a3m, subsample_msa
    from model_utils import ProteinMPNN
    from sc_utils import Packer, pack_side_chains

    import time
    t0=time.time()
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
        print("Choose --model_type flag from currently available models")
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

    model.load_state_dict(checkpoint['model_state_dict'])
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


    #make amino acid bias array [21]
    bias_AA = torch.zeros([21], device=device, dtype=torch.float32)
    if args.bias_AA:
        tmp = [item.split(":") for item in args.bias_AA.split(",")]
        a1 = [b[0] for b in tmp]
        a2 = [float(b[1]) for b in tmp]
        for i, AA in enumerate(a1):
            bias_AA[restype_STRtoINT[AA]] = a2[i]
    #----

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


    if args.fixed_pos_by_pdb:
        with open(args.fixed_pos_by_pdb, 'r') as fh:
            fixed_pos_by_pdb = json.load(fh)
    else:
        fixed_residues = [item for item in args.fixed_residues.split()]
        fixed_pos_by_pdb = {
            args.pdb_path: fixed_residues
        }

    for pdb, fixed_residues in fixed_pos_by_pdb.items():
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
        #----

        #make fixed positions array; those residues will be kept fixed
        fixed_positions = torch.tensor([int(item not in fixed_residues) for item in encoded_residues], device=device)
        #----

        #specify which residues need to be designed; everything else will be fixed
        if args.redesigned_residues:
            redesigned_residues = [item for item in args.redesigned_residues.split()]
            redesigned_positions = torch.tensor([int(item not in redesigned_residues) for item in encoded_residues], device=device)
        else:
            redesigned_positions = torch.zeros_like(fixed_positions)
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
        #----


        #create mask to specify for which residues side chain context can be used/do not repack those side chains
        protein_dict["side_chain_mask"] = protein_dict["chain_mask"]
        #----
    
        if args.msa_path and args.model_type=="msa_mpnn":
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
            feature_dict["bias"] = (-1e8*omit_AA[None,None,:]+bias_AA).repeat([1,L,1])
            if args.pair_bias_AA:
                feature_dict["pair_bias"] = make_pair_bias(feature_dict["chain_labels"][0], feature_dict["R_idx"][0], pair_bias_AA)
            feature_dict["symmetry_residues"] = remapped_symmetry_residues
            feature_dict["symmetry_weights"] = symmetry_weights
            #----

            log_probs_list = []
            for _ in range(args.number_of_batches):
                feature_dict["randn"] = torch.randn([feature_dict["batch_size"], feature_dict["mask"].shape[1]], device=device)
                #main step-----
                output_dict = model.score(feature_dict)
                log_probs_list.append(output_dict["log_probs"])

            log_probs_stack = torch.cat(log_probs_list, 0)

            output_stats_path = base_folder + 'stats/' + name + ".pt"
            out_dict = {}
            out_dict["log_probs"] = log_probs_stack.cpu()
            out_dict["native_sequence"] = feature_dict["S"][0].cpu()
            out_dict["mask"] = feature_dict["mask"][0].cpu()
            out_dict["chain_mask"] = feature_dict["chain_mask"][0].cpu()
            out_dict["seed"] = seed
            if args.save_stats:
                torch.save(out_dict, output_stats_path)

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
    argparser.add_argument("--checkpoint_msa_mpnn", type=str, default="/databases/mpnn/msa_mpnn_model_weights/last.pt", help="Path to model weights.")

    argparser.add_argument("--out_folder", type=str, help="Path to a folder to output sequences, e.g. /home/out/")
    argparser.add_argument("--file_ending", type=str, default="", help="adding_string_to_the_end")
    argparser.add_argument("--checkpoint_path_sc", type=str, default="/projects/ml/struc2seq/ligandMPNN_models/b_v1/s_300756.pt", help="Path to model weights.")
    argparser.add_argument("--pdb_path", type=str, default="", help="Path to the input PDB.")
    argparser.add_argument("--fixed_pos_by_pdb", type=str, default="", help="Path to json mapping of fixed residues for each pdb i.e., {'/path/to/pdb': 'A12 A13 A14 B2 B25'}")
    argparser.add_argument("--packed_suffix", type=str, default="_packed", help="Suffix for packed PDB paths")
    argparser.add_argument("--force_hetatm", type=int, default=0, help="To force ligand atoms to be written as HETATM to PDB file after packing.")
    argparser.add_argument("--zero_indexed", type=str, default=0, help="1 - to start output PDB numbering with 0")
    argparser.add_argument("--seed", type=int, default=0, help="Set seed for torch, numpy, and python random.")
    argparser.add_argument("--batch_size", type=int, default=1, help="Number of sequence to generate per one pass.")
    argparser.add_argument("--number_of_batches", type=int, default=1, help="Number of times to design sequence using a chosen batch size.")
    argparser.add_argument("--temperature", type=float, default=0.1, help="Temperature to sample sequences.")
    argparser.add_argument("--save_stats", type=int, default=1, help="Save output statistics")


    argparser.add_argument("--ligand_mpnn_use_atom_context", type=int, default=1, help="1 - use atom context, 0 - do not use atom context.")
    argparser.add_argument("--ligand_mpnn_cutoff_for_score", type=float, default=8.0, help="Cutoff in angstroms between protein and context atoms to select residues for reporting score.")
    argparser.add_argument("--ligand_mpnn_use_side_chain_context", type=int, default=0, help="Flag to use side chain atoms as ligand context for the fixed residues")  

    argparser.add_argument("--pack_side_chains", type=int, default=0, help="1 - to pack side chains, 0 - do not")
    argparser.add_argument("--sc_num_denoising_steps", type=int, default=3, help="Number of denoising steps for side-chain packing.")
    argparser.add_argument("--sc_num_samples", type=int, default=16, help="Number of sc samples")
    argparser.add_argument("--repack_everything", type=int, default=1, help="Flag to repack everything, otherwise only newly designed residues will be repacked")

    argparser.add_argument("--chains_to_design", type=str, default=None, help="Specify which chains to redesign, all others will be kept fixed.")
    argparser.add_argument("--omit_AA", type=str, default="X", help="Omit amino acids from design, e.g. XCG")
    argparser.add_argument("--fixed_residues", type=str, default="", help="Provide fixed residues, A12 A13 A14 B2 B25")
    argparser.add_argument("--redesigned_residues", type=str, default="", help="Provide to be redesigned residues, everything else will be fixed, A12 A13 A14 B2 B25")
    argparser.add_argument("--parse_these_chains_only", type=str, default="", help="Provide chains letters for parsing backbones, 'ABCF'")
    argparser.add_argument("--bias_AA", type=str, default="", help="Bias generation of amino acids, e.g. 'A:-1.024,P:2.34,C:-12.34'")
    argparser.add_argument("--pair_bias_AA", type=str, default="", help="Add pair bias for neighboring positions, e.g. 'KK:-10.0,KE:-10.0,EK:-10.0'")
    argparser.add_argument("--symmetry_residues", type=str, default="", help="Add list of lists for which residues need to be symmetric, e.g. 'A12,A13,A14|C2,C3|A5,B6'")
    argparser.add_argument("--symmetry_weights", type=str, default="", help="Add weights that match symmetry_residues, e.g. '1.01,1.0,1.0|-1.0,2.0|2.0,2.3'")

    argparser.add_argument("--transmembrane_buried", type=str, default="", help="Provide buried residues when using checkpoint_per_residue_label_membrane_mpnn model, A12 A13 A14 B2 B25")
    argparser.add_argument("--transmembrane_interface", type=str, default="", help="Provide interface residues when using checkpoint_per_residue_label_membrane_mpnn model, A12 A13 A14 B2 B25")

    argparser.add_argument("--global_transmembrane_label", type=int, default=0, help="Provide global label for global_label_membrane_mpnn model. 1 - transmembrane, 0 - soluble")

    argparser.add_argument("--pssm_input", type=str, default="", help="Path to json file with pssm log odds [20] real numbers, alphabet - ARNDCQEGHILKMFPSTWYV")

    argparser.add_argument("--msa_path", type=str, default="", help="Path to the a3m file")
    argparser.add_argument("--msa_confidence", type=float, default=1.0, help="MSA confidence from 0.0 to 1.0")
    argparser.add_argument("--num_MSA_to_use", type=int, default=0, help="Maximum number of MSAs to use when subsampling MSA if 0 then no subsampling")
    argparser.add_argument("--num_MSA_from_top", type=int, default=1000, help="Maximum number of MSAs to use when subsampling MSA if 0 then no subsampling")

    args = argparser.parse_args()    
    main(args)   
