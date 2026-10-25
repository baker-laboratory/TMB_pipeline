import argparse
import copy
import json
import logging
import os.path
import random
import sys
import warnings
from typing import Optional

import numpy as np
import torch
from typing import Counter
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.registry import MODELS
from mmengine.runner import load_checkpoint, load_state_dict
from mmengine.structures import InstanceData

from mpnn.datasets.fused_data_utils import (
    get_score,
    get_seq_rec,
    make_pair_bias,
    parse_a3m,
    subsample_msa,
    save_output_structure,
)
from mpnn.datasets.residue_mapping import (
    ALPHABET,
    restype_1to3,
    restype_INTtoSTR,
    restype_STRtoINT,
)
from sc_utils import Packer, pack_side_chains


def build_model(cfg: Optional[dict] = None, checkpoint=None, enhance=None, device="cpu"):
    """
    TODO: should we build model from checkpoint? For some configs, like noise, we may don't want to load from checkpoint.
    if checkpoint is not None:
        assert enhance is None, "Please provide either a checkpoint or an enhance key, not both."
        logging.info(f"Loading model from checkpoint: {checkpoint}")
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        cfg_str = state_dict["meta"]["cfg"]
        cfg = Config.fromstring(cfg_str, file_format=".py")
        model = MODELS.build(cfg.model)
        load_state_dict(model, state_dict["state_dict"])
    """
    model = MODELS.build(cfg.model)

    if enhance:
        cfg.load_from = f"/net/scratch/faxue/share/mpnn_enhancer/{enhance}.pth"
        if not os.path.exists(cfg.load_from):
            raise ValueError(
                f"Enhance key {enhance} is not correct, please contact Fanglei for proper value."
            )
    if checkpoint:
        if cfg.load_from is not None:
            warnings.warn(f"default checkpoint is overwritten by args.checkpoint ({checkpoint}).")
        cfg.load_from = args.checkpoint
    else:
        assert cfg.load_from is not None, "Please provide a checkpoint path."

    load_checkpoint(model, cfg.load_from, map_location="cpu", strict=True)
    model.to(device)
    model.eval()
    return model


def main(args):

    # fix seeds
    if args.seed:
        seed = args.seed
    else:
        seed = int(np.random.randint(0, high=99999, size=1, dtype=int)[0])

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # make folders for outputs
    folder_for_outputs = args.out_folder
    base_folder = folder_for_outputs
    if base_folder[-1] != "/":
        base_folder = base_folder + "/"
    if not os.path.exists(base_folder):
        os.makedirs(base_folder, exist_ok=True)
    if not os.path.exists(base_folder + "seqs"):
        os.makedirs(base_folder + "seqs", exist_ok=True)
    # if not os.path.exists(base_folder + "backbones"):
    #     os.makedirs(base_folder + "backbones", exist_ok=True)
    if args.pack_side_chains:
        if not os.path.exists(base_folder + "packed"):
            os.makedirs(base_folder + "packed", exist_ok=True)
    if args.save_stats:
        if not os.path.exists(base_folder + "stats"):
            os.makedirs(base_folder + "stats", exist_ok=True)

    cfg_file = os.path.join(os.path.dirname(__file__), args.model_type)
    cfg = Config.fromfile(cfg_file)
    cfg.merge_from_dict(dict(init_cfg=None))  # use load_from

    model = build_model(cfg, args.checkpoint, args.enhance, device)

    test_loader = Compose(cfg.test_pipeline)

    # load side chain packing model if needed
    if args.pack_side_chains:
        model_sc = Packer(
            node_features=128,
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
            num_mix=3,
        )

        checkpoint_sc = torch.load(args.checkpoint_path_sc, map_location=device, weights_only=True)
        model_sc.load_state_dict(checkpoint_sc["model_state_dict"])
        model_sc.to(device)
        model_sc.eval()
    if args.pdb_path_multi:
        with open(args.pdb_path_multi, "r") as fh:
            pdb_paths = list(json.load(fh))
    else:
        pdb_paths = [args.pdb_path]

    if args.fixed_residues_multi:
        with open(args.fixed_residues_multi, "r") as fh:
            fixed_residues_multi = json.load(fh)
            for k, v in fixed_residues_multi.items():
                if isinstance(v, str):
                    fixed_residues_multi[k] = v.split()
    else:
        fixed_residues = [item for item in args.fixed_residues.split()]
        fixed_residues_multi = {}
        for pdb in pdb_paths:
            fixed_residues_multi[pdb] = fixed_residues

    if args.tmbb_surface_condition_multi:
        with open(args.tmbb_surface_condition_multi, "r") as fh:
            tmbb_surface_condition_multi = json.load(fh)
        for k, v in tmbb_surface_condition_multi.items():
            assert isinstance(v, list), f"tmbb_surface_condition_multi values should be a list, but got {type(v)}"
        for pdb in pdb_paths:
            assert pdb in tmbb_surface_condition_multi, f"pdb {pdb} not found in tmbb_surface_condition_multi"
    elif args.tmbb_surface_condition:
        tmbb_surface_condition_single = args.tmbb_surface_condition.replace(',', ' ')
        tmbb_surface_condition_single = [float(item) for item in tmbb_surface_condition_single.split()]
        for i in tmbb_surface_condition_single:
            assert i.is_integer, f"tmbb_surface_condition values should be integers, but got {i}"
            i = int(i)
            assert i in [0, 1, 2, 3], f"tmbb_surface_condition values should be 0, 1, 2 or 3, but got {i}"
        tmbb_surface_condition_multi = {pdb: tmbb_surface_condition_single for pdb in pdb_paths}
    else:
        tmbb_surface_condition_multi = None

    if args.redesigned_residues_multi:
        with open(args.redesigned_residues_multi, "r") as fh:
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
    
    infer_fixed_residues_from_cif = False
    if not any(list(fixed_residues_multi.values())+list(redesigned_residues_multi.values())):
        print(
            "No residues to be fixed or redesigned were specified. If the input CIF contains the " \
            "`is_motif_atom_with_fixed_seq` annotation, this will be used to infer appropriate values."
        )
        infer_fixed_residues_from_cif = True

    # make amino acid bias array [21]
    bias_AA = torch.zeros([21], device=device, dtype=torch.float32)
    if args.bias_AA:
        tmp = [item.split(":") for item in args.bias_AA.split(",")]
        a1 = [b[0] for b in tmp]
        a2 = [float(b[1]) for b in tmp]
        for i, AA in enumerate(a1):
            bias_AA[restype_STRtoINT[AA]] = a2[i]

    if args.bias_AA_per_residue_multi:
        with open(args.bias_AA_per_residue_multi, "r") as fh:
            bias_AA_per_residue_multi = json.load(fh)  # {"pdb_path" : {"A12": {"G": 1.1}}}
    else:
        if args.bias_AA_per_residue:
            with open(args.bias_AA_per_residue, "r") as fh:
                bias_AA_per_residue = json.load(fh)  # {"A12": {"G": 1.1}}
            bias_AA_per_residue_multi = {}
            for pdb in pdb_paths:
                bias_AA_per_residue_multi[pdb] = bias_AA_per_residue

    if args.omit_AA_per_residue_multi:
        with open(args.omit_AA_per_residue_multi, "r") as fh:
            omit_AA_per_residue_multi = json.load(fh)  # {"pdb_path" : {"A12": "PQR", "A13": "QS"}}
    else:
        if args.omit_AA_per_residue:
            with open(args.omit_AA_per_residue, "r") as fh:
                omit_AA_per_residue = json.load(fh)  # {"A12": "PG"}
            omit_AA_per_residue_multi = {}
            for pdb in pdb_paths:
                omit_AA_per_residue_multi[pdb] = omit_AA_per_residue
                
                
    # specify which residues are linked from an input json file
    if args.symmetry_residues_multi:
        with open(args.symmetry_residues_multi, "r") as fh:
            symmetry_residues_multi = json.load(fh)
            for k, v in symmetry_residues_multi.items():
                if isinstance(v, str):
                    symmetry_residues_multi[k] = [x.split(",") for x in v.split("|")]
                    
    # specify linking weights from an input json file
    if args.symmetry_weights_multi:
        with open(args.symmetry_weights_multi, "r") as fh:
            symmetry_weights_multi = json.load(fh)
            for k, v in symmetry_weights_multi.items():
                if isinstance(v, str):
                    symmetry_weights_multi[k] = [[float(item) for item in x.split(",")] for x in v.split("|")]


    # make amino acid pair bias array [21, 21]
    pair_bias_AA = torch.zeros([21, 21], dtype=torch.float32, device=device)
    if args.pair_bias_AA:
        tmp = [item.split(":") for item in args.pair_bias_AA.split(",")]
        a1 = [b[0][0] for b in tmp]
        a2 = [b[0][1] for b in tmp]
        a3 = [float(b[1]) for b in tmp]
        for i, AA in enumerate(a1):
            pair_bias_AA[restype_STRtoINT[AA], restype_STRtoINT[a2[i]]] = a3[i]

    # make array to indicate which amino acids need to be omitted [21]
    omit_AA_list = args.omit_AA
    omit_AA = torch.tensor(
        np.array([AA in omit_AA_list for AA in ALPHABET]).astype(np.float32), device=device
    )

    for pdb in pdb_paths:
        #adjust input PDB name by dropping .pdb if it does exist
        name = pdb[pdb.rfind("/")+1:]
        if name[-4:] == ".pdb":
            name = name[:-4]
        output_fasta = os.path.join(base_folder, 'seqs', name + '.fa' + args.file_ending)
        if args.verbose:
            print("Designing this PDB:", pdb)

        name = pdb.split("/")[-1].split(".")[0]

        output_fasta = base_folder + "/seqs/" + name + ".fa" + args.file_ending
        # output_backbones = base_folder + "/backbones/"
        output_packed = base_folder + "/packed/"
        output_stats_path = base_folder + "stats/" + name + ".pt"

        if os.path.exists(output_fasta) and not args.overwrite:
            print(f"Output fasta file {output_fasta} already exists, skipping")
            continue

        fixed_residues = fixed_residues_multi[pdb]
        redesigned_residues = redesigned_residues_multi[pdb]

        protein_dict = test_loader(dict(cif_path=pdb, assembly_id=None))

        if infer_fixed_residues_from_cif:
            atom_array = protein_dict["atom_array"]
            if hasattr(atom_array, "is_motif_atom_with_fixed_seq"):
                is_motif_atom_with_fixed_seq = atom_array.is_motif_atom_with_fixed_seq
                if not np.issubdtype(is_motif_atom_with_fixed_seq.dtype, np.bool_):
                    assert np.all((is_motif_atom_with_fixed_seq == "True") | (is_motif_atom_with_fixed_seq == "False"))
                    is_motif_atom_with_fixed_seq = (is_motif_atom_with_fixed_seq == "True")
                fixed_sequence_atoms = atom_array[is_motif_atom_with_fixed_seq]
                fixed_residues = list(np.unique(np.char.add(fixed_sequence_atoms.chain_id, fixed_sequence_atoms.res_id.astype(str))))
                print(f"Residues to keep fixed were inferred from {pdb}")

        icodes = protein_dict["ins_codes"]
        # make chain_letter + residue_idx + insertion_code mapping to integers
        R_idx_list = protein_dict["R_idx"]
        chain_letters_list = list(protein_dict["chain_letters"])
        encoded_residues = []
        for i in range(len(R_idx_list)):
            tmp = str(chain_letters_list[i]) + str(R_idx_list[i]) + icodes[i]
            encoded_residues.append(tmp)
        encoded_residue_dict = dict(zip(encoded_residues, range(len(encoded_residues))))
        encoded_residue_dict_rev = dict(zip(list(range(len(encoded_residues))), encoded_residues))

        bias_AA_per_residue = torch.zeros(
            [len(encoded_residues), 21], device=device, dtype=torch.float32
        )
        if args.bias_AA_per_residue_multi or args.bias_AA_per_residue:
            bias_dict = bias_AA_per_residue_multi[pdb]
            bias_hit_count = 0
            for residue_name, v1 in bias_dict.items():
                if residue_name in encoded_residues:
                    i1 = encoded_residue_dict[residue_name]
                    for amino_acid, v2 in v1.items():
                        if amino_acid in ALPHABET:
                            j1 = restype_STRtoINT[amino_acid]
                            bias_AA_per_residue[i1, j1] = v2
                            bias_hit_count += 1
            assert (
                bias_hit_count > 0
            ), f"--bias_AA_per_residue_multi or --bias_AA_per_residue does not match any residues in the PDB file"
        # ----

        omit_AA_per_residue = torch.zeros(
            [len(encoded_residues), 21], device=device, dtype=torch.float32
        )
        if args.omit_AA_per_residue_multi or args.omit_AA_per_residue:
            omit_dict = omit_AA_per_residue_multi[pdb]
            omit_hit_count = 0
            for residue_name, v1 in omit_dict.items():
                if residue_name in encoded_residues:
                    i1 = encoded_residue_dict[residue_name]
                    for amino_acid in v1:
                        if amino_acid in ALPHABET:
                            j1 = restype_STRtoINT[amino_acid]
                            omit_AA_per_residue[i1, j1] = 1.0
                            omit_hit_count += 1
            assert (
                omit_hit_count > 0
            ), f"--omit_AA_per_residue_multi or --omit_AA_per_residue does not match any residues in the PDB file"
        # ----

        fixed_positions = torch.tensor(
            [int(item not in fixed_residues) for item in encoded_residues], device=device
        )
        redesigned_positions = torch.tensor(
            [int(item not in redesigned_residues) for item in encoded_residues], device=device
        )

        # specify which residues are buried for checkpoint_per_residue_label_membrane_mpnn model
        if args.transmembrane_buried:
            buried_residues = [item for item in args.transmembrane_buried.split()]
            buried_positions = torch.tensor(
                [int(item in buried_residues) for item in encoded_residues], device=device
            )
        else:
            buried_positions = torch.zeros_like(fixed_positions)

        if args.transmembrane_interface:
            interface_residues = [item for item in args.transmembrane_interface.split()]
            interface_positions = torch.tensor(
                [int(item in interface_residues) for item in encoded_residues], device=device
            )
        else:
            interface_positions = torch.zeros_like(fixed_positions)
        protein_dict["membrane_per_residue_labels"] = 2 * buried_positions * (
            1 - interface_positions
        ) + 1 * interface_positions * (1 - buried_positions)

        if args.model_type == "global_label_membrane_mpnn":
            protein_dict["membrane_per_residue_labels"] = (
                args.global_transmembrane_label + 0 * fixed_positions
            )

        # ARNDCQEGHILKMFPSTWYV - for 20 values - log odds ratios
        protein_dict["pssm"] = torch.zeros([fixed_positions.shape[0], 20], device=device)
        if args.model_type == "pssm_mpnn":
            if args.pssm_input:
                with open(args.pssm_input, "r") as file:
                    pssm_dict = json.load(file)
                pssm_hit_count = 0
                for k, v in pssm_dict.items():
                    if k in encoded_residue_dict:
                        idx = encoded_residue_dict[k]
                        protein_dict["pssm"][idx, :] = torch.tensor(
                            v, device=device, dtype=torch.float32
                        )
                        pssm_hit_count += 1
                assert (
                    pssm_hit_count > 0
                ), f"--pssm_input does not match any residues in the PDB file"

        # specify which chains need to be redesigned
        if type(args.chains_to_design) == str:
            chains_to_design_list = args.chains_to_design.split(",")
        else:
            chains_to_design_list = protein_dict["chain_letters"]
        chain_mask = torch.tensor(
            np.array(
                [item in chains_to_design_list for item in protein_dict["chain_letters"]],
                dtype=np.int32,
            ),
            device=device,
        )

        # create chain_mask to notify which residues are fixed (0) and which need to be designed (1)
        if redesigned_residues:
            protein_dict["chain_mask"] = chain_mask * (1 - redesigned_positions)
        elif fixed_residues:
            protein_dict["chain_mask"] = chain_mask * fixed_positions
        else:
            protein_dict["chain_mask"] = chain_mask

        if args.verbose:
            PDB_residues_to_be_redesigned = [
                encoded_residue_dict_rev[item]
                for item in range(protein_dict["chain_mask"].shape[0])
                if protein_dict["chain_mask"][item] == 1
            ]
            PDB_residues_to_be_fixed = [
                encoded_residue_dict_rev[item]
                for item in range(protein_dict["chain_mask"].shape[0])
                if protein_dict["chain_mask"][item] == 0
            ]
            print("These residues will be redesigned: ", PDB_residues_to_be_redesigned)
            print("These residues will be fixed: ", PDB_residues_to_be_fixed)

        # create mask to specify for which residues side chain context can be used/do not repack those side chains
        protein_dict["side_chain_mask"] = protein_dict["chain_mask"]

        if args.msa_path and (args.model_type == "msa_mpnn"):
            msa, ins = parse_a3m(args.msa_path, maxseq=args.num_MSA_from_top)
            if args.num_MSA_to_use:
                msa, ins = subsample_msa(msa, ins, maxseq=args.num_MSA_to_use, sub_type="UNI")
            msa[0, :] = 0 * msa[0, :] + 20  # remove original sequence info
            msa = np.transpose(msa)  # [L,N]

            protein_dict["MSA"] = torch.tensor(msa, device=device, dtype=torch.int64)
            protein_dict["MSA_confidence"] = torch.tensor(
                args.msa_confidence, device=device, dtype=torch.float32
            )

        # specify which residues are linked
        if args.symmetry_residues:
            symmetry_residues_list_of_lists = [
                x.split(",") for x in args.symmetry_residues.split("|")
            ]
            remapped_symmetry_residues = []
            for t_list in symmetry_residues_list_of_lists:
                tmp_list = []
                for t in t_list:
                    tmp_list.append(encoded_residue_dict[t])
                remapped_symmetry_residues.append(tmp_list)
        elif args.symmetry_residues_multi:
            symmetry_residues_list_of_lists = symmetry_residues_multi[pdb]
            remapped_symmetry_residues = []
            for t_list in symmetry_residues_list_of_lists:
                tmp_list = []
                for t in t_list:
                    tmp_list.append(encoded_residue_dict[t])
                remapped_symmetry_residues.append(tmp_list)
        else:
            remapped_symmetry_residues = [[]]
                

        # specify linking weights
        if args.symmetry_weights:
            symmetry_weights = [
                [float(item) for item in x.split(",")] for x in args.symmetry_weights.split("|")
            ]
        elif args.symmetry_weights_multi:
            symmetry_weights = symmetry_weights_multi[pdb]
        else:
            symmetry_weights = [[]]
                
        

        if args.homo_oligomer:
            if args.verbose:
                print("Designing HOMO-OLIGOMER")
            chain_letters_set = list(set(chain_letters_list))
            reference_chain = chain_letters_set[0]
            lc = len(reference_chain)
            per_chain_residue_indices_dict = {}
            for c in chain_letters_set:
                residue_indices = [item[lc:] for item in encoded_residues if item[:lc] == c]
                per_chain_residue_indices_dict[c] = residue_indices
            remapped_symmetry_residues = []
            symmetry_weights = []
            for i, _ in enumerate(per_chain_residue_indices_dict[reference_chain]):
                tmp_list = []
                tmp_w_list = []
                for chain in chain_letters_set:
                    res = per_chain_residue_indices_dict[chain][i]
                    name = chain + res
                    tmp_list.append(encoded_residue_dict[name])
                    tmp_w_list.append(1.0)
                remapped_symmetry_residues.append(tmp_list)
                symmetry_weights.append(tmp_w_list)

        # set other atom bfactors to 0.0
        # if other_atoms:
        #     other_bfactors = other_atoms.getBetas()
        #     other_atoms.setBetas(other_bfactors*0.0)

        # adjust input PDB name by dropping .pdb if it does exist
        name = pdb[pdb.rfind("/") + 1 :]
        if name[-4:] == ".pdb":
            name = name[:-4]

        with torch.no_grad():
            feature_dict = dict()
            for k, v in protein_dict.items():
                if isinstance(v, np.ndarray):
                    if not np.issubdtype(v.dtype, np.str_):
                        v = torch.from_numpy(v)[None,].to(device)
                elif isinstance(v, torch.Tensor):
                    v = v[None,].to(device)
                feature_dict[k] = v
            
            # move to device
            for k, v in feature_dict.items():
                if isinstance(v, torch.Tensor):
                    feature_dict[k] = v.to(device)

            feature_dict["xyz_37"] = feature_dict["X"]
            feature_dict["xyz_37_m"] = feature_dict["X_m"]

            # Looks like the mask_c is only for save fasta file
            sorted_chain_letters = sorted(set(protein_dict["chain_letters"]))
            mask_c = np.array(
                [np.array(protein_dict["chain_letters"]) == c for c in sorted_chain_letters]
            )
            mask_c = torch.tensor(mask_c, dtype=bool)[:,]

            feature_dict["batch_size"] = args.batch_size
            B, L, _, _ = feature_dict["X"].shape  # batch size should be 1 for now.

            # add additional keys to the feature dictionary
            feature_dict["temperature"] = args.temperature
            feature_dict["bias"] = (
                (-1e8 * omit_AA[None, None, :] + bias_AA).repeat([1, L, 1])
                + bias_AA_per_residue[None]
                - 1e8 * omit_AA_per_residue[None]
            )
            if args.pair_bias_AA:
                feature_dict["pair_bias"] = make_pair_bias(
                    feature_dict["chain_labels"][0], feature_dict["R_idx"][0], pair_bias_AA
                )
            feature_dict["symmetry_residues"] = remapped_symmetry_residues
            feature_dict["symmetry_weights"] = symmetry_weights

            # tmbb surface condition
            if tmbb_surface_condition_multi:
                data_sample = InstanceData(metainfo=dict(
                    intf_buried_labels=np.array(tmbb_surface_condition_multi[pdb], dtype=np.int64)))
                feature_dict['data_sample'] = [data_sample]
            else:
                feature_dict['data_sample'] = None

            sampling_probs_list = []
            log_probs_list = []
            decoding_order_list = []
            S_list = []
            loss_list = []
            loss_per_residue_list = []
            loss_XY_list = []
            for _ in range(args.number_of_batches):
                had2discard = False
                feature_dict["randn"] = torch.randn(
                    [feature_dict["batch_size"], feature_dict["mask"].shape[1]], device=device
                )
                # main step-----
                output_dict = model.sample(**feature_dict)
                # See if the percentage of threonines and serines in the final sequence is higher than a given threshold
                slistall = output_dict["S"].cpu().numpy().tolist()
                serthre_indices = []
                for pdbi, slist in enumerate(slistall):
                    seqtmp = "".join([restype_INTtoSTR[AA] for AA in slist])
                    if (seqtmp.count("S") + seqtmp.count("T")) / len(seqtmp) > args.threonine_serine_threshold:
                        # If the condition is met, we need to discard this model output and generate a new one by biasing against serines and threonines.
                        # find residue indices corresponding to all serines and threonines
                        serthre_indices += [i for i, AA in enumerate(seqtmp) if AA in ["S", "T"]]
                
                # Find the frequency of occurence of each value in serthre_indices
                if serthre_indices:
                    serthre_freq = Counter(serthre_indices)
                    serthre_indices = list(set(serthre_indices))
                    # Only keep residues with frequency greater than 0.5*batch_size
                    serthre_indices = [i for i in serthre_indices if serthre_freq[i] > 0.5 * args.batch_size]
                    if serthre_indices:
                        if len(serthre_indices)/len(seqtmp) > args.threonine_serine_threshold:
                            feature_dict["bias"][:, serthre_indices, restype_STRtoINT["S"]] = args.serthre_bias
                            feature_dict["bias"][:, serthre_indices, restype_STRtoINT["T"]] = args.serthre_bias
                            # Add bias for tyrosine as well, since we would like to see any tyrosine motifs
                            feature_dict["bias"][:, serthre_indices, restype_STRtoINT["Y"]] = 1.0
                            print(f"\n\n\n Discarding this batch due to plenty of serines and threonines. Percentage of S and T in more than half the sequences is {len(serthre_indices)/len(seqtmp)}\n\n")
                            had2discard = True
                if had2discard:
                    output_dict = model.sample(**feature_dict)
                # compute confidence scores
                loss, loss_per_residue = get_score(
                    output_dict["S"],
                    output_dict["log_probs"],
                    feature_dict["mask"] * feature_dict["chain_mask"],
                )
                if args.model_type == "ligand_mpnn":
                    combined_mask = (
                        feature_dict["mask"] * feature_dict["mask_XY"] * feature_dict["chain_mask"]
                    )
                else:
                    combined_mask = feature_dict["mask"] * feature_dict["chain_mask"]
                loss_XY, _ = get_score(output_dict["S"], output_dict["log_probs"], combined_mask)

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
            rec_mask = feature_dict["mask"][:1] * feature_dict["chain_mask"][:1]
            rec_stack = get_seq_rec(feature_dict["S"][:1], S_stack, rec_mask)

            # side chain packing
            if args.pack_side_chains:
                sc_feature_dict = copy.deepcopy(feature_dict)
                B = args.batch_size
                for k, v in sc_feature_dict.items():
                    if k != "S":
                        try:
                            num_dim = len(v.shape)
                            if num_dim == 2:
                                sc_feature_dict[k] = v.repeat(B, 1)
                            elif num_dim == 3:
                                sc_feature_dict[k] = v.repeat(B, 1, 1)
                            elif num_dim == 4:
                                sc_feature_dict[k] = v.repeat(B, 1, 1, 1)
                            elif num_dim == 5:
                                sc_feature_dict[k] = v.repeat(B, 1, 1, 1, 1)
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
                        sc_dict = pack_side_chains(
                            sc_feature_dict,
                            model_sc,
                            args.sc_num_denoising_steps,
                            args.sc_num_samples,
                            args.repack_everything,
                        )
                        X_list.append(sc_dict["X"])
                        X_m_list.append(sc_dict["X_m"])
                        b_factor_list.append(sc_dict["b_factors"])

                    X_stack = torch.cat(X_list, 0)
                    X_m_stack = torch.cat(X_m_list, 0)
                    b_factor_stack = torch.cat(b_factor_list, 0)

                    X_stack_list.append(X_stack)
                    X_m_stack_list.append(X_m_stack)
                    b_factor_stack_list.append(b_factor_stack)

            # make input sequence string separated by / between different chains
            native_seq = "".join(
                [restype_INTtoSTR[AA] for AA in feature_dict["S"][0].cpu().numpy()]
            )
            seq_np = np.array(list(native_seq))
            seq_out_str = []
            for mask in mask_c:
                seq_out_str += list(seq_np[mask.cpu().numpy()])
                seq_out_str += ["/"]
            seq_out_str = "".join(seq_out_str)[:-1]

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

            with open(output_fasta, "w") as f:
                f.write(
                    f">{name}, T={args.temperature}, seed={seed}, num_res={torch.sum(rec_mask).cpu().numpy()}, "
                    f"num_ligand_res={torch.sum(combined_mask[:1]).cpu().numpy()}, use_ligand_context={bool(args.ligand_mpnn_use_atom_context)}, "
                    f"batch_size={args.batch_size}, number_of_batches={args.number_of_batches}, model_path={cfg.load_from}\n"
                    f"{seq_out_str}\n"
                )

                for ix in range(S_stack.shape[0]):
                    ix_suffix = ix
                    if not args.zero_indexed:
                        ix_suffix += 1
                    seq_rec_print = np.format_float_positional(
                        rec_stack[ix].cpu().numpy(), unique=False, precision=4
                    )
                    loss_np = np.format_float_positional(
                        np.exp(-loss_stack[ix].cpu().numpy()), unique=False, precision=4
                    )
                    loss_XY_np = np.format_float_positional(
                        np.exp(-loss_XY_stack[ix].cpu().numpy()), unique=False, precision=4
                    )
                    seq = "".join([restype_INTtoSTR[AA] for AA in S_stack[ix].cpu().numpy()])

                    #write fasta lines
                    seq_np = np.array(list(seq))
                    seq_out_str = []
                    for mask in mask_c:
                        seq_out_str += list(seq_np[mask.cpu().numpy()])
                        seq_out_str += ["/"]
                    seq_out_str = "".join(seq_out_str)[:-1]
                    f.write(
                        ">{}, id={}, T={}, seed={}, overall_confidence={}, ligand_confidence={}, seq_rec={}\n{}\n".format(
                            name,
                            ix_suffix,
                            args.temperature,
                            seed,
                            loss_np,
                            loss_XY_np,
                            seq_rec_print,
                            seq_out_str,
                        )
                    )

                    # write full PDB files
                    if args.pack_side_chains:
                        for c_pack in range(args.number_of_packs_per_design):
                            X_stack = X_stack_list[c_pack]
                            X_m_stack = X_m_stack_list[c_pack]
                            b_factor_stack = b_factor_stack_list[c_pack]
                            file_type = ".cif" if args.write_to_cif==1 else ".pdb"
                            save_path = output_packed + name.split('.')[0] + args.packed_suffix + "_" + str(ix_suffix) + "_" + str(c_pack+1) + file_type + args.file_ending

                            save_output_structure(
                                save_path,
                                X_stack[ix].cpu().numpy(),
                                X_m_stack[ix].cpu().numpy(),
                                b_factor_stack[ix].cpu().numpy(),
                                protein_dict["R_idx"],
                                protein_dict["chain_letters"],
                                S_stack[ix].cpu().numpy(),
                                other_atoms=protein_dict["other_atoms"],
                                icodes=icodes,
                                force_hetatm=args.force_hetatm
                            )

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # fmt: off
    argparser.add_argument("--model_type", type=str, default="mpnn/configs/ligand_mpnn.py", help="config file path")
    argparser.add_argument("--checkpoint", type=str, default=None, help="Path to model weights.")

    argparser.add_argument("--enhance", type=str, default=None, help="Property to enhance, e.g. 'pLDDT'.")
    argparser.add_argument("--overwrite", type=int, default=1, help="Overwrite existing files.")

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

    argparser.add_argument("--tmbb_surface_condition", type=str, default="", help="Per-residue surface condition for tmbb model, 0-unlabled, 1-interface, 2-buried, 3-non-TMBB. e.g. '0 0 1 2'")
    argparser.add_argument("--tmbb_surface_condition_multi", type=str, default="", help="Path to json mapping of tmbb surface condition for each pdb i.e., {'/path/to/pdb': [0, 0, 1, 2]}")

    argparser.add_argument("--symmetry_residues", type=str, default="", help="Add list of lists for which residues need to be symmetric, e.g. 'A12,A13,A14|C2,C3|A5,B6'")
    argparser.add_argument("--symmetry_residues_multi", type=str, default="", help="Path to json mapping of symmetry residues for each pdb i.e., {'/path/to/pdb': 'A12,B12|A13,B13|A14,B14'}")
    argparser.add_argument("--symmetry_weights", type=str, default="", help="Add weights that match symmetry_residues, e.g. '1.01,1.0,1.0|-1.0,2.0|2.0,2.3'")
    argparser.add_argument("--symmetry_weights_multi", type=str, default="", help="Path to json mapping of symmetry weights for each pdb i.e., {'/path/to/pdb': '2.0,1.0|2.0,1.0|2.0,1.0'}")
    argparser.add_argument("--homo_oligomer", type=int, default=0, help="Setting this to 1 will automatically set --symmetry_residues and --symmetry_weights to do homooligomer design with equal weighting.")

    argparser.add_argument("--out_folder", type=str, default="output", help="Path to a folder to output sequences, e.g. /home/out/")
    argparser.add_argument("--file_ending", type=str, default="", help="adding_string_to_the_end")
    argparser.add_argument("--checkpoint_path_sc", type=str, default="/projects/ml/struc2seq/ligandMPNN_models/b_v1/s_300756.pt", help="Path to model weights.")
    argparser.add_argument("--packed_suffix", type=str, default="_packed", help="Suffix for packed PDB paths")
    argparser.add_argument("--force_hetatm", type=int, default=1, help="Deprecated argument, kept only for backwards compatibility.")
    argparser.add_argument("--zero_indexed", type=str, default=0, help="1 - to start output PDB numbering with 0")
    argparser.add_argument("--seed", type=int, default=0, help="Set seed for torch, numpy, and python random.")
    argparser.add_argument("--batch_size", type=int, default=1, help="Number of sequence to generate per one pass.")
    argparser.add_argument("--number_of_batches", type=int, default=1, help="Number of times to design sequence using a chosen batch size.")
    argparser.add_argument("--temperature", type=float, default=0.1, help="Temperature to sample sequences.")
    argparser.add_argument("--save_stats", type=int, default=0, help="Save output statistics")
    argparser.add_argument("--write_to_cif", type=int, default=0, help="Whether to write to CIF. If 0 (the default), will write to .pdb format. If 1, will write to .cif format.")

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
    argparser.add_argument("--threonine_serine_threshold", type=float, default=0.1, help="If the percentage of threonines and serines in the output sequence is above this threshold, the sequence will be discarded and a new one will be generated by biasing against serines and threonines. Set to 1000 to effectively disable this feature.")
    argparser.add_argument("--serthre_bias", type=float, default=-2.0, help="Bias value to be used against serines and threonines if the threonine_serine_threshold is exceeded.")
    
    
    # fmt: on
    args = argparser.parse_args()
    main(args)
