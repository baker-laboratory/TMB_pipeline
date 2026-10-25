from openbabel import openbabel
openbabel.obErrorLog.SetOutputLevel(0)
openbabel.cvar.obErrorLog.StopLogging()
import numpy as np
import pandas as pd
import sys
import torch
import time
import json
import os

import cifutils
from ligand_data_utils_sc import PDBDataset, StructureLoader, make_batch_iter 
from ligand_model_utils_v import featurize, loss_smoothed, loss_nll, get_std_opt, ProteinMPNN

JSON = sys.argv[1]
params = json.load(open(JSON))

scaler = torch.cuda.amp.GradScaler()
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

if params["BASE_FOLDER"][-1] != '/':
    params["BASE_FOLDER"] += '/'
if not os.path.exists(params["BASE_FOLDER"]):
    os.makedirs(params["BASE_FOLDER"])

logfile = params["BASE_FOLDER"] + 'log.txt'
if not params["PREV_CHECKPOINT"]:
    with open(logfile, 'w') as f:
        f.write('Epoch\tTrain\tValidation\n')

if params["ATOMS_TO_LOAD"] == "backbone":
    atom_list_to_save = ['N', 'CA', 'C', 'O']
    atom37_order = False
elif params["ATOMS_TO_LOAD"] == "all":
    atom_list_to_save = ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD', 'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3', 'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2', 'CZ3', 'NZ', 'OXT']
    atom37_order = True

model = ProteinMPNN(node_features=params["HIDDEN_DIM"],
                    edge_features=params["HIDDEN_DIM"],
                    hidden_dim=params["HIDDEN_DIM"],
                    num_encoder_layers=params["NUM_ENCODER_LAYERS"],
                    num_decoder_layers=params["NUM_DECODER_LAYERS"],
                    k_neighbors=params["NUM_NEIGHBORS"],
                    dropout=params["DROPOUT"],
                    augment_eps=params["BACKBONE_NOISE"],
                    atom37_order=atom37_order,
                    device=device,
                    atom_context_num=params["NUM_CONTEXT_ATOMS"])
model.to(device)


if params["PREV_CHECKPOINT"]:
    try:
        checkpoint = torch.load(params["PREV_CHECKPOINT"])
        total_step = checkpoint['step'] #write total_step from the checkpoint
        save_step = checkpoint['save_step']
        epoch = checkpoint['epoch'] #write epoch from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Starting from step {total_step}")
    except:
        print("LOADING FROM BAD PATH CHECKPOINT")
        total_step = 0
        epoch = 0
        save_step = 0
        params["PREV_CHECKPOINT"] = []
else:
    total_step = 0
    epoch = 0
    save_step = 0


optimizer = get_std_opt(model.parameters(), params["HIDDEN_DIM"], total_step)

if params["PREV_CHECKPOINT"]:
    optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

Parser = cifutils.CIFParser(skip_res=params["EXCLUDE_RES"])
pdb_dataset = PDBDataset(cif_path=params["CIF_PATH"], 
                         parser=Parser, 
                         atom_list_to_save=atom_list_to_save, 
                         parse_only_DNA_RNA=params["PARSE_ONLY_DNA_RNA"],
                         protein_backbone_occ_cutoff=params["PROTEIN_BACKBONE_OCC_CUTOFF"],
                         protein_side_chain_occ_cutoff=params["PROTEIN_SIDE_CHAIN_OCC_CUTOFF"], 
                         excluded_elements=params["EXCLUDED_ELEMENTS"], 
                         ligand_occ_cutoff=params["LIGAND_OCC_CUTOFF"],
                         num_nn = params["NUM_CONTEXT_ATOMS"],
                         parse_only_proteins=params["PARSE_ONLY_PROTEINS"]
                         )

df_valid = pd.read_csv(params["DF_PATH_VALID"])
df_train = pd.read_csv(params["DF_PATH_TRAIN"])

npz_valid = np.load(params["NPZ_PATH_VALID"], allow_pickle=True)['arr_0'].item()
npz_train = np.load(params["NPZ_PATH_TRAIN"], allow_pickle=True)['arr_0'].item()

for e in range(100000):
    batch_iter_valid = make_batch_iter(df_valid, npz_valid, StructureLoader, params["BATCH_TOKENS"], params["MIN_PROTEIN_LENGTH_CUTOFF"], params["MAX_NUMBER_OF_PDBS_VALID"], params["DATE_CUTOFF"])
    batch_iter_train = make_batch_iter(df_train, npz_train, StructureLoader, params["BATCH_TOKENS"], params["MIN_PROTEIN_LENGTH_CUTOFF"], params["MAX_NUMBER_OF_PDBS_TRAIN"], params["DATE_CUTOFF"])

    valid_sampler = torch.utils.data.sampler.BatchSampler(
        batch_iter_valid,
        batch_size=1,
        drop_last=False)

    train_sampler = torch.utils.data.sampler.BatchSampler(
        batch_iter_train,
        batch_size=1,
        drop_last=False)

    valid_loader = torch.utils.data.DataLoader(
        pdb_dataset,
        sampler=valid_sampler,
        num_workers=params["NUM_WORKERS"],
        pin_memory=True)

    train_loader = torch.utils.data.DataLoader(
        pdb_dataset,
        sampler=train_sampler,
        num_workers=params["NUM_WORKERS"],
        pin_memory=True)

    model.train()
    e = epoch + e
    train_sum, train_weights = 0., 0.
    train_acc = 0.
    t0 = time.time()
    for ix, batch in enumerate(train_loader):
        optimizer.zero_grad()
        feature_dict = featurize(batch, device, params["ATOMS_TO_LOAD"], params["NUM_CONTEXT_ATOMS"])
        if type(feature_dict)==str:
            continue
        S = feature_dict["S"]
        mask = feature_dict["mask"]
        S_mask = 1-(S==20).long()
        mask_residues = (torch.rand([S.shape[0],S.shape[1]], device=device) > params["FRACTION_OF_GIVEN_RESIDUES"]).long()
        feature_dict["mask_residues"] = mask_residues
        if np.random.randint(2, size=1)[0]:
            feature_dict["use_sc"] = params["FRACTION_OF_GIVEN_RESIDUES"]>=0.00001
        else:
            feature_dict["use_sc"] = False
        mask_for_loss = mask*S_mask*mask_residues
        feature_dict["mask_for_loss"] = mask_for_loss
          
        if params["MIXED_PRECISION"]:
            with torch.cuda.amp.autocast():
                log_probs = model(feature_dict)
                _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss, weight=params["LABEL_SMOOTHING"], tokens=params["LOSS_TOKENS"])
            scaler.scale(loss_av_smoothed).backward()

            if params["GRADIENT_NORM"] > 0.0:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), params["GRADIENT_NORM"])

            scaler.step(optimizer)
            scaler.update()

        loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
        train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
        train_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
        train_weights += torch.sum(mask_for_loss).cpu().data.numpy()
        total_step += 1

    model.eval()
    valid_sum, valid_weights = 0., 0.
    valid_acc = 0.
    t1 = time.time()
    with torch.no_grad():
        for ix, batch in enumerate(valid_loader):
            feature_dict = featurize(batch, device, params["ATOMS_TO_LOAD"], params["NUM_CONTEXT_ATOMS"])
            S = feature_dict["S"]
            if type(feature_dict)==str:
                continue
            S = feature_dict["S"]
            mask = feature_dict["mask"]
            S_mask = 1-(S==20).long()
            mask_residues = (torch.rand([S.shape[0],S.shape[1]], device=device) > params["FRACTION_OF_GIVEN_RESIDUES"]).long()
            feature_dict["mask_residues"] = mask_residues
            feature_dict["use_sc"] = params["FRACTION_OF_GIVEN_RESIDUES"]>=0.00001
            mask_for_loss = mask*S_mask*mask_residues
            feature_dict["mask_for_loss"] = mask_for_loss    

            log_probs = model(feature_dict)
            _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss, weight=params["LABEL_SMOOTHING"], tokens=params["LOSS_TOKENS"])

            loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
            valid_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
            valid_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
            valid_weights += torch.sum(mask_for_loss).cpu().data.numpy()

    t2 = time.time()
    train_loss = train_sum / train_weights
    train_accuracy = train_acc / train_weights
    train_perplexity = np.exp(train_loss)

    valid_loss = valid_sum / valid_weights
    valid_accuracy = valid_acc / valid_weights
    valid_perplexity = np.exp(valid_loss)


    train_perplexity_ = np.format_float_positional(np.float32(train_perplexity), unique=False, precision=3)
    valid_perplexity_ = np.format_float_positional(np.float32(valid_perplexity), unique=False, precision=3)

    train_accuracy_ = np.format_float_positional(np.float32(train_accuracy), unique=False, precision=3)
    valid_accuracy_ = np.format_float_positional(np.float32(valid_accuracy), unique=False, precision=3)

    train_dt = np.format_float_positional(np.float32(t1-t0), unique=False, precision=3)
    valid_dt = np.format_float_positional(np.float32(t2-t1), unique=False, precision=3)

    f_string = f"epoch: {e+1}, step: {total_step}, train_time: {train_dt}, valid_time: {valid_dt}, train: {train_perplexity_}, valid: {valid_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {valid_accuracy_}"

    with open(logfile, 'a') as f:
        f.write(f_string + "\n")
    print(f_string)
    torch.save({'epoch': e+1,
                'step': total_step,
                'save_step': save_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.optimizer.state_dict(),
                }, params["BASE_FOLDER"]+'last.pt')
    
    if total_step > save_step + params["SAVE_EVERY_N_STEPS"]:
        save_step += params["SAVE_EVERY_N_STEPS"]
        torch.save({'epoch': e+1,
                    'step': total_step,
                    'save_step': save_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.optimizer.state_dict(),
                    }, params["BASE_FOLDER"]+f's_{total_step}.pt')
    if total_step > params["TOTAL_STEPS"]:
        break
