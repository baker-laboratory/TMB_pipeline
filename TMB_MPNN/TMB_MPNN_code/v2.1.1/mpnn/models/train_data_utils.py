"""from {train_folder}/ligand_model_utils_v.py"""

from typing import List

import numpy as np
import torch


def train_featurize(batch, mode="backbone", num_nn=4):
    """
    batch: [[dict_item, int(seq_len)], ...]
        dict_item keys: Y, Y_t, X, X_m, S, R_idx, chain_labels, protein_L
    """
    batch = [b for b in batch if type(b[0]) != list]
    B = len(batch)
    if B > 0:
        L_stack = torch.stack([b[1] for b in batch]).to("cuda")
        L_max = torch.max(L_stack)
        device = L_max.device
        if mode == "backbone":
            X = torch.zeros([B, L_max, 4, 3], dtype=torch.float32, device=device)
            X_m = torch.zeros([B, L_max, 4], dtype=torch.bool, device=device)
        elif mode == "all":
            X = torch.zeros([B, L_max, 37, 3], dtype=torch.float32, device=device)
            X_m = torch.zeros([B, L_max, 37], dtype=torch.bool, device=device)
        else:
            raise ValueError(f"mode {mode} not recognized")

        Y = torch.zeros([B, L_max, num_nn, 3], dtype=torch.float32, device=device)
        Y_m = torch.zeros([B, L_max, num_nn], dtype=torch.bool, device=device)
        Y_t = torch.zeros([B, L_max, num_nn], dtype=torch.long, device=device)

        mask = torch.zeros([B, L_max], dtype=torch.bool, device=device)
        S = 20 * torch.ones([B, L_max], dtype=torch.long, device=device)
        R_idx = -100 * torch.ones([B, L_max], dtype=torch.long, device=device)
        chain_labels = -1 * torch.ones([B, L_max], dtype=torch.int64, device=device)
        for i, b in enumerate(batch):
            out_dict = b[0]
            X[i, : L_stack[i]] = out_dict["X"][None,]
            X_m[i, : L_stack[i]] = out_dict["X_m"][None,]
            mask[i, : L_stack[i]] = torch.ones_like(out_dict["S"][None,], dtype=torch.bool)
            S[i, : L_stack[i]] = out_dict["S"][None,]
            R_idx[i, : L_stack[i]] = out_dict["R_idx"][None,]
            chain_labels[i, : L_stack[i]] = out_dict["chain_labels"][None,]

            num_atoms = out_dict["Y"].shape[-2]
            Y[i, : L_stack[i], :num_atoms] = out_dict["Y"][None,]
            Y_t[i, : L_stack[i], :num_atoms] = out_dict["Y_t"][None,]
            mask_t_0 = out_dict["Y_t"][None,] != 0
            Y_m[i, : L_stack[i], :num_atoms] = mask_t_0 * torch.ones_like(
                out_dict["Y_t"][None,], dtype=torch.bool
            )

        return X, X_m, mask, S, R_idx, chain_labels, Y, Y_t, Y_m

    else:
        raise ValueError("Batch size is 0")


def padding_batch(batch: List[dict], mode="all") -> dict:
    """
    padding and batch
    batch: [dict_item, ...]
        dict_item keys: Y, Y_t, X, X_m, S, R_idx, chain_labels, protein_L
    """
    B = len(batch)
    if B == 0:
        raise ValueError("Batch size is 0")

    L_stack = torch.tensor([len(b["S"]) for b in batch])
    L_max = torch.max(L_stack)

    assert mode == "all"

    if mode == "backbone":
        X = torch.zeros([B, L_max, 4, 3], dtype=torch.float32)
        X_m = torch.zeros([B, L_max, 4], dtype=torch.bool)
    elif mode == "all":
        X = torch.zeros([B, L_max, 37, 3], dtype=torch.float32)
        X_m = torch.zeros([B, L_max, 37], dtype=torch.bool)
    else:
        raise ValueError(f"mode {mode} not recognized")

    num_nn = max(batch[i]["Y"].shape[-2] for i in range(B))
    Y = torch.zeros([B, L_max, num_nn, 3], dtype=torch.float32)
    Y_m = torch.zeros([B, L_max, num_nn], dtype=torch.bool)
    Y_t = torch.zeros([B, L_max, num_nn], dtype=torch.long)

    mask = torch.zeros([B, L_max], dtype=torch.bool)
    S = 20 * torch.ones([B, L_max], dtype=torch.long)
    R_idx = -100 * torch.ones([B, L_max], dtype=torch.long)
    chain_labels = -1 * torch.ones([B, L_max], dtype=torch.int64)
    for i, out_dict in enumerate(batch):
        # convert to tensor
        out_dict = {
            k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v)
            for k, v in out_dict.items()
        }

        X[i, : L_stack[i]] = out_dict["X"][None,]
        X_m[i, : L_stack[i]] = out_dict["X_m"][None,]
        mask[i, : L_stack[i]] = torch.ones_like(out_dict["S"][None,], dtype=torch.bool)
        S[i, : L_stack[i]] = out_dict["S"][None,]
        R_idx[i, : L_stack[i]] = out_dict["R_idx"][None,]
        chain_labels[i, : L_stack[i]] = out_dict["chain_labels"][None,]

        num_atoms = out_dict["Y"].shape[-2]
        Y[i, : L_stack[i], :num_atoms] = out_dict["Y"][None,]
        Y_t[i, : L_stack[i], :num_atoms] = out_dict["Y_t"][None,]
        mask_t_0 = out_dict["Y_t"][None,] != 0
        Y_m[i, : L_stack[i], :num_atoms] = mask_t_0 * torch.ones_like(
            out_dict["Y_t"][None,], dtype=torch.bool
        )

    # return X, X_m, mask, S, R_idx, chain_labels, Y, Y_t, Y_m
    out_dict = dict(
        X=X,
        X_m=X_m,
        mask=mask,
        S=S,
        R_idx=R_idx,
        chain_labels=chain_labels,
        Y=Y,
        Y_t=Y_t,
        Y_m=Y_m,
    )
    other_keys = set(batch[0].keys()) - set(out_dict.keys())
    for k in other_keys:
        out_dict[k] = [b[k] if k in b else None for b in batch]
    return out_dict
