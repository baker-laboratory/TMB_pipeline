import logging
import os
import time
from typing import Mapping, Sequence

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from mpnn.configs.train_args import parse_args_mpnn
from mpnn.datasets.evaluator import MPNNEvaluator
from mpnn.datasets.pdb_dataset import PDBDataset
from mpnn.datasets.sampler import MPNNSampler
from mpnn.datasets.utils import pseudo_single_batch_collate
from mpnn.models.protein_mpnn import ProteinMPNN
from mpnn.models.scheduler import InverseSquareRootLR


def cast_data(data, device):
    if isinstance(data, Mapping):
        return {key: cast_data(data[key], device) for key in data}
    elif isinstance(data, (str, bytes)) or data is None:
        return data
    elif isinstance(data, tuple) and hasattr(data, "_fields"):
        # namedtuple
        return type(data)(*(cast_data(sample, device) for sample in data))  # type: ignore  # noqa: E501  # yapf:disable
    elif isinstance(data, Sequence):
        return type(data)(cast_data(sample, device) for sample in data)  # type: ignore  # noqa: E501  # yapf:disable
    elif isinstance(data, (torch.Tensor,)):
        return data.to(device)
    else:
        return data


def main(args):

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    result_dir = args.RESULT_DIR

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(result_dir, "training.log"),
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger()

    logger.info("Training Configuration:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")

    ## step 1: prepare the dataset

    pdb_dataset = PDBDataset(
        cif_path=args.CIF_PATH,
        cache_folder=args.CACHE_FOLDER,
        protein_backbone_occ_cutoff=args.PROTEIN_BACKBONE_OCC_CUTOFF,
        protein_side_chain_occ_cutoff=args.PROTEIN_SIDE_CHAIN_OCC_CUTOFF,
        ligand_occ_cutoff=args.LIGAND_OCC_CUTOFF,
        lig_nn_num=args.NUM_CONTEXT_ATOMS,
    )

    train_sampler = MPNNSampler(
        df_path=args.DF_PATH_TRAIN,
        npz_path=args.NPZ_PATH_TRAIN,
        batch_tokens=args.BATCH_TOKENS,
        min_protein_len_cutoff=args.MIN_PROTEIN_LEN_CUTOFF,
        max_num_of_pdb_valid=args.MAX_NUM_OF_PDB_VALID,
        date_cutoff=args.DATE_CUTOFF,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=pdb_dataset,
        batch_size=args.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=args.NUM_WORKERS,
        collate_fn=pseudo_single_batch_collate,
    )

    valid_sampler = MPNNSampler(
        df_path=args.DF_PATH_VALID,
        npz_path=args.NPZ_PATH_VALID,
        batch_tokens=args.BATCH_TOKENS,
        min_protein_len_cutoff=args.MIN_PROTEIN_LEN_CUTOFF,
        max_num_of_pdb_valid=args.MAX_NUM_OF_PDB_VALID,
        date_cutoff=args.DATE_CUTOFF,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=pdb_dataset,
        batch_size=args.BATCH_SIZE,
        sampler=valid_sampler,
        num_workers=args.NUM_WORKERS,
        collate_fn=pseudo_single_batch_collate,
    )

    test_sampler = MPNNSampler(
        df_path=args.DF_PATH_VALID,
        npz_path=args.NPZ_PATH_VALID,
        batch_tokens=args.BATCH_TOKENS,
        min_protein_len_cutoff=args.MIN_PROTEIN_LEN_CUTOFF,
        max_num_of_pdb_valid=args.MAX_NUM_OF_PDB_VALID,
        date_cutoff=args.DATE_CUTOFF,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=pdb_dataset,
        batch_size=args.BATCH_SIZE,
        sampler=test_sampler,
        num_workers=args.NUM_WORKERS,
        collate_fn=pseudo_single_batch_collate,
    )

    ## Step 2: define the model
    model = ProteinMPNN(
        model_type=args.MODEL_TYPE,
        node_features=args.HIDDEN_DIM,
        edge_features=args.HIDDEN_DIM,
        hidden_dim=args.HIDDEN_DIM,
        num_encoder_layers=args.NUM_ENCODER_LAYERS,
        num_decoder_layers=args.NUM_DECODER_LAYERS,
        k_neighbors=args.NUM_NEIGHBORS,
        dropout=args.DROPOUT,
        augment_eps=args.BACKBONE_NOISE,
        atom_context_num=args.NUM_CONTEXT_ATOMS,
    )
    model.to(device)

    Valid_evaluator = MPNNEvaluator()

    optimizer = optim.Adam(model.parameters(), lr=args.LR, eps=1e-9, weight_decay=0)
    scheduler = InverseSquareRootLR(
        optimizer=optimizer, factor=2, model_size=args.HIDDEN_DIM, warmup=4000, by_epoch=False
    )
    scaler = torch.cuda.amp.GradScaler()

    if args.PREV_CHECKPOINT:
        if not os.path.exists(args.PREV_CHECKPOINT):
            raise FileNotFoundError(f"Checkpoint file {args.PREV_CHECKPOINT} not found")
        try:
            checkpoint = torch.load(args.PREV_CHECKPOINT)
            total_step = checkpoint["step"]  # write total_step from the checkpoint
            save_epoch = checkpoint["epoch"]  # write epoch from the checkpoint
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(f"Starting from step {total_step}")
        except Exception as e:
            print(e)
            print("LOADING FROM BAD PATH CHECKPOINT")
            total_step = 0
            save_epoch = 0
            args.PREV_CHECKPOINT = []
    else:
        total_step = 0
        save_epoch = 0

    ## Step 3: training
    checkpoints = []
    for epoch in range(args.MAX_EPOCHS):
        torch.cuda.empty_cache()
        model.train()
        train_sampler.reset()
        for idx, batch in enumerate(train_loader):
            batch = cast_data(batch, device=device)

            optimizer.zero_grad()

            with autocast():
                results = model(**batch, mode="loss")

            scaler.scale(results["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            total_step += 1

            if total_step % 50 == 0:
                output_seq = f"Epoch [{epoch+1+save_epoch}], Step [{total_step}], Loss: {results['loss']:0.4f}, Acc: {results['acc']:0.4f}, Perplexity: {results['perplexity']:0.4f}"
                print(output_seq)
                logger.info(output_seq)

        # validation
        model.eval()
        valid_sampler.reset()
        all_results = []
        for idx, batch in enumerate(valid_loader):
            batch = cast_data(batch, device=device)

            with autocast() and torch.no_grad():
                results = model(**batch, mode="predict")

            for output_data in results:
                result = dict(
                    S=output_data["S"].detach().cpu(),
                    log_probs=output_data["log_probs"].detach().cpu(),
                    mask_for_loss=output_data["mask_for_loss"].detach().cpu(),
                )
                all_results.append(result)

            del result
            del output_data
            del results
            torch.cuda.empty_cache()

        validation_results = Valid_evaluator.compute_metrics(all_results)

        output_seq = f"Validation: Epoch [{epoch+1+save_epoch}], Loss: {validation_results['loss']}, Accuracy: {validation_results['accuracy']}, Perplexity: {validation_results['perplexity']}"
        print(output_seq)
        logger.info(output_seq)

        del all_results
        del validation_results

        ## Step 4: save last 3 models
        checkpoint_path = os.path.join(
            result_dir, f"model_epoch_{epoch+1+save_epoch}_step_{total_step}.pt"
        )
        torch.save(
            {
                "epoch": epoch + 1 + save_epoch,
                "step": total_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            checkpoint_path,
        )

        # create a soft link to the last checkpoint
        last_checkpoint = os.path.join(result_dir, "last.pt")
        if os.path.exists(last_checkpoint):
            os.remove(last_checkpoint)
            time.sleep(0.1)  # avoid FileExistsError
        os.symlink(os.path.realpath(checkpoint_path), last_checkpoint)

        checkpoints.append(checkpoint_path)
        if len(checkpoints) > 3:
            oldest_checkpoint = checkpoints.pop(0)
            if os.path.exists(oldest_checkpoint):
                os.remove(oldest_checkpoint)


if __name__ == "__main__":
    args = parse_args_mpnn()

    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
