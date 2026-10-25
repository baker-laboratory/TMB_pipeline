import argparse
import os


def parse_args_mpnn():
    parser = argparse.ArgumentParser(description="Training MPNN")

    parser.add_argument("--RESULT_DIR", type=str, default="output/training_mpnn/")

    #### Dataset params
    # define pdb_data
    parser.add_argument("--CIF_PATH", type=str, default="/databases/TrRosetta/cif")
    parser.add_argument("--CACHE_FOLDER", type=str, default=None)
    parser.add_argument("--PROTEIN_BACKBONE_OCC_CUTOFF", type=float, default=0.8)
    parser.add_argument("--PROTEIN_SIDE_CHAIN_OCC_CUTOFF", type=float, default=0.5)
    parser.add_argument("--LIGAND_OCC_CUTOFF", type=float, default=0.01)
    parser.add_argument("--NUM_CONTEXT_ATOMS", type=int, default=25)

    # define sampler
    parser.add_argument("--DATA_DIR", type=str, default="./data")
    parser.add_argument("--DF_PATH_TRAIN", type=str, default="train_3.5A_v0.csv")
    parser.add_argument("--NPZ_PATH_TRAIN", type=str, default="train_3.5A_ls_reCalLen.npz")
    parser.add_argument("--DF_PATH_VALID", type=str, default="valid_3.5A_v0.csv")
    parser.add_argument("--NPZ_PATH_VALID", type=str, default="valid_3.5A_ls_reCalLen.npz")
    parser.add_argument("--BATCH_TOKENS", type=int, default=6000)
    parser.add_argument("--MIN_PROTEIN_LEN_CUTOFF", type=int, default=30)
    parser.add_argument("--MAX_NUM_OF_PDB_VALID", type=int, default=1000000)
    parser.add_argument("--DATE_CUTOFF", type=str, default="2022-Jan-01")

    # define dataloader
    parser.add_argument("--BATCH_SIZE", type=int, default=1)
    parser.add_argument("--NUM_WORKERS", type=int, default=6)

    #### Model params
    parser.add_argument("--PREV_CHECKPOINT", type=str, default=None)
    parser.add_argument("--MODEL_TYPE", type=str, default="ligand_mpnn")
    parser.add_argument("--HIDDEN_DIM", type=int, default=128)
    parser.add_argument("--NUM_ENCODER_LAYERS", type=int, default=3)
    parser.add_argument("--NUM_DECODER_LAYERS", type=int, default=3)
    parser.add_argument("--NUM_NEIGHBORS", type=int, default=32)
    parser.add_argument("--DROPOUT", type=float, default=0.1)
    parser.add_argument("--BACKBONE_NOISE", type=float, default=0.1)

    #### Model training
    parser.add_argument("--MAX_EPOCHS", type=int, default=200)
    parser.add_argument("--LR", type=float, default=3e-3)

    args = parser.parse_args()

    # update path
    args.DF_PATH_TRAIN = os.path.join(args.DATA_DIR, args.DF_PATH_TRAIN)
    args.NPZ_PATH_TRAIN = os.path.join(args.DATA_DIR, args.NPZ_PATH_TRAIN)
    args.DF_PATH_VALID = os.path.join(args.DATA_DIR, args.DF_PATH_VALID)
    args.NPZ_PATH_VALID = os.path.join(args.DATA_DIR, args.NPZ_PATH_VALID)

    return args
