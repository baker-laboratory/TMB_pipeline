from mmengine.config import read_base

from mpnn.models.protein_mpnn import ProteinMPNN
from mpnn.datasets.pdb_dataset import PDBDataset
from mpnn.datasets.sampler import MPNNSampler
from mpnn.datasets.evaluator import MPNNEvaluator
from mpnn.models.hooks import UpdateDatasetLengthHook, ResetSamplerHook
from mpnn.models.scheduler import InverseSquareRootLR

with read_base():
    from .base import *
    from .base import default_hooks


model = dict(
    type=ProteinMPNN,
    model_type='ligand_mpnn',
    node_features=128,
    edge_features=128,
    hidden_dim=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    k_neighbors=32,
    dropout=0.1,
    augment_eps=0.1,  # backbone noise
    atom_context_num=25
)

atoms_to_load = 'all'
if atoms_to_load == "backbone":
    atom_list_to_save = ['N', 'CA', 'C', 'O']
    atom37_order = False
elif atoms_to_load == "all":
    atom_list_to_save = ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD', 'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3', 'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2', 'CZ3', 'NZ', 'OXT']
    atom37_order = True


dataset = dict(
    type=PDBDataset,
    cif_path='/databases/TrRosetta/cif',
    parser_exclude_res_list=["HOH", "NA", "CL", "K", "BR"],
    atom_list_to_save=atom_list_to_save,
    parse_only_DNA_RNA=False,
    protein_backbone_occ_cutoff=0.8,
    protein_side_chain_occ_cutoff=0.5,
    excluded_elements=[1],
    ligand_occ_cutoff=0.01,
    num_nn = 25, # params["NUM_CONTEXT_ATOMS"],
    parse_only_proteins=False
)

train_dataloader = dict(
    batch_size=1,
    num_workers=6,
    dataset=dataset,
    collate_fn=dict(type='default_collate'),
    sampler=dict(
        type=MPNNSampler,
        df_path='data/train_3.5A_v0.csv',
        npz_path='data/train_3.5A_ls_reCalLen.npz',
        # npz_path='data/train_3.5A_ls.npz',
        batch_tokens=6000,
        min_protein_len_cutoff=30,
        max_num_of_pdb_valid=1000000,
        date_cutoff='2022-Jan-01',
        ))

val_dataloader = dict(
    batch_size=1,
    num_workers=6,
    dataset=dataset,
    collate_fn=dict(type='default_collate'),
    sampler=dict(
        type=MPNNSampler,
        df_path='data/valid_3.5A_v0.csv',
        # npz_path='data/valid_3.5A_ls.npz',
        npz_path='data/valid_3.5A_ls_reCalLen.npz',
        batch_tokens=6000,
        min_protein_len_cutoff=30,
        max_num_of_pdb_valid=1000000,
        date_cutoff='2022-Jan-01',
        ))
test_dataloader = val_dataloader

val_evaluator = dict(type=MPNNEvaluator)
test_evaluator = dict(type=MPNNEvaluator)


work_dir = 'output/ligand_mpnn'

lr = 3e-3
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type="Adam", lr=lr, eps=1e-9, weight_decay=0),
    clip_grad=dict(max_norm=1.),
    paramwise_cfg=dict(
        custom_keys={
            "graph_feat_pock_gt_embed": dict(decay_mult=0.0),
            "semantic_embedding": dict(decay_mult=0.0),
            "feat_type_embeddings": dict(decay_mult=0.0),
            "gt_type_embeddings": dict(decay_mult=0.0),
        }
    ),
)
max_epoch = 200
param_scheduler = [
    dict(
        type=InverseSquareRootLR,
        factor=2,
        model_size=128,
        warmup=4000,
        by_epoch=False
    )
    # dict(
    #     type="LinearLR",
    #     start_factor=0.001,
    #     by_epoch=True,
    #     begin=0,
    #     end=5,
    #     # update by iter
    #     convert_to_iter_based=True,
    # ),
    # main learning rate scheduler
    # dict(
    #     type="CosineAnnealingLR",
    #     T_max=max_epoch,
    #     eta_min=lr * 0.1,
    #     begin=5,
    #     by_epoch=True),
    # dict(
    #     type='MultiStepLR',
    #     begin=10,
    #     end=max_epoch,
    #     by_epoch=True,
    #     milestones=[80],
    #     gamma=0.1)
]

train_cfg = dict(by_epoch=True, max_epochs=max_epoch, val_interval=1)
default_hooks.update(dict(
    logger=dict(type="LoggerHook", interval=50),
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=3),
))

custom_hooks = [
    dict(type=UpdateDatasetLengthHook),
    dict(type=ResetSamplerHook)]
