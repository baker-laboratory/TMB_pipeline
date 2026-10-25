from mmengine.config import read_base

with read_base():
    from .ligand_mpnn import model
    from .tmbb_mpnn import test_pipeline


dim = 256

model.update(
    dict(
        node_features=dim,
        edge_features=dim,
        hidden_dim=dim,
        num_encoder_layers=6,
        num_decoder_layers=6,
        k_neighbors=96,
        # augment_eps=0.01,
    )
)


load_from = "/net/scratch/faxue/share/mpnn_enhancer/BigMPNN_20250205-55101c9a.pth"
