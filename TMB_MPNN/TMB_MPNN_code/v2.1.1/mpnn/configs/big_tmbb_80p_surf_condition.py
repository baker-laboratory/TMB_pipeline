from mmengine.config import read_base
from mpnn.models.protein_mpnn import TMBBMPNNwithCondition

with read_base():
    from .big_mpnn import model
    from .tmbb_mpnn import test_pipeline

model.type = TMBBMPNNwithCondition

load_from = None