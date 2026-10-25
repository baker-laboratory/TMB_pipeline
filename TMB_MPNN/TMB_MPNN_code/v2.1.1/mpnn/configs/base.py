from mmpretrain.visualization import UniversalVisualizer

train_cfg = dict(by_epoch=True, max_epochs=1000, val_interval=10)
val_cfg = dict()
test_cfg = dict()
# auto_scale_lr = dict(base_batch_size=32*8)

default_scope = "mmpretrain"
default_hooks = dict(
    runtime_info=dict(type="RuntimeInfoHook"),
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=5),
    param_scheduler=dict(type="ParamSchedulerHook"),
    # checkpoint=dict(type='CheckpointHook', interval=50, max_keep_ckpts=3, save_best='bind/loss'),
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=3),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="VisualizationHook", enable=False),
)
env_cfg = dict(
    cudnn_benchmark=False, mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0), dist_cfg=dict(backend="nccl")
)
vis_backends = [
    dict(type="LocalVisBackend"),
]
visualizer = dict(
    type=UniversalVisualizer,
    vis_backends=[
        dict(type="LocalVisBackend"),
        dict(type="TensorboardVisBackend")]
)
log_level = "INFO"
load_from = None
resume = False
randomness = dict(seed=None, deterministic=False)
