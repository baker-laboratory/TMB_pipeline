
from typing import Dict
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class UpdateDatasetLengthHook(Hook):
    """
    Used to update the dataset length for the sampler
    Note: will be discarded after refactoring
    """
    
    def before_val_epoch(self, runner):
        num_samples = runner.val_dataloader.sampler.sample_num
        runner.val_dataloader.dataset.sample_num = num_samples

    def before_test_epoch(self, runner):
        num_samples = runner.test_dataloader.sampler.sample_num
        runner.test_dataloader.dataset.sample_num = num_samples


@HOOKS.register_module()
class ResetSamplerHook(Hook):
    """
    Used to reset the sampler for the next epoch
    Note: will be discarded after refactoring
    """
    def after_train_epoch(self, runner) -> None:
        runner.train_dataloader.sampler.reset()
    
    def after_val_epoch(self, runner, metrics=None) -> None:
        runner.val_dataloader.sampler.reset()
    
    def after_test_epoch(self, runner, metrics=None) -> None:
        runner.test_dataloader.sampler.reset()
