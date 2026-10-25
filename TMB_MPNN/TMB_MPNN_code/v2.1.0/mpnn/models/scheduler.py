import math
from typing import Callable, List, Optional, Sequence, Union

from torch.optim import Optimizer

from mmengine.optim import BaseOptimWrapper
from mmengine.registry import PARAM_SCHEDULERS
from mmengine.optim.scheduler.param_scheduler import _ParamScheduler, INF
from mmengine.optim.scheduler.lr_scheduler import LRSchedulerMixin


@PARAM_SCHEDULERS.register_module()
class InverseSquareRootParamScheduler(_ParamScheduler):
    r"""
    self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
    """

    def __init__(self,
                 optimizer: Union[Optimizer, BaseOptimWrapper],
                 param_name: str,
                 model_size: int,
                 warmup: int,
                 factor: int,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):
        self.model_size = model_size
        self.warmup = warmup
        self.factor = factor

        super().__init__(
            optimizer,
            param_name=param_name,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)

    @classmethod
    def build_iter_from_epoch(cls,
                              *args,
                              T_max=None,
                              begin=0,
                              end=INF,
                              by_epoch=True,
                              epoch_length=None,
                              **kwargs):
        """Build an iter-based instance of this scheduler from an epoch-based
        config."""
        assert by_epoch, 'Only epoch-based kwargs whose `by_epoch=True` can ' \
                         'be converted to iter-based.'
        assert epoch_length is not None and epoch_length > 0, \
            f'`epoch_length` must be a positive integer, ' \
            f'but got {epoch_length}.'
        by_epoch = False
        if T_max is not None:
            T_max = T_max * epoch_length
        begin = int(begin * epoch_length)
        if end != INF:
            end = int(end * epoch_length)
        return cls(
            *args,
            T_max=T_max,
            begin=begin,
            end=end,
            by_epoch=by_epoch,
            **kwargs)

    def _get_value(self) -> list:
        """Compute value using chainable form of the scheduler."""

        if self.last_step == 0:
            return [
                group[self.param_name] for group in self.optimizer.param_groups
            ]
        
        new_lr = self.factor * (self.model_size ** (-0.5) *
            min(self.last_step ** (-0.5), self.last_step * self.warmup ** (-1.5)))

        return [new_lr for group in self.optimizer.param_groups]


@PARAM_SCHEDULERS.register_module()
class InverseSquareRootLR(LRSchedulerMixin, InverseSquareRootParamScheduler):
    pass
