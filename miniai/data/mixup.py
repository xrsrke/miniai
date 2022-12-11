"""Implementation of Mixup."""

from functools import partial
from typing import Callable, Optional, Union

import torch
from fastai.callback.core import Callback
from torch.distributions.beta import Beta


def gamma(tensor: torch.Tensor):
    """Gamma function"""
    return tensor.lgamma().exp()


def lin_comb(vector1: torch.Tensor, vector2: torch.Tensor, beta: Union[float, int]):
    """Return a linear combination of vectors v1 and v2 given beta"""
    return beta * vector1 + (1 - beta) * vector2


class NonReduce:
    """Set the loss function's reduction mode to node if enter"""

    def __init__(self, loss_func: Callable):
        self.loss_func = loss_func
        self.old_reduction: Optional[str] = None

    def __enter__(self):
        if hasattr(self.loss_func, "reduction"):
            self.old_reduction = getattr(self.loss_func, "reduction")
            setattr(self.loss_func, "reduction", "none")
            return self.loss_func

        return partial(self.loss_func, NonReduce="none")

    def __exit__(self, _, value, traceback):
        if self.old_reduction is not None:
            setattr(self.loss_func, "reduction", self.old_reduction)


class Mixup(Callback):
    """Implementation of mixup"""

    _order = 90

    def __init__(self, alpha: float = 0.4):
        super().__init__()
        self.distrib = Beta(torch.tensor([alpha]), torch.tensor([alpha]))
        self.old_loss_func: Optional[Callable] = None
        self.y_batch_shuffled: torch.Tensor = None

    def begin_fit(self):
        """Begin fitting"""
        self.old_loss_func = self.run.loss_func
        self.run.loss_func = self.loss_func

    def begin_batch(self):
        """Begin batch"""
        if not self.in_train:
            return

        n_samples = self.yb.size(0)
        alpha = self.distrib.sample((n_samples,)).squeeze().to(self.xb.device)
        shuffle = torch.randperm(n_samples).to(self.xb.device)
        x_batch_shuffled, self.y_batch_shuffled = self.xb[shuffle], self.yb[shuffle]
        self.run.rx = lin_comb(self.xb, x_batch_shuffled, alpha)

    def after_fit(self):
        self.run.loss_func = self.old_loss_func

    def reduce_loss(self, loss, reduction="mean"):
        """Return the mode of loss."""
        return (
            loss.mean()
            if reduction == "mean"
            else loss.sum()
            if reduction == "sum"
            else loss
        )

    def loss_func(self, pred: torch.Tensor, yb: torch.Tensor):
        """X."""
        if not self.in_train and self.old_loss_func is not None:
            return self.old_loss_func(pred, yb)

        with NonReduce(self.old_loss_func) as loss_func:
            loss1 = loss_func(self.pred, yb)
            loss2 = loss_func(self.pred, self.yb1)

        loss = lin_comb(loss1, loss2, self.alpha)
        reduction = getattr(self.old_loss_func, "reduction", "mean")
        return self.reduce_loss(loss, reduction)
