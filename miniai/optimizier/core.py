"""Generic optimizer."""

from typing import Callable, List

from miniai.utils import compose, listify


class Genericoptimizer:
    """A base class for derived optimizer."""

    def __init__(self, params, steppers: List[Callable], **defaults):
        """Initialize an optimizer."""
        self.param_groups = list(params)

        if not isinstance(self.param_groups[0], list):
            self.param_groups = [self.param_groups]

        self.hyper = [{**defaults} for _ in self.param_groups]
        self.steppers = listify(steppers)

    def grad_params(self):
        """Return a tuple contains parameter and hyperparameters for each param group."""
        return [
            (param, hyper)
            for param_group, hyper in zip(self.param_groups, self.hyper)
            for param in param_group
            if param.grad is not None
        ]

    def zero_grad(self):
        """Zero the gradients."""
        for p, _ in self.grad_params():
            # TODO: explain what does .detach do?
            p.grad.detach_()
            p.grad.zero_()

    def step(self):
        """Apply all stepper functions to each parameter in each group."""
        for param, hyper in self.grad_params():
            compose(param, self.steppers, **hyper)
