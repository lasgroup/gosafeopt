from enum import Enum
from gosafeopt.aquisitions.safe_opt import SafeOpt
from gosafeopt.tools.misc import singleton
import torch
import gosafeopt
from torch import Tensor
from gosafeopt.tools.data import Data
from typing import Optional

from botorch.models.pairwise_gp import GPyTorchPosterior


class OptimizationStep(Enum):
    LOCAL = 1
    GLOBAL = 2


class GoSafeOpt(SafeOpt):
    def __init__(self, model, config: dict, data: Data, context: Optional[Tensor] = None):
        super().__init__(model, config, context, data)
        GoSafeOpt.n_max_local: int = config["n_max_local"]
        GoSafeOpt.n_max_global: int = config["n_max_global"]
        GoSafeOpt.n = 0

    @classmethod
    def go_to_local_exploration(cls):
        cls.n = -1

    @classmethod
    def get_exploration_phase(cls):
        if cls.n < cls.n_max_local:
            return OptimizationStep.LOCAL
        elif cls.n < cls.n_max_global + cls.n_max_local:
            return OptimizationStep.GLOBAL

    @classmethod
    def advance(cls):
        cls.n += 1
        cls.n %= cls.n_max_local + cls.n_max_global

    @property
    def n_steps(self) -> int:
        if GoSafeOpt.get_exploration_phase() == OptimizationStep.LOCAL:
            return 3
        else:
            return 1

    def override_set_initialization(self) -> bool | str:
        if GoSafeOpt.get_exploration_phase() == OptimizationStep.GLOBAL:
            return "random"
        else:
            return super().override_set_initialization()

    def is_internal_step(self, step: int = 0):
        if GoSafeOpt.get_exploration_phase() == OptimizationStep.LOCAL:
            return super().is_internal_step(step)
        else:
            return False

    def evaluate(self, X: Tensor, step: int = 0) -> Tensor:
        if GoSafeOpt.get_exploration_phase() == OptimizationStep.LOCAL:
            return super().evaluate(X, step)
        else:
            return self.s3(X)

    def after_optimization(self):
        GoSafeOpt.advance()

    def s3(self, X: Tensor):
        data = self.data.train_x
        if self.data.failed_k is not None:
            data = torch.vstack([data, self.data.failed_k])

        # TODO: rethink this
        distance = self.model.models[0].covar_module.covar_dist(data.to(gosafeopt.device), X).min(axis=0)[0]
        return distance
