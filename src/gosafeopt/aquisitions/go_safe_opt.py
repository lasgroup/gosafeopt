from enum import Enum
from typing import Optional

import torch
from torch import Tensor

import gosafeopt
from gosafeopt.aquisitions.safe_opt import SafeOpt
from gosafeopt.tools.data import Data


class OptimizationStep(Enum):
    LOCAL = 1
    GLOBAL = 2


class GoSafeOpt(SafeOpt):
    def __init__(
        self,
        dim_obs: int,
        scale_beta: float,
        beta: float,
        n_max_local: int,
        n_max_global: int,
        data: Data,
        context: Optional[Tensor] = None,
    ):
        super().__init__(dim_obs, scale_beta, beta, context)
        self.data = data
        GoSafeOpt.n_max_global = n_max_global
        GoSafeOpt.n_max_local = n_max_local
        GoSafeOpt.n = 0

    @classmethod
    def go_to_local_exploration(cls) -> None:  # noqa: ANN102
        cls.n = -1

    @classmethod
    def get_exploration_phase(cls) -> OptimizationStep:  # noqa: ANN102
        if cls.n < cls.n_max_local:
            return OptimizationStep.LOCAL
        elif cls.n < cls.n_max_global + cls.n_max_local:
            return OptimizationStep.GLOBAL
        else:
            raise Exception("Phase not found")

    @classmethod
    def advance(cls) -> None:  # noqa: ANN102
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

    def evaluate(self, x: Tensor, step: int = 0) -> Tensor:
        if GoSafeOpt.get_exploration_phase() == OptimizationStep.LOCAL:
            return super().evaluate(x, step)
        else:
            return self.s3(x)

    def after_optimization(self):
        GoSafeOpt.advance()

    def s3(self, x: Tensor):
        if self.model is None:
            raise Exception("Model is nod initialized")
        if self.data.train_x is None:
            raise Exception("Train data is none")

        data = self.data.train_x
        if self.data.failed_k is not None:
            data = torch.vstack([data, self.data.failed_k])

        # TODO: rethink this
        distance = self.model.models[0].covar_module.covar_dist(data.to(gosafeopt.device), x).min(axis=0)[0]
        return distance
