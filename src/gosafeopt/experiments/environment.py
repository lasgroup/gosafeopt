from abc import ABCMeta, abstractmethod
from typing import Optional

import torch
from gymnasium import Env
from numpy.typing import NDArray


class Environment(Env, metaclass=ABCMeta):
    def __init__(self, best_param: Optional[torch.Tensor], render_mode: Optional[str] = None):
        self.best_param = best_param
        self.render_mode = render_mode

    def backup(self, params: NDArray):
        """Execute backup strategy with backup parameters."""

    def before_experiment(self, params: NDArray):
        """Execute before each loss evaluation."""

    def after_experiment(self):
        """Execute after each loss evaluation."""
