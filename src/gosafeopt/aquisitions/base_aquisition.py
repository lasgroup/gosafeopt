from abc import abstractmethod
from botorch.acquisition.proximal import ModelListGP
from botorch.models.pairwise_gp import GPyTorchPosterior
from botorch.posteriors.transformed import Posterior
from joblib.externals.loky import backend
import torch
from botorch.acquisition import AnalyticAcquisitionFunction
import gpytorch
from torch import Tensor
from typing import Optional, Tuple
from gosafeopt.tools.data import Data

from abc import ABC, abstractmethod
import gosafeopt


class BaseAquisition(ABC):
    def __init__(
        self, model, config: dict, context: Optional[Tensor] = None, data: Optional[Data] = None, n_steps: int = 1
    ):
        self.model = model
        self.config = config
        self.context = context
        self.data = data
        self.steps = n_steps

        self.fmin = torch.zeros(config["dim_obs"]).to(gosafeopt.device)

    def model_posterior(self, X: Tensor) -> GPyTorchPosterior:
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            with gpytorch.settings.fast_pred_samples():
                x = self.model.posterior(X)
                return x

    @abstractmethod
    def evaluate(self, X: Tensor, step: int = 0) -> Tensor:
        pass

    def after_optimization(self):
        pass

    def before_optimization(self):
        pass

    def override_set_initialization(self) -> bool | str:
        """With this method an aquisition function can override the set initialization"""
        return False

    def is_internal_step(self, step: int = 0):
        """Should return if the result of the aquisition step should be appended to the possible maximizers.
        Useful i.e for calculating global lower bounds as a step of an aquisition.
        """
        if step == 0:
            return True
        else:
            raise NotImplementedError

    def eval(self):
        """Put model in fast evaluation mode."""
        self.model.eval()

    def get_confidence_interval(self, posterior: GPyTorchPosterior) -> Tuple[Tensor, Tensor]:
        mean = posterior.mean.reshape(-1, self.config["dim_obs"])
        var = posterior.variance.reshape(-1, self.config["dim_obs"])

        # Upper and lower confidence bound
        l = mean - self.config["scale_beta"] * torch.sqrt(self.config["beta"] * var)
        u = mean + self.config["scale_beta"] * torch.sqrt(self.config["beta"] * var)

        return l, u

    def safe_set(self, X: Tensor) -> Tensor:
        posterior = self.model_posterior(X)

        l, _ = self.get_confidence_interval(posterior)

        S = torch.all(l[:, 1:] > self.fmin[1:], axis=1)
        return S

    def has_safe_points(self, X: Tensor) -> Tensor:
        return torch.any(self.safe_set(X))

    def soft_penalty(self, slack: Tensor) -> Tensor:
        penalties = torch.clip(slack, None, 0)

        penalties[(slack < 0) & (slack > -0.001)] *= 2
        penalties[(slack <= -0.001) & (slack > -0.1)] *= 5
        penalties[(slack <= -0.1) & (slack > -1)] *= 10

        slack_id = slack < -1
        penalties[slack_id] = -300 * penalties[slack_id] ** 2
        # penalties *= 10000
        return torch.sum(penalties[:, 1:], axis=1)
