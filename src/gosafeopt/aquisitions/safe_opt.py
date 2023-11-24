from botorch.models.pairwise_gp import GPyTorchPosterior
import torch
import gosafeopt
from gosafeopt.aquisitions.base_aquisition import BaseAquisition
from gosafeopt.tools.misc import singleton
from torch.distributions.multivariate_normal import MultivariateNormal
from gosafeopt.tools.data import Data
from typing import Optional
from torch import Tensor


@singleton
class SafeOptState(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.best_lcb = -1e10


class SafeOpt(BaseAquisition):
    def __init__(self, model, config: dict, context: Optional[torch.Tensor] = None, data: Optional[Data] = None):
        super().__init__(model, config=config, context=context, data=data, n_steps=3)

        self.safeOptState = SafeOptState()

    def evaluate(self, X: Tensor, step: int = 0) -> Tensor:
        posterior = self.model_posterior(X)
        match step:
            case 0:
                return self.lower_bound(posterior)
            case 1:
                return self.maximizers(posterior)
            case 2:
                return self.expanders(posterior)
            case _:
                raise NotImplementedError

    def override_set_initialization(self) -> bool | str:
        # TODO: somehow override initialization for global lower_bound
        return super().override_set_initialization()

    def is_internal_step(self, step: int = 0):
        return True if step == 0 else False

    def lower_bound(self, X: GPyTorchPosterior) -> Tensor:
        l, _ = self.get_confidence_interval(X)

        maxLCB = torch.max(l[:, 0])
        if maxLCB > self.safeOptState.best_lcb:
            self.safeOptState.best_lcb = maxLCB

        slack = l - self.fmin

        return l[:, 0] + self.soft_penalty(slack)

    def maximizers(self, X: GPyTorchPosterior) -> Tensor:
        l, u = self.get_confidence_interval(X)
        scale = 1  # if not self.c["normalize_output"] else self.model.models[0].outcome_transform._stdvs_sq[0]
        values = (u - l)[:, 0] / scale
        improvement = u[:, 0] - self.safeOptState.best_lcb

        interest_function = torch.sigmoid(100 * improvement / scale)
        interest_function -= interest_function.min()
        c = interest_function.max() - interest_function.min()
        c[c < 1e-5] = 1e-5
        interest_function /= c

        slack = l - self.fmin
        penalties = self.soft_penalty(slack)

        value = (values + penalties) * interest_function

        return value

    def expanders(self, X: GPyTorchPosterior) -> Tensor:
        l, u = self.get_confidence_interval(X)

        scale = 1  # if not self.c["normalize_output"] else self.model.models[0].outcome_transform._stdvs_sq[0]
        values = (u - l)[:, 0] / scale

        slack = l - self.fmin
        penalties = self.soft_penalty(slack)
        # print(penalties)

        # TODO how to set scale?
        normal = MultivariateNormal(
            loc=torch.zeros_like(slack[:, 1:], device=gosafeopt.device),
            covariance_matrix=torch.eye(slack.shape[1] - 1, device=gosafeopt.device),
        )
        interest_function = normal.log_prob(slack[:, 1:])
        interest_function -= interest_function.min()
        c = interest_function.max() - interest_function.min()
        c[c < 1e-5] = 1e-5
        interest_function /= c

        value = (values + penalties) * interest_function

        return value

    def reset(self):
        self.safeOptState.best_lcb = -1e10
