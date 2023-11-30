from botorch.models import ModelListGP
import torch

from botorch.models.gp_regression import ScaleKernel, SingleTaskGP
from gpytorch.means import ConstantMean, ZeroMean, zero_mean
import torch
import gpytorch
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from typing import List, Optional
from torch import Tensor
from gosafeopt.tools.data import Data

import gosafeopt


class ModelGenerator:
    def __init__(
        self,
        dim_model: int,
        dim_obs: int,
        likelihood_noise: Tensor,
        lenghtscale: Tensor,
        normalize_input: bool = False,
        normalize_output: bool = False,
        domain_start: Optional[Tensor] = None,
        domain_end: Optional[Tensor] = None,
        state_dict: Optional[dict] = None,
    ):
        if normalize_input and (domain_start is None or domain_end is None):
            raise Exception("If normalize_input is True domain bounds have to be set")

        self.domain_start = domain_start
        self.domain_end = domain_end
        self.dim_input = dim_model
        self.dim_obs = dim_obs
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.likelihood_noise = likelihood_noise
        self.lengthscale = lenghtscale
        self.state_dict = state_dict

    def generate(self, data) -> ModelListGP:
        input_transform = (
            Normalize(self.dim_input, bounds=torch.vstack([self.domain_start, self.domain_end]))
            if self.normalize_input
            else None
        )

        models = []

        for i in range(self.dim_obs):
            mean_module = ConstantMean()
            outcome_transform = None

            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            likelihood.noise = self.likelihood_noise
            likelihood.to(gosafeopt.device)

            # TODO: how to update outcome_transform with condition on observation
            # if normalize_output:
            #     outcome_transform = Standardize(m=1)
            #     outcome_transform.train()
            #     outcome_transform(self.data.train_y[:, i].reshape(-1, 1))[0]
            #     outcome_transform.eval()
            #     mean_module.constant.requires_grad_(False)
            #     if i > 0:
            #         mean_module.constant = outcome_transform(torch.zeros(1, 1))[0]
            #     else:
            #         mean_module.constant = outcome_transform.means[0]
            #
            covar_module = ScaleKernel(gpytorch.kernels.MaternKernel(ard_num_dims=self.dim_input))
            covar_module.base_kernel.lengthscale = torch.tensor(self.lengthscale)

            models.append(
                SingleTaskGP(
                    data.train_x,
                    data.train_y[:, i].reshape(-1, 1),
                    likelihood,
                    covar_module,
                    mean_module,
                    outcome_transform,
                    input_transform,
                ).to(gosafeopt.device)
            )

        model = ModelListGP(*models)

        if self.state_dict is not None:
            model.load_state_dict(self.state_dict)

        return model
