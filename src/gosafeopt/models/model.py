from typing import Optional

import gpytorch
import torch
from botorch.models import ModelListGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import ScaleKernel
from gpytorch.means import ConstantMean
from torch import Tensor

import gosafeopt
from gosafeopt.tools.data import Data


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

    def generate(self, data: Data) -> ModelListGP:
        if data.train_x is None or data.train_y is None:
            raise Exception("Data can not be emtpy")

        if self.normalize_input and self.domain_start is not None and self.domain_end is not None:
            input_transform = Normalize(self.dim_input, bounds=torch.vstack([self.domain_start, self.domain_end]))
        else:
            input_transform = None

        models = []

        for i in range(self.dim_obs):
            mean_module = ConstantMean()
            outcome_transform = None

            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            likelihood.noise = self.likelihood_noise
            likelihood.to(gosafeopt.device)

            # TODO: how to update outcome_transform with condition on observation
            if self.normalize_output:
                outcome_transform = Standardize(m=1)
                outcome_transform.train()
                outcome_transform(data.train_y[:, i].reshape(-1, 1))[0]
                outcome_transform.eval()
                mean_module.constant.requires_grad_(False)
                if i > 0:
                    mean_module.constant = outcome_transform(torch.zeros(1, 1))[0]
                else:
                    mean_module.constant = outcome_transform.means[0]

            covar_module = ScaleKernel(gpytorch.kernels.MaternKernel(ard_num_dims=self.dim_input))
            covar_module.base_kernel.lengthscale = torch.tensor(self.lengthscale)

            models.append(
                SingleTaskGP(
                    data.train_x,
                    data.train_y[:, i].reshape(-1, 1),
                    likelihood=likelihood,
                    covar_module=covar_module,
                    mean_module=mean_module,
                    outcome_transform=outcome_transform,
                    input_transform=input_transform,
                ).to(gosafeopt.device)
            )

        model = ModelListGP(*models)

        if self.state_dict is not None:
            model.load_state_dict(self.state_dict)

        return model
