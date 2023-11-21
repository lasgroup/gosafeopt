from botorch.models import ModelListGP
from botorch.models.gp_regression import ScaleKernel, SingleTaskGP
from gpytorch.means import ConstantMean, ZeroMean, zero_mean
import torch
import gpytorch
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize

import gosafeopt


def create_model(config, data, state_dict=None):

    bounds = torch.vstack([torch.tensor(config["domain_start"]),
                          torch.tensor(config["domain_end"])])

    input_transform = Normalize(config["dim"], bounds=bounds) if config["normalize_input"] else None

    models = []

    for i in range(config["dim_obs"]):
        mean_module = ConstantMean()
        outcome_transform = None

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = config["init_variance"]
        likelihood.to(gosafeopt.device)

        if config["normalize_output"]:
            outcome_transform = Standardize(m=1)
            outcome_transform.train()
            outcome_transform(data.train_y[:, i].reshape(-1, 1))[0]
            outcome_transform.eval()
            mean_module.constant.requires_grad_(False)
            if i > 0:
                mean_module.constant = outcome_transform(torch.zeros(1, 1))[0]
            else:
                mean_module.constant = outcome_transform.means[0]

        covar_module = ScaleKernel(gpytorch.kernels.MaternKernel(ard_num_dims=config["dim"]))
        covar_module.base_kernel.lengthscale = torch.tensor(config["init_lenghtscale"])

        models.append(SingleTaskGP(data.train_x, data.train_y[:, i].reshape(-1, 1), likelihood,
                      covar_module, mean_module, outcome_transform, input_transform).to(gosafeopt.device))

    model = ModelListGP(*models)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model
