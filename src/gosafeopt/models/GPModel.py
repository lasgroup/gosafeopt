from botorch.models.gp_regression import SingleTaskGP
import torch
import gpytorch
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize


class ExactGPModel(SingleTaskGP):
    def __init__(self, config, data, state_dict=None):
        self.config = config

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = config["init_variance"]
        mean_module = gpytorch.means.ZeroMean()
        covar_module = gpytorch.kernels.MaternKernel(ard_num_dims=config["dim"])
        covar_module.lengthscale = torch.tensor(config["init_lenghtscale"])

        bounds = torch.vstack([torch.tensor(config["domain_start"]), torch.tensor(config["domain_end"])])
        input_transform = Normalize(config["dim"], bounds=bounds) if config["normalize_input"] else None
        output_transform = Standardize(m=config["dim_obs"]) if config["normalize_output"] else None

        super().__init__(data.train_x, data.train_y, likelihood, covar_module, mean_module,
                         input_transform=input_transform, outcome_transform=output_transform)

        if state_dict is not None:
            self.load_state_dict(state_dict)

    def save(self, path):
        torch.save(self.state_dict(), path)
