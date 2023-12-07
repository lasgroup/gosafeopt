from typing import Optional

import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.models import ModelListGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from rich.progress import track
from torch import Tensor

from gosafeopt.aquisitions.base_aquisition import BaseAquisition
from gosafeopt.experiments.experiment import Experiment
from gosafeopt.models.model import ModelGenerator
from gosafeopt.optim.base_optimizer import BaseOptimizer, SafeSet
from gosafeopt.tools.data import Data
from gosafeopt.tools.data_logger import WandbLogger
from gosafeopt.tools.logger import Logger


class Trainer:
    def __init__(
        self,
        dim_params: int,
        dim_obs: int,
        n_opt_samples: int,
        save_interval: int = 5,
        show_progress: bool = False,
        refit_interval: int = 0,
        context: Optional[Tensor] = None,
        logger: Optional[WandbLogger] = None,
        data: Optional[Data] = None,
    ):
        self.dim_params = dim_params
        self.dim_obs = dim_obs
        self.n_opt_samples = n_opt_samples
        self.show_progress = show_progress
        self.save_interval = save_interval
        self.refit_interval = refit_interval

        self.logger = logger
        self.data = Data() if data is None else data
        self.context = context
        self.reward_max = -1e10 * torch.ones(self.dim_obs)  # Something small
        self.best_k = None

    # TODO: make parameters explicit instead of config
    def train(
        self,
        experiment: Experiment,
        model_generator: ModelGenerator,
        optimizer: BaseOptimizer,
        aquisition: BaseAquisition,
        safe_points: torch.Tensor,
    ):
        param = np.zeros(self.dim_params)

        for_expression = (
            track(range(self.n_opt_samples), description="Training...")
            if self.show_progress
            else range(self.n_opt_samples)
        )

        model = None

        for episode in for_expression:
            # Collect data from known safe points
            if safe_points is not None and episode < safe_points.shape[0]:
                [param, acf_val] = [safe_points[episode], torch.tensor([0])]
                reward, trajectory, backup_triggered, info = experiment.rollout(param, episode)
                self.data.append_data(param.reshape(1, -1), reward.reshape(1, -1))
            # Optimization loop
            else:
                # For now we generate new model since condition_on_observations has a problem with input transforms...
                model = model_generator.generate(self.data)

                aquisition.update_model(model)
                aquisition.before_optimization()
                self.refit_reward_model(model, episode)

                param, acf_val = optimizer.next_params()
                Logger.info(f"{episode}/{self.n_opt_samples} next k: {param}")

                reward, trajectory, backup_triggered, info = experiment.rollout(param, episode)

                aquisition.after_optimization()

                if not backup_triggered:
                    self.data.append_data(param.reshape(1, -1), reward.reshape(1, -1))

            if torch.any(reward[1:] < 0):
                Logger.warn(f"Constraint violated at iteration {episode} with {reward} at {param}")

            if not backup_triggered:
                if reward[0] > self.reward_max[0]:
                    self.reward_max = reward
                    self.best_k = param
                    experiment.env.best_param = param
                    Logger.success(f"New minimum at Iteration: {episode},yMin:{self.reward_max} at {param}")
                else:
                    Logger.info(f"Reward at Iteration: {episode},y:{reward} at {param}")

                SafeSet.calculate_current_set(reward[0])

                if experiment.backup is not None:
                    self.data.append_backup(trajectory, reward, param)

            if self.logger is not None:
                self.logger.log(param, reward, self.reward_max, acf_val, backup_triggered, self.data, info, episode)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def refit_reward_model(self, model: ModelListGP, episode: int):
        if self.refit_interval != 0 and episode % self.refit_interval == 0:
            # Only fit reward model
            mll = ExactMarginalLogLikelihood(model.models[0].likelihood, model.models[0])
            Logger.info("Fitting GP")
            fit_gpytorch_mll(mll)
            Logger.info(f"New Lenghtscale: {model.models[0].covar_module.base_kernel.lengthscale}")
            self.state_dict = model.state_dict()
