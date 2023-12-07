import json
import random
from pathlib import Path

import numpy as np
import torch
import typer
import wandb
from gosafeopt.aquisitions.go_safe_opt import GoSafeOpt
from gosafeopt.experiments.backup import GoSafeOptBackup
from gosafeopt.experiments.experiment import Experiment
from gosafeopt.models.model import ModelGenerator
from gosafeopt.optim.swarm_opt import SwarmOpt
from gosafeopt.tools.data import Data
from gosafeopt.tools.data_logger import WandbLogger
from gosafeopt.tools.logger import Logger
from gosafeopt.trainer import Trainer
from pendulum.environments import PendulumGymEnv
from torch import Tensor

app = typer.Typer()


def get_configs(
    config_path: str = f"{Path().absolute()}/examples/config.json",
    config_path_pendulum: str = f"{Path().absolute()}/examples/config_pendulum.json",
):
    with open(config_path) as f:
        config = json.load(f)
    with open(config_path_pendulum) as f:
        config_pendulum = json.load(f)

    return config, config_pendulum


@app.command()
def train():
    config, config_pendulum = get_configs()
    config.update(config_pendulum)
    Logger.set_verbosity(4)

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    data = Data()

    render_mode = "human"

    context = torch.tensor([1.0])
    x_safe = torch.tensor([[-1, 1, context]])

    backup = GoSafeOptBackup(data=data, **config["GoSafeOptBackupStrategy"])
    environment = PendulumGymEnv(config, config_pendulum, render_mode=render_mode)
    experiment = Experiment(config, environment, data=data, backup=backup)
    logger = WandbLogger(config, **config["wandb"])
    trainer = Trainer(
        **config["Trainer"], dim_obs=config["dim_obs"], dim_params=config["dim_params"], data=data, logger=logger
    )
    aquisition = GoSafeOpt(**config["GoSafeOpt"], data=data, context=context, dim_obs=config["dim_obs"])

    model = ModelGenerator(
        **config["model"],
        domain_start=Tensor(config["domain_start"]),
        domain_end=Tensor(config["domain_end"]),
        dim_obs=config["dim_obs"],
        dim_model=config["dim_model"],
    )

    optimizer = SwarmOpt(
        aquisition,
        **config["Optimization"],
        context=context,
        domain_start=Tensor(config["domain_start"]),
        domain_end=Tensor(config["domain_end"]),
        dim_params=config["dim_params"],
        dim_context=config["dim_context"],
        data=data,
    )

    trainer.train(experiment, model, optimizer, aquisition, x_safe)

    wandb.summary["rewardMax"] = trainer.reward_max
    wandb.summary["bestK"] = trainer.best_k
    logger.finish()


if __name__ == "__main__":
    app()
