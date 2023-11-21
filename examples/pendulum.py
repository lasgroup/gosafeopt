from botorch import fit_gpytorch_mll
from gpytorch.mlls import SumMarginalLogLikelihood
import typer
import numpy as np
import wandb
from gosafeopt.aquisitions.go_safe_opt import GoSafeOptState
from gosafeopt.aquisitions.max_mean import MaxMean
from gosafeopt.aquisitions.safe_opt_multistage import SafeOptMultiStageState
from gosafeopt.aquisitions.safe_ucb import SafeUCB
from gosafeopt.aquisitions.safe_lcb import SafeLCB
from gosafeopt.experiments.backup import Backup
from gosafeopt.experiments.experiment import Experiment
from gosafeopt.models import create_model
from gosafeopt.optim import get_optimizer
from gosafeopt.optim.base_optimizer import OptimizerState
from gosafeopt.optim.swarm_opt import SwarmOpt
from examples.pendulum.wandblogger import PendulumWandbLogger
from pendulum.environments import PendulumGymEnvWithDynamics
from gosafeopt.tools.data import Data
from gosafeopt.tools.logger import Logger
from gosafeopt.trainer import Trainer
from pendulum.dynamics import U_learned, U_ideal
from pendulum.plot import PlotPendulum, PlotPendulumContour
import torch
from gosafeopt.tools.data_logger import load
from gosafeopt.tools.file import makeGIF
import matplotlib.pyplot as plt
from pendulum.environments import PendulumGymEnv
import json
from joblib import Parallel, delayed
from moviepy.editor import ImageSequenceClip
import random
import gosafeopt
from typing import List

app = typer.Typer()


def getConfigs(
    config_path="examples/config.json",
    config_path_pendulum="examples/config_pendulum.json",
):
    with open(config_path) as f:
        config = json.load(f)["optimization"]
    with open(config_path_pendulum) as f:
        config_pendulum = json.load(f)

    return config, config_pendulum


@app.command()
def make_gif():
    makeGIF()


@app.command()
def plot_gym(
    i: int = typer.Option(-1, help="Which epoch to simulate"),
    ideal: bool = typer.Option(False, help="Plot ideal undisturbed trajectory"),
    data_path: str = typer.Option(..., help="data_path"),
    context: float = typer.Option(1, help="Gate"),
):
    config, config_pendulum = getConfigs()

    context1 = context

    logger = load(f"{data_path}/datalogger.obj")

    config_pendulum["m"] = context1
    config_pendulum["kp"] *= context1
    config_pendulum["kd"] *= context1

    config_pendulum["kp_bo"] = logger.x_buffer[i][0]
    config_pendulum["kd_bo"] = logger.x_buffer[i][1]
    k = np.array([logger.x_buffer[i][0], logger.x_buffer[i][1]])

    U = U_ideal if ideal else U_learned

    env = PendulumGymEnvWithDynamics(config_pendulum, U, render_mode="human")
    experiment = Experiment(config, env)
    experiment.rollout(k)


@app.command()
def plot_contour(
    i: int = typer.Option(-1, help="Which epoch to simulate"),
    data_path: str = typer.Option(..., help="data_path"),
):
    config, config_pendulum = getConfigs()

    logger = load(f"{data_path}/datalogger.obj")

    env = PendulumGymEnvWithDynamics(config_pendulum, U_ideal)
    experiment = Experiment(config, env)

    _, trajectory_ideal, backup_triggered, _ = experiment.rollout(np.zeros(2))

    plotter = PlotPendulumContour(trajectory_ideal, config, config_pendulum)
    if i == -1:
        i = len(logger.x_buffer) - 1
    plotter.plotIdx(logger, i)
    plt.show()


@app.command()
def plot(
    i: int = typer.Option(-1, help="Which epoch to simulate"),
    all: bool = typer.Option(False, help="Plot ideal undisturbed trajectory"),
    data_path: str = typer.Option(..., help="data_path"),
):
    config, config_pendulum = getConfigs()

    logger = load(f"{data_path}/datalogger.obj")

    env = PendulumGymEnvWithDynamics(config_pendulum, U_ideal)
    experiment = Experiment(config, env)

    _, trajectory_ideal, backup_triggered, _ = experiment.rollout(np.zeros(2))

    plotter = PlotPendulum(trajectory_ideal, config, config_pendulum)
    if all:
        plotter.plot(logger)
    else:
        if i == -1:
            i = len(logger.x_buffer) - 1
        plotter.plotIdx(logger, i)
    plt.show()


@app.command()
def train(
    seed: int = typer.Option(None, help="Seed"),
    aquisition: str = typer.Option(None, help="Plot ideal undisturbed trajectory"),
    dry_run: bool = typer.Option(False, help="Plot ideal undisturbed trajectory"),
    data_dir: str = typer.Option(None, help="data_dir"),
    contexts: List[float] = typer.Option([1], help="Gate"),
    wandbdir: str = typer.Option(None, help="wandbdirectory"),
    load_data_dir: str = typer.Option(None, help="Load data directory"),
    safe_opt_interior_lb: float = typer.Option(None, help="Interior lb"),
    safe_opt_marginal_lb: float = typer.Option(None, help="marginal lb"),
    safe_opt_interior_prob: float = typer.Option(None, help="interior prob"),
    safe_opt_marginal_prob: float = typer.Option(None, help="marginal prob"),
    sigma: float = typer.Option(None, help="sigma"),
    beta: float = typer.Option(None, help="Beta"),
    config_path: str = typer.Option("config.json", help="Config path"),
):
    for context in contexts:
        config, config_pendulum = getConfigs(config_path=config_path)
        config.update(config_pendulum)

        config["context"] = context

        Logger.setVerbosity(4)
        Logger.info(
            "Using: {} with device {}".format(config["aquisition"], gosafeopt.device)
        )

        if beta is not None:
            config["beta"] = beta

        if sigma is not None:
            config["safe_opt_sigma"] = sigma

        if safe_opt_interior_lb is not None:
            config["safe_opt_interior_lb"] = safe_opt_interior_lb

        if safe_opt_marginal_lb is not None:
            config["safe_opt_marginal_lb"] = safe_opt_marginal_lb

        if safe_opt_interior_prob is not None:
            config["safe_opt_interior_prob"] = safe_opt_interior_prob

        if safe_opt_marginal_prob is not None:
            config["safe_opt_marginal_prob"] = safe_opt_marginal_prob

        if seed is not None:
            config["seed"] = seed

        if aquisition is not None:
            config["aquisition"] = aquisition

        if dry_run:
            config["n_opt_samples"] = 2

        if wandbdir is not None:
            config["wandbdir"] = wandbdir

        if data_dir is not None:
            config["data_dir"] = data_dir

        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        random.seed(config["seed"])

        data = Data()
        backup = Backup(config, data) if config["aquisition"] == "GoSafeOpt" else None

        config_pendulum["m"] = context
        config_pendulum["kp"] *= context
        config_pendulum["kd"] *= context

        # Environment
        render_mode = "rgb_array_list" if config["log_video"] else None
        environment = PendulumGymEnv(config, config_pendulum, render_mode=render_mode)
        experiment = Experiment(config, environment, backup=backup)

        model = create_model

        context = torch.tensor([1.0])

        x_safe = torch.tensor(
            [
                [-1, 1, context],
                # [0.1, -0.1, context],
                # [-0.1, 0.1, context],
                # [0.1, 0.1, context],
                # [-0.1, -0.1, context],
            ]
        )

        envIdeal = PendulumGymEnvWithDynamics(config_pendulum, U_ideal)
        experimentIdeal = Experiment(config, envIdeal)

        _, trajectory_ideal, backup_triggered, info = experimentIdeal.rollout(
            np.zeros(2)
        )

        logger = PendulumWandbLogger(config, config_pendulum, trajectory_ideal, context)

        trainer = Trainer(config, logger=logger, context=context, data=data)
        trainer.train(experiment, model, x_safe)

        if config["log_video"]:
            seq = ImageSequenceClip("{}/plot".format(wandb.run.dir), fps=4)
            seq.write_gif("{}/plot/animation.gif".format(wandb.run.dir), fps=4)
            wandb.log(
                data={
                    "animation": wandb.Image(
                        "{}/plot/animation.gif".format(wandb.run.dir)
                    )
                }
            )

        OptimizerState(config).reset()
        GoSafeOptState(config).reset()
        SafeOptMultiStageState().reset()

        logger.save("{}/res/datalogger.obj".format(wandb.run.dir))
        # ml(config_path=config_path, data_path="{}/res".format(wandb.run.dir))
        wandb.summary["rewardMax"] = trainer.rewardMax
        wandb.summary["bestK"] = trainer.bestK
        logger.finish()


# @app.command()
def train_seeds():
    seeds = [41, 13, 345, 453, 10987, 4546, 13, 1234, 864, 1265]
    Parallel(n_jobs=5)(delayed(train)(seed) for seed in seeds)


@app.command()
def optimize_context(context: float = typer.Option(1.0, help="Context")):
    with open("config.json") as f:
        config = json.load(f)["optimization"]

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    data = Data()
    data.load(config["save_data"])

    state_dict = torch.load(config["save_model"])

    model = create_model(config, data, state_dict)
    aq = MaxMean(model, config, torch.tensor([[context]]))
    opt = SwarmOpt(aq, config, torch.tensor([[context]]))
    opt.getNextPoint()


@app.command()
def ml(
    config_path: str = typer.Option("config.txt", help="Path to config file"),
    data_path: str = typer.Option("data", help="Path data"),
):
    config, config_pendulum = getConfigs()
    context = torch.tensor([1.0])

    config["swarmopt_n_iterations"] = 100
    config["set_size"] = 5000

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    data = Data()
    data.load(data_path)

    model = create_model(config, data)
    aquisition = SafeUCB(model, config, context, data=data)
    optimizer = get_optimizer(aquisition, config, context)
    k, acf_val = optimizer.getNextPoint()
    Logger.info(f"Best k for context {context} is: {k}")

    env = PendulumGymEnv(config, config_pendulum)
    experiment = Experiment(config, env)
    reward, trajetory, backup, info = experiment.rollout(k)
    Logger.info(f"Best reward for context {context} is: {reward}")


@app.command()
def lower():
    CONTEXT = {
        "STAND": [0, torch.tensor([0.75, 1.2, 0.25, 0.5, 0.75])],
        "WALK": [1, torch.tensor([0.75, 0.8, 0.25, 0.5, 0.75])],
        "TROT": [2, torch.tensor([0.5, 0.5, 0.5, 0.5, 0])],
        "CRAWL": [3, torch.tensor([0.75, 1.2, 0.25, 0.5, 0.75])],
        "FLYING_TROT": [4, torch.tensor([0.4, 0.6, 0.5, 0.5, 0])],
        "PACE": [5, torch.tensor([0.5, 0.6, 0.5, 0.5, 0])],
        "PRONK": [6, torch.tensor([0.9, 0.6, 0.0, 0.0, 0.0])],
        "BOUND": [7, torch.tensor([0.5, 0.4, 0.5, 0.5, 0])],
        "HOP_TROT": [8, torch.tensor([0.35, 1, 0.5, 0.5, 0])],
    }

    torch.manual_seed(50)
    np.random.seed(50)

    data = Data()
    data.load("/var/home/dw/development/gosafeopt/optim/")

    with open("/var/home/dw/development/gosafeopt/optim/config.json") as f:
        config = json.load(f)["optimization"]

    model = create_model(config, data)

    context = CONTEXT["FLYING_TROT"][1]
    aquisition = SafeLCB(model, config, context, data=data)
    optimizer = get_optimizer(aquisition, config, context)
    k, acf_val = optimizer.getNextPoint()

    print(k)


@app.command()
def optimize_gp():
    with open("config.json") as f:
        config = json.load(f)["optimization"]

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    data = Data()
    data.load(config["save_data"])

    state_dict = torch.load(config["save_model"])

    gp = create_model(config, data, state_dict)

    mll = SumMarginalLogLikelihood(gp.likelihood, gp)

    for m in gp.models:
        Logger.info(f"Old Lenghtscale: {m.covar_module.base_kernel.lengthscale}")

    fit_gpytorch_mll(mll)

    for m in gp.models:
        Logger.info(f"New Lenghtscale: {m.covar_module.base_kernel.lengthscale}")


if __name__ == "__main__":
    app()
