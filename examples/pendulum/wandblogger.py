from gosafeopt.tools.data_logger import DataLogger, WandbLogger
from pendulum.plot import PlotPendulum, PlotPendulumPlotly
import wandb
import os.path
import matplotlib.pyplot as plt
import numpy as np
import torch


class PendulumWandbLogger(WandbLogger):
    def __init__(self, config, config_pendulum, X_star, context=None):
        super().__init__(config, "gosafeopt4", context)
        self.datalogger = DataLogger(config, context)
        self.plotter = PlotPendulumPlotly(X_star, config, config_pendulum)
        self.pltPlotter = PlotPendulum(X_star, config, config_pendulum)
        self.config_pendulum = config_pendulum
        self.config = config
        self.rewardMax = -1e10

    def log(
        self,
        model,
        trajectory,
        x,
        y,
        data,
        rewardMax,
        loss_aq,
        backup_triggered,
        episode,
        info,
    ):
        self.datalogger.log(
            model,
            trajectory,
            x,
            y,
            data,
            rewardMax,
            loss_aq,
            backup_triggered,
            episode,
            info,
        )
        content = {}

        newMin = rewardMax[0] > self.rewardMax
        if newMin:
            self.rewardMax = rewardMax[0]

        if self.config["log_video"]:
            self.plotter.plotIdx(self.datalogger, len(self.datalogger.x_buffer) - 1)
            self.pltPlotter.plotIdx(self.datalogger, episode)
        if self.config["log_plots"] and episode == self.config["n_opt_samples"] - 1:
            # self.plotter.plotIdx(self.datalogger, len(self.datalogger.x_buffer)-1)
            self.pltPlotter.plotIdx(self.datalogger, episode)
            # content["State"] = self.plotter.figState
            # content["Reward"]= self.plotter.figReward
            # content["Contour"] = self.plotter.figContour
            content["Chart"] = wandb.Image(self.pltPlotter.getLastPlot())

        if (
            backup_triggered
            or newMin
            or not episode % self.config["save_interval"]
            or torch.any(y[1:] < 0)
        ):
            if self.config["log_video"]:
                fname = (
                    "{}/video/".format(wandb.run.dir)
                    + f"/rl-video-episode-{episode}.mp4"
                )
                if os.path.isfile(fname):
                    content["video"] = wandb.Video(
                        fname, format="gif"
                    )  # "Surface": self.plotter.figSurface

        super().log(
            model,
            trajectory,
            x,
            y,
            data,
            rewardMax,
            loss_aq,
            backup_triggered,
            episode,
            content,
        )

    def finish(self):
        directory = "{}/res/".format(wandb.run.dir)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            wandb.save("{}/res/{}".format(wandb.run.dir, filename))
        super().finish()

    def save(self, path):
        self.datalogger.save(path)
