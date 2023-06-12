import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from rich.progress import track
from matplotlib.colors import TwoSlopeNorm
import gpytorch
import copy
from plotly.subplots import make_subplots
import wandb

from torch.nn.modules import loss
import plotly.express as px
import plotly.graph_objects as go
from enum import Enum

import gosafeopt

class Colors:
    BLUE = "#636EFA"
    RED = "#EF553B"
    GREEN = "#00CC96"
    LIGHT_GREEN = "#B6E880"
    ORANGE = "#FFA15A"
    YELLOW = "#FECB52"


class PlotPendulumPlotly:
    def __init__(self, X_star, config, config_pendulum):
        # self.setupAxis()
        self.X_bo_buffer = []
        self.y_min_buffer = []
        self.X_star = X_star
        self.config = config
        self.config_pendulum = config_pendulum

        self.maxIdx = 0
        self.lossMin = -1e10

        self.setUpFigs()

    def setUpFigs(self):
        x = np.arange(len(self.X_star[:, 0]))

        self.figState = make_subplots(rows=2, cols=1, subplot_titles=("Q, Qdot"), shared_xaxes=True)
        self.figState.update_yaxes(title_text="Q", row=1, col=1)
        self.figState.update_yaxes(title_text="Qdot", row=2, col=1)
        self.figState.update_xaxes(title_text="Time", row=2, col=1)
        self.figState.add_trace(go.Scatter(x=x, y=self.X_star[:, 0], line=dict(color=Colors.GREEN), name="Ideal trajectory"), row=1, col=1)
        self.figState.add_trace(go.Scatter(x=x, y=self.X_star[:, 1], line=dict(color=Colors.GREEN), name="Ideal trajectory"), row=2, col=1)
        self.figState['data'][1]['showlegend'] = False

        self.figReward = make_subplots(rows=1, cols=1, subplot_titles=("Reward"))
        self.figReward.update_yaxes(title_text="Reward", row=2, col=1)
        self.figReward.update_xaxes(title_text="Time", row=2, col=1)

        self.figContour = make_subplots(rows=1, cols=1, subplot_titles=("Contour"))

        self.figSurface = make_subplots(rows=1, cols=1, subplot_titles=("Surface"))

    def plotIdx(self, logger, i):
        model, trajectories, ks, rewards, rewardMaxBuffer, backupTriggerBuffer = logger.getDataFromEpoch(i)

        if len(backupTriggerBuffer) > 0:
            backupTriggerBuffer = np.array([backupTriggerBuffer], dtype=bool)
        else:
            backupTriggerBuffer = np.zeros(len(trajectories), dtype=bool)
        backupTriggerBuffer = backupTriggerBuffer.reshape(-1)

        # self.plotState(maxIdx, i, trajectories[i][:, 0], backupTriggerBuffer[i], newMin, 1)
        # self.plotState(maxIdx, i, trajectories[i][:, 1], backupTriggerBuffer[i], newMin, 2)

        self.plotReward(rewards[:i, 0], rewardMaxBuffer[:i])

        inp = self.createGrid(logger.context).to(gosafeopt.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            with gpytorch.settings.fast_pred_samples():
                out = model.posterior(inp)

        mean = out.mean.detach().to("cpu")
        var = out.variance.detach().to("cpu").numpy()
        inp = inp.to("cpu")

        rewardsBack = copy.copy(rewards)
        if np.any(backupTriggerBuffer):
            rewards[:i+1][backupTriggerBuffer] = -1e10
        maxIdx = np.argmax(rewards[: i + 1, 0])
        x_min = ks[maxIdx, :]
        y_min = rewards[maxIdx, :]
        rewards = rewardsBack

        self.plotContour(mean, var, ks, i, backupTriggerBuffer, x_min)
        # self.plotSurface(mean, var, ks, i, backupTriggerBuffer, x_min)

    def plotSurface(self, mean, var, ks, i, backupTriggerBuffer, x_min):
        self.figSurface["data"] = []

        colors = var[:, 0]/np.amax(var[:, 0])

        grid_x = torch.linspace(
            self.config["domain_start"][0],
            self.config["domain_end"][0],
            self.config["plotting_n_samples"],
        )
        grid_y = torch.linspace(
            self.config["domain_start"][1],
            self.config["domain_end"][1],
            self.config["plotting_n_samples"],
        )

        self.figSurface.add_trace(go.Surface(
            x=grid_x,
            y=grid_y,
            z=mean,
            surfacecolor=cm.jet(colors),
            colorscale="RdBu",
            showscale=False
        ))

    def plotContour(self, mean, var, ks, i, backupTriggerBuffer, x_min):

        self.figContour["data"] = []

        colors = (torch.min(mean[:, 1:]-self.config["scale_beta"]*np.sqrt(self.config["beta"]*var[:, 1:]), dim=1)
                  [0]).reshape(self.config["plotting_n_samples"], self.config["plotting_n_samples"])

        grid_x = torch.linspace(
            self.config["domain_start"][0],
            self.config["domain_end"][0],
            self.config["plotting_n_samples"],
        )
        grid_y = torch.linspace(
            self.config["domain_start"][1],
            self.config["domain_end"][1],
            self.config["plotting_n_samples"],
        )

        self.figContour.add_trace(go.Contour(
            x=grid_x,
            y=grid_y,
            z=colors,
            colorscale="RdBu",
            colorbar=None,
            showscale=False,
            ncontours=20
        ))

        successfullk = ks[:i+1][~backupTriggerBuffer[:i+1]]
        self.figContour.add_trace(go.Scatter(
            x=successfullk[:, 0],
            y=successfullk[:, 1],
            marker=dict(color=Colors.BLUE, symbol="circle"),
            mode="markers",
            name="Successfull experiments"
        ))

        failedk = ks[:i+1][backupTriggerBuffer[:i+1]]
        self.figContour.add_trace(go.Scatter(
            x=failedk[:, 0],
            y=failedk[:, 1],
            marker=dict(color=Colors.RED, symbol="circle"),
            mode="markers",
            name="Backup triggered"
        ))

        self.figContour.add_trace(go.Scatter(
            x=[ks[i, 0]],
            y=[ks[i, 1]],
            marker=dict(color=Colors.BLUE, size=20, symbol="x"),
            mode="markers",
            name="Next evaluation point"
        ))

        self.figContour.add_trace(go.Scatter(
            x=[self.config_pendulum["kp"]],
            y=[self.config_pendulum["kd"]],
            marker=dict(color="black", size=20, symbol="x"),
            mode="markers",
            name="Solution"
        ))

        self.figContour.add_trace(go.Scatter(
            x=[x_min[0]],
            y=[x_min[1]],
            marker=dict(color="green", size=20, symbol="x"),
            mode="markers",
            name="Best Found solution"
        ))

        self.figContour.update_coloraxes(showscale=False)

    def plotReward(self, reward, rewardMax):
        self.figReward['data'] = []
        x = np.arange(len(reward))
        self.figReward.add_trace(go.Scatter(x=x, y=reward, line=dict(color=Colors.YELLOW), name="Reward"), row=1, col=1)
        self.figReward.add_trace(go.Scatter(x=x, y=rewardMax, line=dict(color=Colors.GREEN), name="Best Reward"), row=1, col=1)

    def plotState(self, maxIdx, i, data, backup_triggered, newMin, row):
        x = np.arange(len(data))
        if i == 0:
            self.figState.add_trace(go.Scatter(x=x, y=data, line=dict(color=Colors.YELLOW), name="Initial Trajectory"), row=row, col=1)
        else:
            n = len(self.figState['data'])
            if i >= 1 and self.maxIdx != n-1 and self.figState['data'][n-1]['line'] != dict(color=Colors.RED):
                self.figState['data'][n-1]['line'] = dict(color=Colors.BLUE)
                self.figState['data'][n-1]['showlegend'] = False

            if backup_triggered:
                self.figState.add_trace(go.Scatter(x=x, y=data, line=dict(color=Colors.RED), opacity=0.2,
                                                   legendgroup="failed", legendgrouptitle_text="Failed trajectories"), row=row, col=1)
            else:
                self.figState.add_trace(go.Scatter(x=x, y=data, line=dict(color=Colors.YELLOW), name="Current Trajectory"), row=row, col=1)

            if newMin and i != 0:
                if self.maxIdx > 0:
                    self.figState['data'][self.maxIdx]['showlegend'] = False
                    self.figState['data'][self.maxIdx]['line'] = dict(color=Colors.BLUE)
                    self.figState['data'][self.maxIdx]['opacity'] = 0.2

                self.figState['data'][maxIdx]["name"] = "Best found"
                self.figState['data'][self.maxIdx]['opacity'] = 1.0
                self.figState['data'][maxIdx]['showlegend'] = True
                self.figState['data'][maxIdx]['line'] = dict(color=Colors.ORANGE)

                self.maxIdx = maxIdx

            if row > 1:
                self.figState['data'][n]["showlegend"] = False

    def createGrid(self, context=None):
        grid_x = torch.linspace(
            self.config["domain_start"][0],
            self.config["domain_end"][0],
            self.config["plotting_n_samples"],
        )
        grid_y = torch.linspace(
            self.config["domain_start"][1],
            self.config["domain_end"][1],
            self.config["plotting_n_samples"],
        )
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing="xy")

        inp = torch.stack((grid_x, grid_y), dim=2).float()
        inp = inp.reshape(-1, 2)

        if context is not None:
            inp = torch.hstack([inp, context.repeat(len(inp), 1)])
        return inp


class PlotPendulum:
    def __init__(self, X_star, config, config_pendulum):
        self.setupAxis()
        self.X_bo_buffer = []
        self.y_min_buffer = []
        self.X_star = X_star
        self.config = config
        self.config_pendulum = config_pendulum

    def setupAxis(self):
        # plt.ion()
        self.fig = plt.figure(figsize=(19.20, 10.80))
        sns.set_theme()
        grid = self.fig.add_gridspec(3, 2)
        self.ax = self.fig.add_subplot(grid[:2, 0], projection="3d")
        self.ax5 = self.fig.add_subplot(grid[2, 0])
        self.ax2 = self.fig.add_subplot(grid[0, 1])
        self.ax3 = self.fig.add_subplot(grid[1, 1])
        self.ax4 = self.fig.add_subplot(grid[2, 1])

    def clearSurface(self):
        self.ax.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.ax5.clear()
        self.ax.set_xlim(self.config["domain_start"][0], self.config["domain_end"][0])
        self.ax.set_ylim(self.config["domain_start"][1], self.config["domain_end"][1])
        self.ax.set_ylabel("kd")
        self.ax.set_xlabel("kp")
        self.ax.set_zlabel("f(x)")
        self.ax.zaxis.set_label_coords(.5, -10)
        self.ax2.set_ylabel("theta")
        self.ax2.set_xlabel("t")
        self.ax3.set_ylabel("theta_dot")
        self.ax3.set_xlabel("t")
        self.ax4.set_ylabel("error")
        self.ax4.set_xlabel("t")
        self.ax3.set_ylim(-4, 4)
        self.ax2.set_ylim(-3, 3)
        self.ax3.set_ylim(-9, 9)
        self.ax5.set_ylabel("kd")
        self.ax5.set_xlabel("kp")
        self.ax4.set_xlabel("t")
        self.ax5.set_xlim(self.config["domain_start"][0], self.config["domain_end"][0])
        self.ax5.set_ylim(self.config["domain_start"][1], self.config["domain_end"][1])

    def getLastPlot(self):
        return self.lastPlot

    def createGrid(self, context=None):
        grid_x = torch.linspace(
            self.config["domain_start"][0],
            self.config["domain_end"][0],
            self.config["plotting_n_samples"],
        )
        grid_y = torch.linspace(
            self.config["domain_start"][1],
            self.config["domain_end"][1],
            self.config["plotting_n_samples"],
        )
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing="xy")

        inp = torch.stack((grid_x, grid_y), dim=2).float()
        inp = inp.reshape(-1, 2)

        if context is not None:
            inp = torch.hstack([inp, context.repeat(len(inp), 1)])
        return inp

    def plotSurface(self, i, inp, mean, var, x, y, x_min, y_min, model, backupTriggerBuffer):
        _inp = inp.reshape(self.config["plotting_n_samples"],
                           self.config["plotting_n_samples"], -1)

        colors = (torch.min(mean[:, 1:]-self.config["scale_beta"]*np.sqrt(self.config["beta"]*var[:, 1:]), dim=1)
                  [0]).reshape(self.config["plotting_n_samples"], self.config["plotting_n_samples"])

        self.ax5.contourf(
            _inp[:, :, 0],
            _inp[:, :, 1],
            colors,
            norm=TwoSlopeNorm(0),
            cmap=cm.RdBu,
            levels=50
        )
        successfullX = x[:i+1][~backupTriggerBuffer[:i+1]]
        self.ax5.scatter(
            successfullX[:, 0],
            successfullX[:, 1],
            color="blue",
            marker="o",
            label="sucessfull"
        )
        failedX = x[:i+1][backupTriggerBuffer[:i+1]]

        self.ax5.scatter(
            failedX[:, 0],
            failedX[:, 1],
            color="red",
            marker="o",
            label="backup triggered"
        )
        self.ax5.plot(
            x[i, 0],
            x[i, 1],
            color="blue",
            marker="X",
            markersize=40,
            label="next"
        )
        self.ax5.plot(
            self.config_pendulum["kp"],
            self.config_pendulum["kd"],
            color="black",
            marker="X",
            markersize=20,
            label="Ideal solution"
        )
        self.ax5.plot(
            x_min[0],
            x_min[1],
            color="green",
            marker="X",
            markersize=20,
            label="Best solution"
        )
        colors = var[:, 0].reshape(self.config["plotting_n_samples"],
                                   self.config["plotting_n_samples"])/np.amax(var[:, 0])
        self.ax.plot_surface(
            _inp[:, :, 0],
            _inp[:, :, 1],
            mean[:, 0].reshape(
                self.config["plotting_n_samples"], self.config["plotting_n_samples"]),
            vmax=10,
            alpha=0.3,
            facecolors=cm.jet(colors),
        )

        self.ax.scatter(
            successfullX[:, 0],
            successfullX[:, 1],
            y[:i+1][~backupTriggerBuffer[:i+1]],
            s=20,
            color="blue",
            marker="o",
            label="sucessful"
        )

        self.ax.plot(
            x[i, 0],
            x[i, 1],
            y[i],
            color="blue",
            marker="X",
            markersize=40,
            label="next"
        )

        self.ax.plot(
            self.config_pendulum["kp"],
            self.config_pendulum["kd"],
            0,
            color="black",
            marker="X",
            markersize=20,
            label="solution"
        )

        self.ax.plot(
            x_min[0],
            x_min[1],
            y_min[0],
            color="green",
            marker="X",
            markersize=20,
            label="best found"
        )

        # self.ax.legend()
        # self.ax5.legend()

    def plotIdx(self, logger, i):
        self.clearSurface()

        model, trajectories, ks, rewards, rewardMaxBuffer, backupTriggerBuffer = logger.getDataFromEpoch(i)
        if len(backupTriggerBuffer) > 0:
            backupTriggerBuffer = np.array([backupTriggerBuffer], dtype=bool)
        else:
            backupTriggerBuffer = np.zeros(len(trajectories), dtype=bool)
        backupTriggerBuffer = backupTriggerBuffer.reshape(-1)

        inp = self.createGrid(logger.context).to(gosafeopt.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            with gpytorch.settings.fast_pred_samples():
                out = model.posterior(inp)

        mean = out.mean.detach().to("cpu")
        var = out.variance.detach().to("cpu").numpy()
        inp = inp.to("cpu")

        rewardsBack = copy.copy(rewards)
        if np.any(backupTriggerBuffer):
            rewards[:i+1, 0][backupTriggerBuffer] = -1e10
        maxIdx = np.argmax(rewards[: i + 1, 0])
        xMax = ks[maxIdx, :]
        yMax = rewards[maxIdx, :]
        rewards = rewardsBack

        self.plotSurface(i, inp, mean, var, ks, rewards[:, 0], xMax, yMax, model, backupTriggerBuffer)

        for i in range(len(trajectories)):
            if backupTriggerBuffer[i]:
                self.ax2.plot(trajectories[i][:, 0], color="red", alpha=0.1)
                self.ax3.plot(trajectories[i][:, 1], color="red", alpha=0.1)
            else:
                self.ax2.plot(trajectories[i][:, 0], color="blue", alpha=0.1)
                self.ax3.plot(trajectories[i][:, 1], color="blue", alpha=0.1)

        self.ax2.plot(logger.trajectory_buffer[0][:, 0], color="orange", label="initial")
        self.ax2.plot(logger.trajectory_buffer[maxIdx][:, 0], color="green", label="bestfound")
        self.ax2.plot(self.X_star[:, 0], color="black", label="ideal")
        self.ax2.legend()
        self.ax3.plot(logger.trajectory_buffer[0][:, 1], color="orange", label="initial")
        self.ax3.plot(logger.trajectory_buffer[maxIdx][:, 1], color="green", label="bestfound")
        self.ax3.plot(self.X_star[:, 1], color="black", label="ideal")
        self.ax3.legend()

        # self.ax4.set_title(r"k_star =  [{} {}] and k_hat = [{} {}]], error: {}".format(
        #     self.config_pendulum["kp"]*self.config_pendulum["L"], self.config_pendulum["kd"] *
        #     self.config_pendulum["L"], xMax[0], xMax[1], yMax)
        # )
        self.ax4.plot(-np.array(rewardMaxBuffer[:i+1]), label="Best found", color="green")
        self.ax4.plot(-rewards[:i+1, 0], label="Loss", color="orange")
        self.ax4.set_xticks(range(0,i))
        self.ax4.legend()

        try:
            self.lastPlot = "{}/plot/{}".format(wandb.run.dir, "/{0:0>3}.png".format(i))
            plt.savefig(self.lastPlot, dpi=196)
        except Exception:
            pass

    def plot(self, logger):
        N = len(logger.trajectory_buffer)
        plt.pause(3)

        for i in track(range(N), description="Generating Plot..."):
            self.plotIdx(logger, i)
            plt.pause(0.001)


class PlotPendulumContour:
    def __init__(self, X_star, config, config_pendulum):
        self.X_bo_buffer = []
        self.y_min_buffer = []
        self.X_star = X_star
        self.config = config
        self.config_pendulum = config_pendulum
        self.setupAxis()

    def setupAxis(self):
        # plt.ion()
        self.fig, self.ax = plt.subplots()
        sns.set_theme()
        self.ax.set_ylabel("kd")
        self.ax.set_xlabel("kp")
        self.ax.set_xlim(self.config["domain_start"][0], self.config["domain_end"][0])
        self.ax.set_ylim(self.config["domain_start"][1], self.config["domain_end"][1])

    def createGrid(self, context=None):
        grid_x = torch.linspace(
            self.config["domain_start"][0],
            self.config["domain_end"][0],
            self.config["plotting_n_samples"],
        )
        grid_y = torch.linspace(
            self.config["domain_start"][1],
            self.config["domain_end"][1],
            self.config["plotting_n_samples"],
        )
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing="xy")

        inp = torch.stack((grid_x, grid_y), dim=2).float()
        inp = inp.reshape(-1, 2)

        if context is not None:
            inp = torch.hstack([inp, context.repeat(len(inp), 1)])
        return inp

    def plotSurface(self, i, inp, mean, var, x, y, x_min, y_min, model, backupTriggerBuffer):
        _inp = inp.reshape(self.config["plotting_n_samples"],
                           self.config["plotting_n_samples"], -1)

        colors = (torch.min(mean[:, 1:]-self.config["scale_beta"]*np.sqrt(self.config["beta"]*var[:, 1:]), dim=1)
                  [0]).reshape(self.config["plotting_n_samples"], self.config["plotting_n_samples"])

        self.ax.contourf(
            _inp[:, :, 0],
            _inp[:, :, 1],
            colors,
            norm=TwoSlopeNorm(0),
            cmap=cm.RdBu,
            levels=50
        )
        successfullX = x[:i+1][~backupTriggerBuffer[:i+1]]
        self.ax.scatter(
            successfullX[:, 0],
            successfullX[:, 1],
            color="blue",
            marker="o",
            label="sucessfull"
        )
        failedX = x[:i+1][backupTriggerBuffer[:i+1]]

        self.ax.scatter(
            failedX[:, 0],
            failedX[:, 1],
            color="red",
            marker="o",
            label="backup triggered"
        )
        self.ax.plot(
            x[i, 0],
            x[i, 1],
            color="blue",
            marker="X",
            markersize=40,
            label="next"
        )
        self.ax.plot(
            self.config_pendulum["kp"],
            self.config_pendulum["kd"],
            color="black",
            marker="X",
            markersize=20,
            label="Ideal solution"
        )
        self.ax.plot(
            x_min[0],
            x_min[1],
            color="green",
            marker="X",
            markersize=20,
            label="Best solution"
        )

    def plotIdx(self, logger, i):

        model, trajectories, ks, rewards, rewardMaxBuffer, backupTriggerBuffer = logger.getDataFromEpoch(i)
        if len(backupTriggerBuffer) > 0:
            backupTriggerBuffer = np.array([backupTriggerBuffer], dtype=bool)
        else:
            backupTriggerBuffer = np.zeros(len(trajectories), dtype=bool)
        backupTriggerBuffer = backupTriggerBuffer.reshape(-1)

        inp = self.createGrid(logger.context).to(gosafeopt.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            with gpytorch.settings.fast_pred_samples():
                out = model.posterior(inp)

        mean = out.mean.detach().to("cpu")
        var = out.variance.detach().to("cpu").numpy()
        inp = inp.to("cpu")

        rewardsBack = copy.copy(rewards)
        if np.any(backupTriggerBuffer):
            rewards[:i+1, 0][backupTriggerBuffer] = -1e10
        maxIdx = np.argmax(rewards[: i + 1, 0])
        xMax = ks[maxIdx, :]
        yMax = rewards[maxIdx, :]
        rewards = rewardsBack

        self.plotSurface(i, inp, mean, var, ks, rewards[:, 0], xMax, yMax, model, backupTriggerBuffer)
