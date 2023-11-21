from logging import warn
import torch
from gosafeopt.aquisitions.go_safe_opt import GoSafeOpt, GoSafeOptState
from gosafeopt.optim.base_optimizer import OptimizerState
from gosafeopt.tools.logger import Logger
import numpy as np
from gymnasium.utils.save_video import save_video
import wandb
import gc
import gosafeopt
import sys
import os
import fcntl
import time


class Experiment:
    """
    Base class for experiments
    """

    def __init__(self, config, env, backup=None):
        self.c = config
        self.env = env
        self.backup = backup
        self.first_setup_done = False


    def rollout(self, k, episode=0, ignore_backup=False):

        rewards = torch.zeros(self.c["dim_obs"])

        M = self.c["n_average"]
        N = self.c["n_rollout"]
        backup_triggered = False

        info = None

        renderList = []

        for j in range(M):
            initial_state, _ = self.env.reset()

            if self.backup is not None:
                self.backup.reset()

            if self.c["use_setup_experiment"] or (not self.first_setup_done and not self.c["skip_initial_setup"]): 
                input("Press Enter after setting up envionment")
                self.first_setup_done = True

            trajectory = torch.zeros(N, self.c["dim_state"])
            trajectory[0, :] = torch.from_numpy(initial_state)

            warning = True

            self.env.startExperiment(k)
            for i in range(1, N):
                reward, observation, stepinfo = self.step(k)

                if info is None and stepinfo is not None:
                    info = stepinfo
                elif stepinfo is not None and info is not None and info is not False:
                    for key, value in stepinfo.items():
                        info[key] += value

                trajectory[i, :] = torch.from_numpy(observation)
                if rewards.shape[0] == reward.shape[0]:
                    rewards += torch.from_numpy(reward)
                elif warning:
                    Logger.warn("Ignoring rewards")
                    warning = False

                if self.backup is not None and not ignore_backup and not backup_triggered:
                    if not self.backup.isSafe(trajectory[i]):
                        self.backup.adddFail(k, trajectory[i])
                        k = self.backup.getBackupPolicy(trajectory[i])
                        self.env.backup(k)

                        if not backup_triggered:
                            Logger.warn(f"Backup policy triggered at step {i} with policy {k}")
                            backup_triggered = True

                if self.c["log_video"] and self.env.render_mode == "rgb_array_list":
                    renderList.append(self.env.render())

                if self.c["log_video"] and self.env.render_mode == "human":
                    self.env.render()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self.env.afterExperiment()

            if self.c["log_video"] and self.env.render_mode == "rgb_array_list":
                def epsodeTrigger(x): return True
                try:
                    save_video(renderList, "./",
                               fps=self.env.metadata["render_fps"], episode_index=episode, episode_trigger=epsodeTrigger)
                except Exception as e:
                    warn(f"couldnt log video: {e}")

        if not backup_triggered and self.backup is not None and not ignore_backup and not self.backup.goState.skipBackupAtRollout() and not torch.any(rewards[1:] < 0):
            k.to(gosafeopt.device)
            GoSafeOptState(self.c).goToS1()
            OptimizerState(self.c).addSafeSet(k.reshape(1, -1))
            OptimizerState(self.c).changeToLastSafeSet()

        rewards = rewards/(M*N)
        return rewards, trajectory, backup_triggered, info

    def step(self, k):
        """
        returns state x after step
        """
        observation, reward, done, truncated, info = self.env.step(k[0:self.c["dim_params"]])
        if not isinstance(reward, np.ndarray):
            reward = np.array([reward])
        return reward, observation, info

    def reset(self):
        self.env.reset()
        if self.backup is not None:
            self.backup.reset()

    def finish(self):
        self.env.close()
