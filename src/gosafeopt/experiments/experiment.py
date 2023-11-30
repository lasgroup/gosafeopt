from logging import warn
import torch
from gosafeopt.aquisitions.go_safe_opt import GoSafeOptState
from gosafeopt.optim.base_optimizer import SafeSet
from gosafeopt.tools.logger import Logger
import numpy as np
from gymnasium.utils.save_video import save_video
import gosafeopt
from gosafeopt.aquisitions.go_safe_opt import OptimizationStep
from gosafeopt.experiments.environment import Environment
from gosafeopt.experiments.backup import BackupStrategy, GoSafeOptBackup
from typing import Optional, Tuple
from torch import Tensor
from gosafeopt.tools.data import Data


class Experiment:
    """
    Base class for experiments
    """

    def __init__(self, config: dict, env: Environment, data: Data, backup: Optional[BackupStrategy] = None):
        self.c = config
        self.env = env
        self.backup = backup
        self.data = data
        self.render_list = []

    def rollout(self, param: Tensor, episode=0) -> Tuple[Tensor, Tensor, bool, dict]:
        initial_state, _ = self.env.reset()
        trajectory = [initial_state]
        rewards = np.zeros(self.c["dim_obs"])
        done = False
        backup_triggered = False
        info = None

        if self.backup is not None:
            self.backup.reset()

        self.env.before_experiment(param)

        i = 0
        while not done:
            observation, reward, done, truncated, info = self.env.step(param[0 : self.c["dim_params"]])

            rewards += reward
            trajectory.append(observation)

            if not backup_triggered:
                backup_triggered = self.process_backup_strategy(observation, reward, i)

            self.process_render()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            i += 1

        self.env.after_experiment()
        if self.backup is not None:
            self.backup.after_rollout(param, rewards)
        self.process_video(episode)

        rewards = rewards / len(trajectory)
        return torch.from_numpy(rewards), torch.tensor(trajectory), backup_triggered, info

    def reset(self):
        self.env.reset()
        self.render_list = []
        if self.backup is not None:
            self.backup.reset()

    def finish(self):
        self.env.close()

    def process_backup_strategy(self, observation, reward, i):
        if self.backup is not None:
            if not self.backup.is_safe(observation):
                param = self.backup.get_backup_policy(observation)
                self.env.backup(param)
                self.data.append_failed(param, torch.tensor(observation))
                Logger.warn(f"Backup policy triggered at step {i} with policy {param}")
                return True

        return False

    def process_render(self):
        if self.c["log_video"] and self.env.render_mode == "rgb_array_list":
            self.render_list.append(self.env.render())

        if self.c["log_video"] and self.env.render_mode == "human":
            self.env.render()

    def process_video(self, episode):
        if self.c["log_video"] and self.env.render_mode == "rgb_array_list":

            def epsodeTrigger(x):
                return True

            try:
                save_video(
                    self.render_list,
                    "./",
                    fps=self.env.metadata["render_fps"],
                    episode_index=episode,
                    episode_trigger=epsodeTrigger,
                )
            except Exception as e:
                warn(f"couldnt log video: {e}")
