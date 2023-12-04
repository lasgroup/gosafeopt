from typing import Optional, Tuple

import numpy as np
import torch
from gymnasium.utils.save_video import save_video
from numpy.typing import NDArray
from torch import Tensor

from gosafeopt.experiments.backup import BackupStrategy
from gosafeopt.experiments.environment import Environment
from gosafeopt.tools.data import Data
from gosafeopt.tools.logger import Logger


class Experiment:
    """Base class for experiments."""

    def __init__(self, config: dict, env: Environment, data: Data, backup: Optional[BackupStrategy] = None):
        self.c = config
        self.env = env
        self.backup = backup
        self.data = data
        self.render_list = []

    def rollout(self, param: Tensor, episode: int = 0) -> Tuple[Tensor, Tensor, bool, dict]:
        np_param = param.numpy()
        initial_state, info = self.reset()
        trajectory = [initial_state]
        rewards: NDArray = np.zeros(self.c["dim_obs"])
        done = False
        backup_triggered = False

        if self.backup is not None:
            self.backup.reset()

        self.env.before_experiment(np_param)

        i = 0
        while not done:
            observation, reward, done, truncated, info = self.env.step(np_param[0 : self.c["dim_params"]])

            rewards += reward
            trajectory.append(observation)

            if not backup_triggered:
                backup_triggered = self.process_backup_strategy(observation, reward, i)  # type: ignore

            self.process_render()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            i += 1

        self.env.after_experiment()

        if self.backup is not None:
            self.backup.after_rollout(np_param, rewards, backup_triggered)
        self.process_video(episode)

        rewards = rewards / len(trajectory)
        return torch.from_numpy(rewards), torch.tensor(trajectory), backup_triggered, info

    def reset(self):
        initial_state, info = self.env.reset()
        self.render_list = []
        if self.backup is not None:
            self.backup.reset()
        return initial_state, info

    def finish(self):
        self.env.close()

    def process_backup_strategy(self, observation: NDArray, reward: NDArray, i: int):
        if self.backup is not None and not self.backup.is_safe(observation, reward):
            param = self.backup.get_backup_policy(observation)
            self.env.backup(param)
            self.data.append_failed(torch.tensor(param), torch.tensor(observation))
            Logger.warn(f"Backup policy triggered at step {i} with policy {param}")
            return True

        return False

    def process_render(self):
        if self.c["log_video"] and self.env.render_mode == "rgb_array_list":
            self.render_list.append(self.env.render())

        if self.c["log_video"] and self.env.render_mode == "human":
            self.env.render()

    def process_video(self, episode: int):
        if self.c["log_video"] and self.env.render_mode == "rgb_array_list":

            def epsodeTrigger(x):  # noqa: ANN202, ANN001, N802, ARG001
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
                Logger.warn(f"couldnt log video {e}")
