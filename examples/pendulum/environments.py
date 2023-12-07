import copy
from typing import Optional

import numpy as np
from gosafeopt.experiments.environment import Environment
from gosafeopt.experiments.experiment import Experiment
from gymnasium.envs.classic_control.pendulum import PendulumEnv
from pendulum.dynamics import U_ideal, U_learned
import torch


class PendulumGymEnv(Environment):
    def __init__(self, config, pendulumConfig, render_mode=None):
        super().__init__(render_mode)

        self.c = config
        self.i = 0

        ideal_env = PendulumGymEnvWithDynamics(pendulumConfig, U_ideal, render_mode=None)
        ideal_experiment = Experiment(config, ideal_env)

        _, self.idealTrajectory, _, _ = ideal_experiment.rollout(torch.zeros(2), 0)

        self.env = PendulumGymEnvWithDynamics(pendulumConfig, U_learned, render_mode=render_mode)

        self.metadata = self.env.metadata

    def _get_obs(self):
        return self.env._get_obs()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        observation, _ = self.env.reset()
        super().reset()
        self.i = 0
        return observation, {}

    def backup(self, params):
        self.env.backup(params)

    def step(self, k):
        observation, reward, terminated, truncated, info = self.env.step(k)
        self.i += 1

        norm = np.linalg.norm(self.idealTrajectory[self.i, :] - observation)
        loss = -0.5 * norm * norm
        c2 = 2.2 - np.abs(observation[1])

        rewards = np.array([loss, c2])
        return observation, rewards, terminated, truncated, info

    def render(self):
        return self.env.render()


class PendulumGymEnvWithDynamics(PendulumEnv, Environment):
    def __init__(self, config, U, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.config = config

        self.max_speed = 100
        self.max_torque = 100
        self.dt = config["dt"]
        self.g = config["g"]
        self.m = config["m"]
        self.l = config["L"]
        self.n_rollout = config["n_rollout"]
        self.n = 0
        self.U = U
        self.backup_params = None

    def _get_obs(self):
        return self.state

    def backup(self, params):
        self.backup_params = params

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset()
        self.state = np.array([self.config["x0"], self.config["x0_dot"]])
        self.n = 0
        self.backup_params = None
        return self._get_obs(), {}

    def step(self, k):
        config = copy.deepcopy(self.config)
        if self.backup_params is None:
            config["kp_bo"] = k[0]
            config["kd_bo"] = k[1]
        else:
            config["kp_bo"] = self.backup_params[0]
            config["kd_bo"] = self.backup_params[1]

        state = self._get_obs()
        action = np.array([self.U(state, config)])
        obs, cost, done, truncated, info = super().step(action)

        self.n += 1

        done = self.n > self.n_rollout
        return obs, cost, done, truncated, info
