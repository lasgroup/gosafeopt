from gymnasium import Env
from gymnasium.core import RenderFrame
from gymnasium.envs.classic_control.pendulum import PendulumEnv
from gosafeopt.experiments.environment import Environment
from gosafeopt.experiments.experiment import Experiment
from pendulum.dynamics import U_learned, U_ideal
from typing import Optional
import copy
import numpy as np
import moviepy


class PendulumGymEnv(Environment):
    def __init__(self, config, pendulumConfig, render_mode=None):
        super().__init__(render_mode)

        self.c = config
        self.i = 0

        idealEnv = PendulumGymEnvWithDynamics(pendulumConfig, U_ideal, render_mode=None)
        idealExperiment = Experiment(config, idealEnv)

        _, self.idealTrajectory, _, _ = idealExperiment.rollout(np.zeros(2), 0, ignore_backup=True)

        self.env = PendulumGymEnvWithDynamics(pendulumConfig, U_learned, render_mode=render_mode)

        self.metadata = self.env.metadata

    def getIdealTrajecory(self):
        return self.idealTrajectory

    def _get_obs(self):
        return self.env._get_obs()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        observation, _ = self.env.reset()
        super().reset()
        self.i = 0
        return observation, {}

    def step(self, k):
        observation, reward, terminated, truncated, info = self.env.step(k)
        self.i += 1

        norm = np.linalg.norm(self.idealTrajectory[self.i, :]-observation)
        loss = -0.5*norm*norm
        # c1 = (2 - 0.5*(self.idealTrajectory[self.i, 0] - observation[0])**2)
        # c2 = (60- 0.5*(self.idealTrajectory[self.i, 1] - observation[1])**2)
        # c1 = (4 - np.abs(observation[0]))
        # c2 = (15 - 0.5*(self.idealTrajectory[self.i, 1] - observation[1])**2)
        c2 = 2.2 - np.abs(observation[1])

        rewards = np.array([loss, c2])
        return observation, rewards, terminated, truncated, info

    def render(self):
        return self.env.render()


class PendulumGymEnvWithDynamics(PendulumEnv):
    def __init__(self, config, U, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.config = config

        self.max_speed = 100
        self.max_torque = 100
        self.dt = config["dt"]
        self.g = config["g"]
        self.m = config["m"]
        self.l = config["L"]
        self.U = U

    def _get_obs(self):
        return self.state

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset()
        self.state = np.array([self.config["x0"], self.config["x0_dot"]])

        return self._get_obs(), {}

    def step(self, k):
        config = copy.deepcopy(self.config)
        config["kp_bo"] = k[0]
        config["kd_bo"] = k[1]

        state = self._get_obs()
        action = np.array([self.U(state, config)])

        return super().step(action)

    def startExperiment(self,k):
        pass

    def afterExperiment(self):
        pass
