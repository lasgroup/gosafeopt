from abc import ABC, abstractmethod
from typing import Optional
from gymnasium import Env
import torch


class Environment(Env, ABC):
    def __init__(self, best_param: Optional[torch.Tensor], render_mode: Optional[str] = None):
        self.best_param = best_param
        self.render_mode = render_mode

    @abstractmethod
    def backup(self, params):
        """
        This method will be called with the backup parameters of the backup strategy.
        This method should send the backup parameters to the robot/sim.
        """
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def before_experiment(self, params):
        """
        This method will be executed before each loss evaluation.
        This method should be implemented to set the parameters that will be evaluated on the robot/sim.
        I.e in hardware experiments this could also be used for waiting until the experiment is setup.
        """
        pass

    @abstractmethod
    def after_experiment(self):
        """
        This method will be executed after each loss evaluation.
        Useful i.e for setting the best found parameters so far to stabilize the robot.
        """
