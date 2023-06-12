import torch
import gpytorch

from gosafeopt.aquisitions.base_aquisition import BaseAquisition
from gosafeopt.tools.logger import Logger


class MultiStageAquisition(BaseAquisition):

    def __init__(self, model, c, stages, context=None, data=None):
        super().__init__(model, c, context, data)
        self.stages = stages
        self.currentStage = 0

    def getStage(self):
        return self.stages[self.currentStage]

    def advanceStage(self):
        self.currentStage += 1

    def advanceState(self):
        pass

    def hasNextStage(self):
        return self.currentStage <= len(self.stages) - 1

    def forward(self, X, stage):
        self.points = X
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            with gpytorch.settings.fast_pred_samples():
                x = self.model.posterior(X)

        return stage(x)

    def evaluate(self, X):
        Logger.warn("Evaluate not implemented for multistage aquisition")
