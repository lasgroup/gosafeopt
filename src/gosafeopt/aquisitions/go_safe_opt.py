
from gosafeopt.aquisitions.safe_opt_multistage import SafeOptMultiStage
from gosafeopt.tools.misc import singleton
import torch
import gosafeopt


@singleton
class GoSafeOptState(object):

    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.max_s1 = self.config["max_s1"]
        self.max_s3 = self.config["max_s3"]
        self.n = 0
        self.previousState = "s1"

    def goToS1(self):
        self.n = 0

    def advance(self):
        self.previousState = self.getState()
        self.n += 1
        self.n %= self.max_s1 + self.max_s3

    def skipBackupAtRollout(self):
        return self.previousState == "s1"

    def getState(self, n=None):
        if self.n < self.max_s1:
            return "s1"
        elif self.n < self.max_s3 + self.max_s1:
            return "s3"


class GoSafeOpt(SafeOptMultiStage):
    def __init__(self, model, c, context=None, data=None):
        super().__init__(model, c, context, data)

        self.goState = GoSafeOptState(c)
        self.s3_executed = False

    def hasNextStage(self):
        state = self.goState.getState()
        has = state == "s1" and super().hasNextStage() or state == "s3" and not self.s3_executed
        return has

    def getStage(self):
        state = self.goState.getState()
        if state == "s1":
            return super().getStage()
        else:
            return self.s3, True

    def advanceState(self):
        self.goState.advance()

    def advanceStage(self):
        if self.goState.getState() == "s1":
            return super().advanceStage()
        else:
            self.s3_executed = True

    # TODO No need to compute posterior
    def s3(self, X):
        data = self.data.train_x
        if self.data.failed_k is not None:
            data = torch.vstack([data, self.data.failed_k])

        distance = self.model.models[0].covar_module.covar_dist(data.to(gosafeopt.device), self.points).min(axis=0)[0]

        return distance
