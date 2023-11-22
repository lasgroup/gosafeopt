import torch
import gosafeopt
from gosafeopt.aquisitions.multi_stage_aquisition import MultiStageAquisition
from gosafeopt.tools.misc import singleton
from torch.distributions.multivariate_normal import MultivariateNormal


@singleton
class SafeOptMultiStageState(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.best_lcb = -1e10


class SafeOptMultiStage(MultiStageAquisition):
    def __init__(self, model, c, context=None, data=None):
        stages = [
            (self.lowerBound, False),
            (self.maximizers, True),
            (self.expanders, True)
        ]

        super().__init__(model, c, stages, context, data)

        self.safeOptState = SafeOptMultiStageState()

    def hasNextStage(self):
        if self.c["use_ucb"]:
            return self.currentStage < 2
        else:
            return super().hasNextStage()


    def lowerBound(self, X):
        l, _ = self.getBounds(X)

        maxLCB = torch.max(l[:, 0])
        if maxLCB > self.safeOptState.best_lcb:
            self.safeOptState.best_lcb = maxLCB

        slack = l - self.fmin

        return l[:, 0] + self.penalties(slack)

    def maximizers(self, X):
        l, u = self.getBounds(X)
        scale = 1  # if not self.c["normalize_output"] else self.model.models[0].outcome_transform._stdvs_sq[0]
        values = (u - l)[:, 0]/scale
        improvement = u[:, 0]-self.safeOptState.best_lcb

        interest_function = torch.sigmoid(100*improvement/scale)
        interest_function -= interest_function.min()
        c = interest_function.max() - interest_function.min()
        c[c < 1e-5] = 1e-5
        interest_function /=  c

        slack = l - self.fmin
        penalties = self.penalties(slack)

        value = (values + penalties)*interest_function

        return value

    def expanders(self, X):
        l, u = self.getBounds(X)

        scale = 1  # if not self.c["normalize_output"] else self.model.models[0].outcome_transform._stdvs_sq[0]
        values = (u - l)[:, 0]/scale

        slack = l - self.fmin
        penalties = self.penalties(slack)
        # print(penalties)

        # TODO how to set scale?
        normal = MultivariateNormal(
            loc=torch.zeros_like(slack[:, 1:], device=gosafeopt.device),
            covariance_matrix=torch.eye(slack.shape[1]-1, device=gosafeopt.device)
        )
        interest_function = normal.log_prob(slack[:, 1:])
        interest_function -= interest_function.min()
        c = interest_function.max() - interest_function.min()
        c[c < 1e-5] = 1e-5
        interest_function /=  c

        value = (values + penalties)*interest_function

        return value

    def reset(self):
        self.safeOptState.best_lcb = -1e10
