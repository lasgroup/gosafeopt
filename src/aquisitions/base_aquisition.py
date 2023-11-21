from abc import abstractmethod
from joblib.externals.loky import backend
import torch
from botorch.acquisition import AnalyticAcquisitionFunction
import gpytorch

import gosafeopt


class BaseAquisition(AnalyticAcquisitionFunction):

    def __init__(self, model, c, context=None, data=None):
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.model = model
        self.c = c
        self.context = context
        self.data = data

        self.fmin = torch.zeros(c["dim_obs"]).to(gosafeopt.device)

    def forward(self, X):
        self.points = X
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            with gpytorch.settings.fast_pred_samples():
                x = self.model.posterior(X)

        return self.evaluate(x)

    @abstractmethod
    def evaluate(self, x):
        pass

    def eval(self):
        self.model.eval()

    def getBounds(self, X):
        mean = X.mean.reshape(-1, self.c["dim_obs"])
        var = X.variance.reshape(-1, self.c["dim_obs"])

        # Upper and lower confidence bound
        l = mean - self.c["scale_beta"]*torch.sqrt(self.c["beta"]*var)
        u = mean + self.c["scale_beta"]*torch.sqrt(self.c["beta"]*var)

        return l, u

    def safeSet(self, X):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            with gpytorch.settings.fast_pred_samples():
                xTmp = self.model.posterior(X)

        l, _ = self.getBounds(xTmp)

        S = torch.all(l[:, 1:] > self.fmin[1:], axis=1)
        return S

    def hasSafePoints(self, X):
        return torch.any(self.safeSet(X))

    def penalties(self, slack):
        penalties = torch.clip(slack, None, 0)

        penalties[(slack < 0) & (slack > -0.001)] *= 2
        penalties[(slack <= -0.001) & (slack > -0.1)] *= 5
        penalties[(slack <= -0.1) & (slack > -1)] *= 10

        slack_id = slack < -1
        penalties[slack_id] = -300 * penalties[slack_id] ** 2
        # penalties *= 10000
        return torch.sum(penalties[:, 1:], axis=1)

    def reset(self):
        pass
