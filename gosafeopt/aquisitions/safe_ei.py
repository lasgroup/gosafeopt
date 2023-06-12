from gosafeopt.aquisitions.base_aquisition import BaseAquisition
from scipy.stats import norm
import torch


class SafeEI(BaseAquisition):
    def __init__(self, model, c, context = None, data=None):
        super().__init__(model, c, context, data)

    def evaluate(self, X):
        l, u = self.getBounds(X)
        
        xi = 0.01

        mu = X.mean.reshape(-1, self.c["dim_obs"])[:,0]
        sigma  = X.variance.reshape(-1, self.c["dim_obs"])[:,0]

        mean_sample = self.model.posterior(self.data.train_x).mean.detach()[:,0]
        mu_sample_opt = torch.max(mean_sample)

        imp = mu - mu_sample_opt - xi
        Z = (imp / sigma).detach().numpy()
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

        slack = l - self.fmin


        if self.c["use_soft_penalties"]:
            ei += self.penalties(slack)
        else:
            ei[:] = torch.all(l[:, 1:] > self.fmin[1:], axis=1)
            ei [~S] = -1e10
        return ei  
