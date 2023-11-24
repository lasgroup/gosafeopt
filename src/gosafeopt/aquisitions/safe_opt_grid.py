from gosafeopt.aquisitions.base_aquisition import BaseAquisition
import torch

import gpytorch


class SafeOptGrid(BaseAquisition):
    def __init__(self, model, c, context=None, data=None):
        super().__init__(model, c, context, data)
        self.min_var = torch.ones(c["dim_obs"]) * c["min_var"]

    def evaluate(self, X):
        # Safe set, expanders and maximizes
        S = torch.zeros(X.mean.shape[0], dtype=bool)
        S[:] = False
        G = S.clone()
        M = S.clone()

        # Upper and lower confidence bound
        l, u = self.get_confidence_interval(X)

        # Compute Safe Set
        S[:] = torch.all(l[:, 1:] > self.fmin[1:], axis=1)

        if not torch.any(S):
            res = -1e10 * torch.ones_like(X.mean[:, 0])
            safestPoint = torch.argmax(l.min(axis=1)[0])
            res[safestPoint] = -1e5
            return res

        # Set of maximisers
        M[S] = u[S, 0] >= torch.max(l[S, 0])
        max_var = torch.max(u[M, 0] - l[M, 0])

        # Optimistic set of possible expanders
        s = torch.logical_and(S, ~M)
        idx = s.clone()
        s[idx] = torch.max((u[idx, 1:] - l[idx, 1:]), axis=1)[0] > max_var
        idx = s.clone()
        s[idx] = torch.any(u[idx, 1:] - l[idx, 1:] > self.min_var[1:], axis=1)

        if torch.any(s) and not torch.all(S):
            # set of safe expanders
            G_safe = torch.zeros(torch.count_nonzero(s), dtype=bool)
            sort_index = torch.max(u[s, :] - l[s, :], axis=1)[0].argsort()
            for index in reversed(sort_index):
                fantasyTarget = u[s][index]
                fantasyInput = self.points[s][index]

                # Fantasize doesn't transform inputs somehow
                if self.config["normalize_input"]:
                    fantasyInput = self.model.models[0].input_transform(fantasyInput)[0].squeeze()

                fModel = self.model.condition_on_observations(
                    fantasyInput.repeat(self.config["dim_obs"], 1, 1), fantasyTarget.reshape(1, 1, -1)
                )
                fModel.eval()

                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    with gpytorch.settings.fast_pred_samples():
                        pred = fModel.posterior(self.points[~S])

                l2 = (
                    pred.mean.detach()[0]
                    - self.config["scale_beta"] * torch.sqrt(self.config["beta"] * pred.variance.detach())[0]
                )
                G_safe[index] = torch.any(torch.all(l2[:, 1:] >= self.fmin[1:], axis=1))

                if G_safe[index]:
                    break

            G[s] = G_safe

        MG = torch.logical_or(M, G)
        value = torch.max((u - l), axis=1)[0]

        if self.config["use_soft_penalties"]:
            slack = l - self.fmin
            value[~MG] += self.soft_penalty(slack[~MG])
        else:
            value[~MG] = -1e10

        return value.double()
