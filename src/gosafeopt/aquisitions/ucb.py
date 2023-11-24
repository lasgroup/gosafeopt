from gosafeopt.aquisitions.base_aquisition import BaseAquisition


class UCB(BaseAquisition):
    def __init__(self, model, c, context=None, data=None):
        super().__init__(model, c, context, data)

    def evaluate(self, X):
        _, ucb = self.get_confidence_interval(X)

        loss_perf = ucb[:, 0]

        return loss_perf
