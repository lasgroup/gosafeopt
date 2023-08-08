import torch
import gosafeopt
from gosafeopt.aquisitions.go_safe_opt import GoSafeOpt
from gosafeopt.aquisitions.multi_stage_aquisition import MultiStageAquisition
from gosafeopt.aquisitions.safe_opt_multistage import SafeOptMultiStage
from gosafeopt.tools.points import random, uniform
from gosafeopt.tools.logger import Logger
from abc import abstractmethod
from gosafeopt.tools.misc import singleton


@singleton
class OptimizerState(object):

    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.safeSet = []
        self.safeSetNr = 0
        self.bestSafeSetNr = 0
        self.yMin = -1e10 
        self.globalYMin= -1e10 
        self.config = self.config 
        self.i = 0

    def getSafeSet(self):
        if len(self.safeSet) == 0:
            return None
        else:
            return self.safeSet[self.safeSetNr]

    def updateSafeSet(self, idx):
        self.safeSet[self.safeSetNr] = self.safeSet[self.safeSetNr][idx]

    def appendSafeSet(self, safeset):
        safeset.to(gosafeopt.device)
        self.safeSet[self.safeSetNr] = torch.vstack([self.getSafeSet(), safeset])

        # if self.safeSet[self.safeSetNr].shape[0] > 10*self.config["set_size"]:
        #     randIdx = torch.randint(0, self.safeSet[self.safeSetNr].shape[0], (10*self.config["set_size"],))
        #     self.safeSet[self.safeSetNr] = self.safeSet[self.safeSetNr][randIdx]

    def addSafeSet(self, safeset):
        safeset.to(gosafeopt.device)
        self.safeSet.append(safeset)

    def changeToLastSafeSet(self):
        self.i = 0
        self.safeSetNr = len(self.safeSet) - 1
        self.yMin = -1e10 
        Logger.info(f"BestSet: {self.bestSafeSetNr} / CurrentSet: {self.safeSetNr}")

    def changeToBestSafeSet(self):
        self.i = 0
        self.safeSetNr =  self.bestSafeSetNr
        Logger.info(f"Changing to best set Nr. {self.bestSafeSetNr}")

    def updateSafeSets(self, yMin):
        if self.globalYMin < yMin:
            self.globalYMin = yMin
            self.bestSafeSetNr = self.safeSetNr
            Logger.info(f"BestSet: {self.bestSafeSetNr}")

        if self.yMin < yMin:
            self.yMin = yMin

        if self.yMin < (self.config["safe_opt_max_tries_without_progress_tolerance"]*self.globalYMin if self.globalYMin > 0 else (2-self.config["safe_opt_max_tries_without_progress_tolerance"])*self.globalYMin):
            self.i += 1

        if self.i >= self.config["safe_opt_max_tries_without_progress"]:
            self.changeToBestSafeSet()


class BaseOptimizer:

    def __init__(self, aquisition, c, context):
        self.aquisition = aquisition
        self.c = c
        self.context = context

    def evaluate_aquisition(self, X):
        if isinstance(self.aquisition, MultiStageAquisition):
            return self.aquisition.forward(X, self.stage)
        else:
            return self.aquisition(X)

    @abstractmethod
    def optimize(self):
        pass

    def getInitPoints(self, mode, append_train_set=False, useContext=True):
        # TODO shouldn't be here
        goSafeOptOverride = isinstance(self.aquisition, GoSafeOpt) and self.aquisition.goState.getState() == "s3"
        goSafeOptS1Override = isinstance(self.aquisition, GoSafeOpt) and self.aquisition.getStage()[0].__name__ == "lowerBound"

        if mode == "random" or goSafeOptOverride:
            X = random(self.c["domain_start"], self.c["domain_end"], self.c["set_size"], self.c["dim_params"])

        elif mode == "uniform":
            X = uniform(self.c["domain_start"], self.c["domain_end"], self.c["set_size"], self.c["dim_params"])

        elif mode == "safe":
            state = OptimizerState(self.c)
            N = self.c["set_size"]

            for i in in range(len(state.safeSet)):
                state.safeSet[i] = state.safeSet[i].to(gosafeopt.device)

            safeSet = torch.vstack(state.safeSet) if goSafeOptS1Override and state.getSafeSet() is not None else state.getSafeSet()

            #TODO this doesn't work for different contexts
            if safeSet is None or len(safeSet) == 0:
                safeSet = self.aquisition.data.train_x[-1:].to(gosafeopt.device)


            if safeSet.shape[0] >= N:
                randIdx = torch.randint(0, safeSet.shape[0], (N,))
                X = safeSet[randIdx]
                useContext = False
            else:
                # print(safeSet.mean(axis=0))
                distribution = torch.distributions.MultivariateNormal(safeSet.mean(axis=0), 1e-3*torch.eye(safeSet.shape[1], device=gosafeopt.device))
                X = distribution.rsample([N])
                X[:, -self.c["dim_context"]:] = self.context.repeat(N, 1)
                X[:len(safeSet)] = safeSet
                useContext = False

        else:
            raise Exception("Set init not defined")

        if self.context is not None and useContext:
            X = torch.hstack([X, self.context.repeat(len(X), 1)])

        if append_train_set:
            train_inputs = self.aquisition.data.train_x
            X = torch.vstack([X, train_inputs])

        return X.to(gosafeopt.device)

    def updateSafeSet(self, X):
        safeSet = self.aquisition.safeSet(X)
        state = OptimizerState(self.c)
        if len(state.safeSet) > 0:
            stillSafe = self.aquisition.safeSet(state.getSafeSet())
            state.updateSafeSet(stillSafe)  # Remove unsafe points
            state.appendSafeSet(X[safeSet])
        else:
            state.addSafeSet(X[safeSet])

    def optimizeMultiStage(self):
        X, reward = None, None
        rewardMax = -1e10
        rewardStage = 0

        while self.aquisition.hasNextStage():
            self.stage, append_data = self.aquisition.getStage()
            [Xtmp, rewardTmp] = self.optimize()
            if append_data:
                if X is None:
                    X, reward = Xtmp, rewardTmp
                else:
                    X = torch.vstack([X, Xtmp])
                    reward = torch.vstack([reward, rewardTmp])

                if rewardTmp.max() > rewardMax:
                    rewardMax = rewardTmp.max()
                    rewardStage = self.stage.__name__
            self.aquisition.advanceStage()
        self.aquisition.advanceState()

        Logger.info(f"MultiStageAquisition is taken from stage {rewardStage}")
        return [X, reward]

    def getNextPoint(self):
        # Some aquisitions have multiple optimization stages...
        if isinstance(self.aquisition, MultiStageAquisition):
            [X, reward] = self.optimizeMultiStage()
        else:
            [X, reward] = self.optimize()

        self.aquisition.reset()

        nextX = X[torch.argmax(reward)]
        reward = reward.max()


        if self.c["set_init"] == "safe":
            self.updateSafeSet(X)
        if not self.aquisition.hasSafePoints(X):
            Logger.warn("Could not find safe set")

        return [nextX.detach().to("cpu"), reward.detach().to("cpu")]
