from gymnasium import Env


class Environment(Env):

    def __init__(self, render_mode=None):
        self.best_k = None
        self.render_mode = render_mode

    def setBestK(self, best_k):
        self.best_k = best_k

    def getIdealTrajecory(self):
        return None

    def startExperiment(self,k):
        pass

    def backup(self, k):
        pass

    def reset(self):
        pass

    def afterExperiment(self):
        pass

