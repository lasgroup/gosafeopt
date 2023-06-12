
from gosafeopt.optim.grid_opt import GridOpt
from gosafeopt.optim.scipy_opt import ScipyOpt
from gosafeopt.optim.swarm_opt import SwarmOpt


def get_optimizer(aquisition, config, context=None):
        if config["acf_optim"] == "scipy":
            optim = ScipyOpt

        elif config["acf_optim"] == "swarm":
            optim = SwarmOpt

        elif config["acf_optim"] == "grid":
            optim = GridOpt

        else:
            raise Exception("Optimizer is not Implemented") 

        return optim(aquisition, config, context)
