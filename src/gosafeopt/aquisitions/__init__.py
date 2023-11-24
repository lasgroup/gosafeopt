from gosafeopt.aquisitions.go_safe_opt import GoSafeOpt
from gosafeopt.aquisitions.max_mean import MaxMean
from gosafeopt.aquisitions.safe_ei import SafeEI
from gosafeopt.aquisitions.safe_opt import SafeOpt
from gosafeopt.aquisitions.safe_ucb import SafeUCB
from gosafeopt.aquisitions.safe_opt_grid import SafeOptGrid
from gosafeopt.aquisitions.ucb import UCB


def get_aquisition(model, config, context, data):
    if config["aquisition"] == "SafeOptGrid":
        aquisition = SafeOptGrid
    elif config["aquisition"] == "SafeUCB" and config["acf_optim"] == "swarm":
        aquisition = SafeOpt
        config["use_ucb"] = True
    elif config["aquisition"] == "SafeUCB":
        aquisition = SafeUCB
    elif config["aquisition"] == "SafeEI":
        aquisition = SafeEI
    elif config["aquisition"] == "UCB":
        aquisition = UCB
        config["set_init"] = "random"
    elif config["aquisition"] == "MaxMean":
        aquisition = MaxMean
    elif config["aquisition"] == "GoSafeOpt":
        aquisition = GoSafeOpt
    elif config["aquisition"] == "SafeOpt":
        aquisition = SafeOpt
    else:
        raise Exception("Aquisition not implemented")

    return aquisition(model, config, context, data)
