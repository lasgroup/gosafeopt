import numpy as np


def integrate(x, f, dt):
    x_new = np.zeros_like(x)
    x_dot = f(x)
    x_new[1] = x[1] + dt * x_dot[1]
    x_new[0] = x[0] + dt * x_new[1]
    return x_new
