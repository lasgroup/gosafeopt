import numpy as np
from gosafeopt.tools.math import clamp, angleDiff


def U_ideal(x, c):
    return (
        c["m"]
        * c["L"]
        * c["L"]
        * (-c["k1"] * angleDiff(c["pi"], x[0]) - c["k2"] * x[1] - c["g"] / c["L"] * np.sin(x[0]))
    )


def U_learned(x, c):
    return U_applied(x, c) + c["kp_bo"] * angleDiff(c["pi"], x[0]) + c["kd_bo"] * x[1]


def U_applied(x, c):
    return U_ideal(x, c) - c["kp"] * angleDiff(c["pi"], x[0]) - c["kd"] * x[1]


def dynamics_ideal(x, U, c):
    def f(u):
        return np.array([u[1], c["g"] / c["L"] * np.sin(u[0]) + U(u, c) / (c["m"] * c["L"] * c["L"])]).T

    res = integrate(x, f, c["dt"])
    return res


def dynamics_real(x, U, c):
    def U_p(t, c):
        return U_applied(t, c, U)  # Disturbance

    return dynamics_ideal(x, U_p, c)


def integrate(x, f, dt):
    x_new = np.zeros_like(x)
    x_dot = f(x)
    x_new[1] = x[1] + dt * x_dot[1]
    x_new[0] = x[0] + dt * x_new[1]
    return x_new
