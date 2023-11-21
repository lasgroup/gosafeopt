import math
import torch


def clamp(t, a):
    return abs(t) / t * a if abs(t) > a else t


def clamp2dTensor(inp, lower, upper):
    N = inp.shape[1]
    for i in range(N):
        inp[:, i] = torch.clamp(inp[:, i], lower[i], upper[i])

    return inp


def scale(X, s):
    return X/s


def iScale(X, s):
    return X*s


def angleDiff(a, b):
    c = b - a
    return (c + math.pi) % (2 * math.pi) - math.pi
