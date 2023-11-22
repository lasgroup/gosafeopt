import torch
import math
from torch.autograd import Variable

from gosafeopt.tools.rand import rand2n_torch


def random(start, end, set_size, dim):
    return rand2n_torch(start, end, set_size, dim)


def uniform(start, end, set_size, dim):

    init_points = []

    for i in range(dim):
        init_points.append(torch.linspace(start[i], end[i], math.isqrt(set_size)))

    X = torch.meshgrid(init_points, indexing="xy")
    init = torch.stack([x.flatten() for x in X]).T
    init = torch.reshape(init, (-1, dim))
    X = Variable(init, requires_grad=False)
    return X
