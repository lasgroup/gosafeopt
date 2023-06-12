import numpy as np
import torch


def rand(start, end, n, m=1):
    return (start - end) * np.random.rand(m, n) + end * np.ones(shape=(m, n))


def rand_torch(start, end, n):
    return (start - end) * torch.rand(n) + end * torch.ones(n)


def rand2d_torch(start1, end1, start2, end2, n):
    rand = torch.ones(n, 2)
    rand[:, 0] = rand_torch(start1, end1, n)[:]
    rand[:, 1] = rand_torch(start2, end2, n)[:]
    return rand


def rand2n_torch(start, end, n, m):
    rand = torch.zeros(n, m)
    for i in range(m):
        rand[:, i] = rand_torch(start[i], end[i], n)[:]
    return rand
