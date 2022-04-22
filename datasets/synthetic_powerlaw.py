import math
import torch


def generate_synthetic_data(size: int, kappa: float = 1.0, nu: float = 1.0):
    spec = torch.arange(1, size + 1) ** (-nu)
    U = torch.empty(size, size)
    torch.nn.init.orthogonal_(U)
    # create NTK matrix
    K = size * U.T @ (spec[:, None] * U)
    # create init error
    coef = torch.arange(1, size+1) ** (-(kappa+1) / 2)
    coef /= (2.0 * (coef ** 2).sum())
    d_f = math.sqrt(2 * size) * (U.T @ coef)
    return K, d_f
    