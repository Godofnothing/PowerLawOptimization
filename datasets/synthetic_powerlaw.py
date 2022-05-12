import math
import torch


def generate_synthetic_data(size: int, kappa: float = 1.0, nu: float = 1.0, lambda_min: float = 0.0):
    # min lambda is lambda_min + N ** (-nu) in fact but we neglect the second term
    spec = lambda_min + (1 - lambda_min) * torch.arange(1, size + 1) ** (-nu)
    U = torch.empty(size, size)
    torch.nn.init.orthogonal_(U)
    # create NTK matrix
    K = size * U.T @ (spec[:, None] * U)
    # create init error
    coef  = torch.arange(1, size+1) ** (-(kappa+1) / 2)
    coef *= (1 / torch.linalg.norm(coef))
    d_f = U.T @ coef
    return K, d_f
    