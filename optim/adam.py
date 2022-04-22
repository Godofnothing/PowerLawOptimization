import torch
from typing import Callable
from .base import Optimizer


__all__ = ['Adam']


class Adam(Optimizer):

    def __init__(
        self,  alpha_fn: Callable, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8
    ) -> None:
        self.alpha_fn = alpha_fn
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    def step(self, step: int, state: dict, K: torch.Tensor, batch_idx: torch.Tensor):
        # get current rate and momentum 
        alpha, b1, b2, eps = self.alpha_fn(step), self.b1, self.b2, self.eps
        B = len(batch_idx)
        # unwrap state
        d_f, p_f, v_f = state.get('d_f'), state.get('p_f'), state.get('v_f')
        # get grad
        g_f = K[:, batch_idx] @ d_f[batch_idx] / B
        # First moment
        p_f = (1 - b1) * g_f + b1 * p_f 
        # Second moment
        v_f = (1 - b2) * (g_f ** 2) + b2 * v_f
        # Bias correction
        p_hat = p_f / (1 - b1 ** (step + 1)) 
        v_hat = v_f / (1 - b2 ** (step + 1))
        d_f -= alpha * p_hat / (torch.sqrt(v_hat) + eps) 
        # update state
        state['d_f'], state['p_f'], state['v_f'] = d_f, p_f, v_f
        return state
