import torch
from typing import Callable
from .base import Optimizer

__all__ = ['SGD']


class SGD(Optimizer):

    def __init__(
        self, alpha_fn: Callable, beta_fn : Callable = lambda step: 0.0
    ) -> None:
        self.alpha_fn = alpha_fn
        self.beta_fn = beta_fn

    def step(self, step: int, state: dict, K: torch.Tensor, batch_idx: torch.Tensor):
        # get current rate and momentum 
        alpha, beta = self.alpha_fn(step), self.beta_fn(step)
        # get batch size
        B = len(batch_idx)
        # unwrap state
        d_f, p_f = state.get('d_f'), state.get('p_f')
        # get grad
        g_f = K[:, batch_idx] @ d_f[batch_idx] / B
        # momentum step
        if beta > 0.0:
            p_f = g_f + beta * p_f
            # update momentum
            state['p_f'] = p_f
        else:
            p_f = g_f
        d_f -= alpha * p_f 
        # update state
        state['d_f'] = d_f
        return state
