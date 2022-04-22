import torch
import numpy as np

from optim import Optimizer


__all__ = ['train_linearized']


def train_linearized(
    optimizer: Optimizer,
    state: dict, 
    K: torch.Tensor, 
    n_steps: int, 
    batch_size: int, 
):
    loss_curve = []
    # create all ids
    all_idx = torch.arange(K.shape[0])
    for step in range(n_steps):
        # generate batch idx
        batch_idx = np.random.choice(all_idx, size=batch_size, replace=False)
        # make optimizer step
        state = optimizer.step(step, state, K, batch_idx)
        # compute loss
        loss = 0.5 * torch.mean(state['d_f'] ** 2)
        # update loss curve
        loss_curve.append(loss.item())
        
    return state, loss_curve
