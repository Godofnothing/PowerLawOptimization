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
    track_diag_err = False,
    track_freq = 1
):
    loss_curve = []
    # add loss on 0th iteration
    loss = 0.5 * torch.mean(state['d_f'] ** 2)
    loss_curve.append(loss.item())
    if track_diag_err:
        Cs = torch.zeros((n_steps // track_freq, len(K)))
        # one will the eigenbasis
        _, U = torch.linalg.eigh(K)
        U = torch.flip(U, dims=(1,))
    else:
        Cs, U = None, None
    # create all ids
    all_idx = torch.arange(K.shape[0])
    for step in range(1, n_steps + 1):
        # generate batch idx
        batch_idx = np.random.choice(all_idx, size=batch_size, replace=False)
        # make optimizer step
        state = optimizer.step(step, state, K, batch_idx)
        # compute loss
        loss = 0.5 * torch.mean(state['d_f'] ** 2)
        # update loss curve
        loss_curve.append(loss.item())
        if track_diag_err and step % track_freq == 0:
            Cs[step // track_freq] = (state['d_f'] @ U).cpu()
        
    return state, loss_curve, Cs
