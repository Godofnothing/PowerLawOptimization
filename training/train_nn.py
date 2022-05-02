import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def train_step(
    model : nn.Module, 
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer : torch.optim.Optimizer
):
    model.train()
    # get model output
    outputs = model(inputs).squeeze(-1)
    # compute loss
    loss = 0.5 * F.mse_loss(outputs, targets)
    # make gradient step  
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 
    # compute acc
    acc = torch.sum(torch.round(outputs) == targets) / len(inputs)
    return loss, acc


@torch.no_grad()
def val_step(
    model : nn.Module, 
    inputs: torch.Tensor,
    targets: torch.Tensor
):
    model.eval()
    # get model output
    outputs = model(inputs).squeeze(-1)
    # compute loss
    loss = 0.5 * F.mse_loss(outputs, targets)
    # compute acc
    acc = torch.sum(torch.round(outputs) == targets) / len(inputs)
    return loss, acc


def train(
    n_steps: int,
    model: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    inputs: torch.Tensor, 
    targets: torch.Tensor, 
    batch_size: int,
    val_inputs: torch.Tensor = None,
    val_targets: torch.Tensor = None,
    val_frequency: int = 1,
    log_frequency: int = 1,
    verbose: bool = False
):
    history = {'train/loss': [], 'train/acc': [], 'val/loss': [], 'val/acc': []}
    # create all ids
    all_idx = torch.arange(len(inputs))
    for step in range(n_steps):
        # generate batch idx
        batch_idx = np.random.choice(all_idx, size=batch_size, replace=False)
        batch_inputs, batch_targets = inputs[batch_idx], targets[batch_idx]
        # make step
        train_step(model, batch_inputs, batch_targets, optimizer)
        # evaluate on whole train
        train_loss, train_acc = val_step(model, inputs, targets)
        # update history
        history['train/loss'].append(train_loss.item())
        history['train/acc'].append(train_acc.item())
        if step % val_frequency == 0:
            # evaluate on whole test
            val_loss, val_acc = val_step(model, val_inputs, val_targets)
            history['val/loss'].append(val_loss.item())
            history['val/acc'].append(val_acc.item())
        # log only train (for simplicity)
        if verbose and step % log_frequency == 0:
            print('-' * 10)
            print(f'Step {step}')
            print(f"{'Train':>5} Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
    
    return history