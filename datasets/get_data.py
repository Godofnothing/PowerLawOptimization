import torch

from torchvision.datasets import MNIST, CIFAR10
from sklearn.model_selection import train_test_split


def get_image_dataset(dataset='mnist', N: int =1000, data_root='./data', seed: int = 42, device='cpu'):
    DATASET = MNIST if dataset == 'mnist' else CIFAR10
    train_dataset = DATASET(root=data_root, train=True, download=True)
    val_dataset = DATASET(root=data_root, train=False, download=True)
    with torch.no_grad():
        train_inputs  = torch.tensor(train_dataset.data.reshape(len(train_dataset), -1), dtype=torch.float32)
        train_targets = torch.tensor(train_dataset.targets, dtype=torch.float32)
    # get mean and std
    inputs_mean, inputs_std   = train_inputs.mean(), train_inputs.std()
    targets_mean, targets_std = train_targets.mean(), train_targets.std()
    train_inputs = (train_inputs - inputs_mean) / inputs_std
    # extract inputs and targets
    train_inputs, _, train_targets, _ = train_test_split(
        train_inputs,
        train_targets,
        shuffle=True,
        stratify=train_dataset.targets,
        train_size=N,
        random_state=seed
    )
    with torch.no_grad():
        val_inputs  = torch.tensor(val_dataset.data.reshape(len(val_dataset), -1), dtype=torch.float32)
        val_targets = torch.tensor(val_dataset.targets, dtype=torch.float32)
    val_inputs = (val_inputs - inputs_mean) / inputs_std
    # normalize targets as well
    train_targets = (train_targets - targets_mean) / targets_std
    val_targets   = (val_targets   - targets_mean) / targets_std
    # all to device
    return train_inputs.to(device), train_targets.to(device), val_inputs.to(device), val_targets.to(device)