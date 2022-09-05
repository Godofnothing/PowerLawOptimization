import torch
import numpy as np
import pandas as pd

from torchvision.datasets import MNIST, CIFAR10
from sklearn.datasets import load_digits, fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


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


def get_sklearn_data(dataset='olivetti_faces', N: int = 1000, data_root='./data', seed: int = 42, device='cpu'):
    if dataset == 'olivetti_faces':
        # get dataset
        X, y = fetch_olivetti_faces(
            data_home=data_root, 
            download_if_missing=True, 
            return_X_y=True
        )
    elif dataset == 'digits':
        X, y = load_digits(return_X_y=True)

    
    if len(X) > N:
        # get subset of uniformly distributed between classes digits
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, shuffle=True, random_state=seed, train_size=N)
    else:
        X_train, y_train = X, y
        X_val, y_val = np.empty_like(X), np.empty_like(y) 
    # select samples
    X_train = torch.tensor(X_train).to(torch.float32)
    y_train = torch.tensor(y_train).to(torch.float32)
    X_val = torch.tensor(X_val).to(torch.float32)
    y_val = torch.tensor(y_val).to(torch.float32)
    # normalize
    X_mean, X_std = X_train.mean(), X_train.std()
    y_mean, y_std = y_train.mean(), y_train.std()
    X_train, X_val = (X_train - X_mean) / X_std, (X_val - X_mean) / X_std
    y_train, y_val = (y_train - y_mean) / y_std, (y_val - y_mean) / y_std
    return X_train.to(device), y_train.to(device), X_val.to(device), y_val.to(device)


def get_uci_data(dataset='bike_sharing', N: int = 1000, data_root='./data', seed: int = 42, device='cpu'):
    if dataset == 'bike_sharing':
        # get data
        df = pd.read_csv(f'{data_root}/Bike-Sharing/hour.csv')
        # extract labels and targets
        X = df.drop(['instant', 'dteday', 'casual', 'registered', 'cnt'], axis=1)
        y = df['cnt']
        X, y = X.to_numpy(), y.to_numpy()
    elif dataset == 'sgemm_product':
        # get data
        df = pd.read_csv(f'{data_root}/sgemm_product/sgemm_product.csv')
        # transform categorical columns
        X = df.drop([f'Run{i} (ms)' for i in range(1, 5)], axis=1)
        y = df[[f'Run{i} (ms)' for i in range(1, 5)]]
        X, y = X.to_numpy(), y.to_numpy().mean(axis=1)
    else:
        raise NotImplementedError("Not added to the bases")
    
    if len(X) > N:
        # get subset of uniformly distributed between classes digits
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, shuffle=True, random_state=seed, train_size=N)
    else:
        X_train, y_train = X, y
        X_val, y_val = np.empty_like(X), np.empty_like(y) 
    # select samples
    X_train = torch.tensor(X_train).to(torch.float32)
    y_train = torch.tensor(y_train).to(torch.float32)
    X_val = torch.tensor(X_val).to(torch.float32)
    y_val = torch.tensor(y_val).to(torch.float32)
    # normalize
    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
    y_mean, y_std = y_train.mean(axis=0), y_train.std(axis=0)
    X_train, X_val = (X_train - X_mean) / X_std, (X_val - X_mean) / X_std
    y_train, y_val = (y_train - y_mean) / y_std, (y_val - y_mean) / y_std
    return X_train.to(device), y_train.to(device), X_val.to(device), y_val.to(device)

    