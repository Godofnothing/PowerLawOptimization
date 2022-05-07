import os
import torch
import numpy as np


def load_kernel_and_err(data_root: str, model: str, dataset: str, normalize_ntk: bool = True):
    # assert os.path.isdir('{data_root}/NT_kernels')
    # assert os.path.isdir('{data_root}/NT_errors')
    # save kernel
    K = torch.tensor(np.load(f'{data_root}/NT_kernels/{model}_{dataset}_label.npy'))
    # save err vec
    d_f = torch.tensor(np.load(f'{data_root}/NT_errors/{model}_{dataset}_label.npy'))
    if normalize_ntk:
        # get kernel size
        N = K.shape[0]
        # renormalize NTK
        mult_factor = N / torch.linalg.eigvalsh(K).max()
        K *= mult_factor
    # normalize err
    d_f = (d_f - d_f.mean()) / d_f.std()
    return K, d_f 
