import os
import torch
import pickle
import argparse

from copy import deepcopy
from training import train_spectral_diagonal
from datasets import generate_synthetic_data


def parse_args():
    parser = argparse.ArgumentParser('Training of synthetic power law data', add_help=False)
    # Path to hparams
    parser.add_argument('--hparam_dict_path',  type=str, required=True)
    # Data params
    parser.add_argument('--N', default=1000, type=int)
    # Approx type
    parser.add_argument('--tau', default=1.0, type=float)
    # Optimizer
    parser.add_argument('--momentum', nargs='+', default=[0.0], type=float)
    # Training params
    parser.add_argument('--n_steps', default=10000, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--workers', default=8, type=int)
    # Whether to track diag err
    # Save params
    parser.add_argument('--save_dir',  type=str, default='./output')

    args = parser.parse_args()
    return args


def format_params(params):
    param_str = []
    for k, v in params.items():
        fmt_v = f'{v:.2e}' if isinstance(v, float) else f'{v}'
        param_str.append(f'{k}:{fmt_v}')
    return ' | '.join(param_str)


if __name__ == '__main__':
    args = parse_args()
    # get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # limit num workers
    torch.set_num_threads(args.workers)
    # make experiment dir if needed
    os.makedirs(f'{args.save_dir}', exist_ok=True)

    loss_curves = {}
    diag_errs = {}
    
    with open(args.hparam_dict_path, 'rb') as f:
        hparam_dict = pickle.load(f)

    for (nu, kappa), lr in hparam_dict.items():
        # define hparams for this experiment
        params = {
            'nu': nu,
            'kappa': kappa, 
            'batch_size' : args.batch_size
        }
        # generate data
        K, d_f = generate_synthetic_data(size=args.N, kappa=kappa, nu=nu)
        K, d_f = K.to(device), d_f.to(device)
        # get normalized spectrum
        lambda_f, U = torch.linalg.eigh(K)
        # get normalized spectrum
        lambda_f /= lambda_f.max()
        # covariances
        C = torch.diag(U.T @ torch.outer(d_f, d_f) @ U)
        for momentum in args.momentum:
            alpha_fn = lambda step: lr
            beta_fn  = lambda step: momentum
            params['lr'] = lr
            params['momentum'] = momentum
            # at the moment only constant batch_size is supported
            batch_fn = lambda step: args.batch_size
            # make initial state
            state = {'C' : deepcopy(C), 'J': torch.zeros_like(C), 'P': torch.zeros_like(C)}
            print(format_params(params), flush=True)
            state, loss_curve, Cs = train_spectral_diagonal(
                args.n_steps, state, lambda_f, 
                alpha_fn, beta_fn, batch_fn, tau=args.tau, 
            )
            # update dict with loss curves
            loss_curves[tuple(params.values())] = loss_curve

    print('Training completed!')
    # save data
    torch.save(loss_curves, f'{args.save_dir}/loss_curves.pth')
