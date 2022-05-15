import os
import torch
import argparse
import itertools

from copy import deepcopy
from training import train_approx
from datasets import generate_synthetic_data, generate_mnist_ntk_data, \
    generate_cifar10_ntk_data, load_kernel_and_err
from schedules import JacobiScheduleA, JacobiScheduleB


def parse_args():
    parser = argparse.ArgumentParser('Training of synthetic power law data', add_help=False)
    # Approx type
    parser.add_argument('--approx', default='mean_field', choices=['mean_field', 'gauss'], type=str)
    # Data params
    parser.add_argument('--dataset', default='synthetic', choices=['synthetic', 'mnist', 'cifar10'], type=str)
    parser.add_argument('--from_saved', action='store_true')
    parser.add_argument('--ntk_model', default='', type=str)
    parser.add_argument('--data_root', default='./data', type=str)
    parser.add_argument('--N', default=1000, type=int)
    parser.add_argument('--nu', default=1.0, type=float)
    parser.add_argument('--kappa', default=1.0, type=float)
    parser.add_argument('--lambda_min', default=0.0, type=float)
    # Optimizer
    parser.add_argument('--sched', default='constant', type=str)
    parser.add_argument('--lr', nargs='+', default=[1.0], type=float)
    parser.add_argument('--momentum', nargs='+', default=[0.0], type=float)
    parser.add_argument('--a', nargs='+', default=[1.0], type=float)
    parser.add_argument('--b', nargs='+', default=[1.0], type=float)
    # Training params
    parser.add_argument('--n_steps', default=10000, type=int)
    parser.add_argument('--batch_size', nargs='+', default=[10], type=int)
    parser.add_argument('--workers', default=8, type=int)
    # Whether to track diag err
    parser.add_argument('--track_diag_err', action='store_true')
    parser.add_argument('--track_freq', default=100, type=int)
    # Iteration type
    parser.add_argument('--aggr_type', default='prod', choices=['prod', 'zip'], type=str)
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
    # create aggregation
    if args.aggr_type == 'zip':
        n_iters = max(len(args.batch_size), len(args.lr), len(args.momentum))
        if len(args.batch_size) > 1:
            assert len(args.batch_size) == n_iters
        else:
            args.batch_size = args.batch_size * n_iters
        if len(args.lr) > 1:
            assert len(args.lr) == n_iters
        else:
            args.lr = args.lr * n_iters
        if len(args.momentum) > 1:
            assert len(args.momentum) == n_iters
        else:
            args.momentum = args.momentum  * n_iters
        args.a = args.a * n_iters
        args.b = args.b * n_iters

        aggregator = zip
    else:
        aggregator = itertools.product
        
    if args.from_saved:
        K, d_f = load_kernel_and_err(data_root=args.data_root, model=args.ntk_model, dataset=args.dataset)
    else:
        if args.dataset == 'synthetic':
            # generate data
            K, d_f = generate_synthetic_data(size=args.N, kappa=args.kappa, nu=args.nu, lambda_min=args.lambda_min)
        elif args.dataset == 'mnist':
            # generate data
            K, d_f = generate_mnist_ntk_data(size=args.N, data_root=args.data_root)
        elif args.dataset == 'cifar10':
            # generate data
            K, d_f = generate_cifar10_ntk_data(size=args.N, data_root=args.data_root)
    K, d_f = K.to(device), d_f.to(device)
    # get normalized spectrum
    lambda_f, U = torch.linalg.eigh(K)
    # get normalized spectrum
    lambda_f /= lambda_f.max()
    # covariances
    C = torch.diag(U.T @ torch.outer(d_f, d_f) @ U)
    # make experiment dir if needed
    os.makedirs(f'{args.save_dir}', exist_ok=True)

    loss_curves = {}
    diag_errs = {}

    for B, lr, momentum, a, b in aggregator(
        args.batch_size, args.lr, args.momentum, args.a, args.b
    ):
        params = {'batch_size' : B}
        if args.sched == 'constant':
            alpha_fn = lambda step: lr
            beta_fn  = lambda step: momentum
            params['lr'] = lr
            params['momentum'] = momentum
        elif args.sched == 'jacobi':
            alpha_fn = JacobiScheduleA(lr, a, b)
            beta_fn  = JacobiScheduleB(a, b)
            params['lr'] = lr
            params['a'] = a
            params['b'] = b
        # at the moment only constant batch_size is supported
        batch_fn = lambda step: B
        # make initial state
        state = {'C' : deepcopy(C), 'J': torch.zeros_like(C), 'P': torch.zeros_like(C)}
        print(format_params(params), flush=True)
        state, loss_curve, Cs = train_approx(
            args.n_steps, state, lambda_f, alpha_fn, beta_fn, batch_fn,  
            track_diag_err=args.track_diag_err, track_freq=args.track_freq
        )
        # update dict with loss curves
        loss_curves[tuple(params.values())] = loss_curve
        if args.track_diag_err:
            diag_errs[tuple(params.values())] = Cs

    print('Training completed!')
    # save data
    torch.save(loss_curves, f'{args.save_dir}/loss_curves.pth')
    if args.track_diag_err:
        torch.save(diag_errs, f'{args.save_dir}/diag_errs.pth')
