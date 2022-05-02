import os
import torch
import argparse
import itertools

from training import train_linearized
from datasets import generate_synthetic_data, generate_mnist_ntk_data
from optim import SGD, Adam
from schedules import JacobiScheduleA, JacobiScheduleB


def parse_args():
    parser = argparse.ArgumentParser('Training of synthetic power law data', add_help=False)
    # Data params
    parser.add_argument('--dataset', default='synthetic', choices=['synthetic', 'mnist'], type=str)
    parser.add_argument('--data-root', default='./data', type=str)
    parser.add_argument('--N', default=1000, type=int)
    parser.add_argument('--nu', default=1.0, type=float)
    parser.add_argument('--kappa', default=1.0, type=float)
    parser.add_argument('--lambda_min', default=0.0, type=float)
    # Optimizer
    parser.add_argument('--opt', default='sgd', type=str)
    parser.add_argument('--sched', default='constant', type=str)
    parser.add_argument('--lr', nargs='+', default=[1.0], type=float)
    parser.add_argument('--momentum', nargs='+', default=[0.0], type=float)
    parser.add_argument('--a', nargs='+', default=[1.0], type=float)
    parser.add_argument('--b', nargs='+', default=[1.0], type=float)
    parser.add_argument('--b1', nargs='+', default=0.9, type=float)
    parser.add_argument('--b2', nargs='+', default=0.999, type=float)
    # Training params
    parser.add_argument('--n_steps', default=10000, type=int)
    parser.add_argument('--batch_size', nargs='+', default=[10], type=int)
    parser.add_argument('--workers', default=8, type=int)
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

    if args.dataset == 'synthetic':
        # generate data
        K, d_f = generate_synthetic_data(size=args.N, kappa=args.kappa, nu=args.nu, lambda_min=args.lambda_min)
    elif args.dataset == 'mnist':
        # generate data
        K, d_f = generate_mnist_ntk_data(size=args.N, data_root=args.data_root)
    K, d_f = K.to(device), d_f.to(device)
    # make experiment dir if needed
    os.makedirs(f'{args.save_dir}', exist_ok=True)

    loss_curves = {}

    for B, lr, momentum, a, b in aggregator(
        args.batch_size, args.lr, args.momentum, args.a, args.b
    ):
        params = {'batch_size' : B}
        # make schedule
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
        # make optimizer
        if args.opt == 'sgd':
            optimizer = SGD(alpha_fn=alpha_fn, beta_fn=beta_fn)
        elif args.opt == 'adam':
            optimizer = Adam(alpha_fn=alpha_fn, b1=args.b1, b2=args.b2)
        
         # init momentum
        p_f = torch.zeros(args.N, dtype=torch.float32, device=device)
        # init state
        state = {'d_f' : d_f.clone(), 'p_f': p_f}
        # for adam state has 2nd moment as well
        if args.opt == 'adam':
            v_f = torch.zeros(args.N, dtype=torch.float32, device=device) 
            state['v_f'] = v_f

        print(format_params(params))
        state, loss_curve = train_linearized(optimizer, state, K, n_steps=args.n_steps, batch_size=B)
        # update dict with loss curves
        loss_curves[tuple(params.values())] = loss_curve

    print('Training completed!')
    # save data
    torch.save(loss_curves, f'{args.save_dir}/loss_curves.pth')
