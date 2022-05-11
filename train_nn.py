import os
import torch
import argparse
import itertools

from torchvision.datasets import MNIST, CIFAR10
from sklearn.model_selection import train_test_split
from functorch import make_functional, vmap, jacrev

from models.mlp import NTKTwoLayerMLP
from training import train


def empirical_ntk(fnet_single, params, x1, x2):
    # Compute J(x1)
    jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
    jac1 = [j.flatten(2) for j in jac1]
    
    # Compute J(x2)
    jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
    jac2 = [j.flatten(2) for j in jac2]
    
    # Compute J(x1) @ J(x2).T
    result = torch.stack([torch.einsum('Naf,Mbf->NMab', j1, j2) for j1, j2 in zip(jac1, jac2)])
    result = result.sum(0)
    return result


def parse_args():
    parser = argparse.ArgumentParser('Training of synthetic power law data', add_help=False)
    # Data params
    parser.add_argument('--data-root', default='./data', type=str)
    parser.add_argument('--dataset', default='mnist', type=str)
    # Model
    parser.add_argument('--hidden_dim', default=1000, type=int)
    # Optimizer
    parser.add_argument('--opt', default='sgd', type=str)
    parser.add_argument('--sched', default='constant', type=str)
    parser.add_argument('--lr', nargs='+', default=[1.0], type=float)
    parser.add_argument('--momentum', nargs='+', default=[0.0], type=float)
    # Training params
    parser.add_argument('--n_steps', default=10000, type=int)
    parser.add_argument('--batch_size', nargs='+', default=[10], type=int)
    parser.add_argument('--N', default=1000, type=int)
    parser.add_argument('--lr_scale', default=None, type=float)
    # Logging and evaluation params
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--log_frequency', default=1, type=int)
    parser.add_argument('--val_frequency', default=1, type=int)
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
            args.momentum = args.momentum * n_iters
        aggregator = zip
    else:
        aggregator = itertools.product
    # get dataset 
    DATASET = MNIST if args.dataset == 'mnist' else CIFAR10
    train_dataset = DATASET(root=args.data_root, train=True, download=True)
    val_dataset = DATASET(root=args.data_root, train=False, download=True)
    if args.dataset == 'mnist':
        train_inputs  = train_dataset.data.reshape(-1, 28 * 28).to(torch.float32)
        train_targets = train_dataset.targets.to(torch.float32)
    elif args.dataset == 'cifar10':
        train_inputs  = torch.tensor(train_dataset.data.reshape(-1, 3 * 32 * 32), dtype=torch.float32)
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
        train_size=args.N
    )
    if args.dataset == 'mnist':
        val_inputs  = val_dataset.data.reshape(-1, 28 * 28).to(torch.float32)
        val_targets = val_dataset.targets.to(torch.float32)
    elif args.dataset == 'cifar10':
        val_inputs  = torch.tensor(val_dataset.data.reshape(-1, 3 * 32 * 32), dtype=torch.float32)
        val_targets = torch.tensor(val_dataset.targets, dtype=torch.float32)
    val_inputs = (val_inputs - inputs_mean) / inputs_std
    # normalize targets as well
    train_targets = (train_targets - targets_mean) / targets_std
    val_targets   = (val_targets   - targets_mean) / targets_std
    # all to device
    train_inputs, train_targets, val_inputs, val_targets = \
        train_inputs.to(device), train_targets.to(device), val_inputs.to(device), val_targets.to(device)
    # make experiment dir if needed
    os.makedirs(f'{args.save_dir}', exist_ok=True)  

    histories = {}

    for B, lr, momentum in aggregator(args.batch_size, args.lr, args.momentum):
        # init model (note that only 1 class since we predict single number)
        model = NTKTwoLayerMLP(
            in_dim=784 if args.dataset == 'mnist' else 3072, 
            hidden_dim=args.hidden_dim, 
            num_classes=1,
            activation='relu'
        ).to(device)
        # functional version of the model
        fmodel, params = make_functional(model)
        if not args.lr_scale:
            # get learning rate scaling factor
            K_emp = empirical_ntk(fmodel, params, train_inputs, train_inputs)[..., 0, 0]
            with torch.no_grad():
                max_eigv = torch.linalg.eigvalsh(K_emp).max().item()
            lr_scale = args.N / max_eigv
            # free memory
            del K_emp
            torch.cuda.empty_cache()
        else:
            lr_scale = args.lr_scale

        params = {'batch_size' : B}
        # make schedule
        if args.sched == 'constant':
            params['lr'] = lr
            params['momentum'] = momentum
        # make optimizer
        if args.opt == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr_scale * lr, momentum=momentum)

        print(format_params(params), flush=True)
        history = train(
            args.n_steps, 
            model, optimizer, 
            inputs=train_inputs, 
            targets=train_targets,
            batch_size=B,
            val_inputs=val_inputs,
            val_targets=val_targets,
            val_frequency=args.val_frequency,
            log_frequency=args.log_frequency,
            verbose=args.verbose
        )
        # update dict with loss curves
        histories[tuple(params.values())] = history

    print('Training completed!')
    # save data
    torch.save(histories, f'{args.save_dir}/histories.pth')
