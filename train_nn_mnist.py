import os
import torch
import argparse
import itertools

from torchvision.datasets import MNIST
from sklearn.model_selection import train_test_split

from models.mlp import NTKTwoLayerMLP
from training import train


def parse_args():
    parser = argparse.ArgumentParser('Training of synthetic power law data', add_help=False)
    # Data params
    parser.add_argument('--data-root', default='./data', type=str)
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
    parser.add_argument('--train_size', default=60000, type=int)
    # Logging and evaluation params
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--log_frequency', default=1, type=int)
    parser.add_argument('--val_frequency', default=1, type=int)
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
    # get dataset 
    train_dataset = MNIST(root=args.data_root, train=True, download=True)
    val_dataset = MNIST(root=args.data_root, train=False, download=True)
    # get mnist mean and std
    train_inputs  = train_dataset.data.reshape(-1, 28 * 28).to(torch.float32)
    train_targets = train_dataset.targets.to(torch.float32)
    mean_, std_ = train_inputs.mean(), train_inputs.std()
    train_inputs = (train_inputs - mean_) / std_
    # extract inputs and targets
    train_inputs, _, train_targets, _ = train_test_split(
        train_inputs,
        train_targets,
        shuffle=True,
        stratify=train_dataset.targets,
        train_size=args.train_size
    )
    val_inputs  = val_dataset.data.reshape(-1, 28 * 28).to(torch.float32)
    val_targets = val_dataset.targets.to(torch.float32)
    val_inputs = (val_inputs - mean_) / std_
    # all to device
    train_inputs, train_targets, val_inputs, val_targets = \
        train_inputs.to(device), train_targets.to(device), val_inputs.to(device), val_targets.to(device)
    # make experiment dir if needed
    os.makedirs(f'{args.save_dir}', exist_ok=True)    

    histories = {}

    for B, lr, momentum in itertools.product(args.batch_size, args.lr, args.momentum):
        # init model (note that only 1 class since we predict single number)
        model = NTKTwoLayerMLP(
            in_dim=784, 
            hidden_dim=args.hidden_dim, 
            num_classes=1,
            activation='relu'
        ).to(device)

        params = {'batch_size' : B}
        # make schedule
        if args.sched == 'constant':
            params['lr'] = lr
            params['momentum'] = momentum
        # make optimizer
        if args.opt == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

        print(format_params(params))
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
