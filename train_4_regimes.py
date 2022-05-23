import os
import torch
import argparse
import itertools
# mute warnings
import warnings
warnings.simplefilter('ignore')

from copy import deepcopy
from functorch import make_functional, vmap, jacrev

from training import train_linearized, train_spectral_diagonal, train
from datasets import get_image_dataset, get_uci_data, get_sklearn_data
from models.mlp import NTKTwoLayerMLP
from optim import SGD


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
    parser = argparse.ArgumentParser('Training in 4 regimes', add_help=False)
    # Data params
    parser.add_argument('--dataset', default='mnist', 
                        choices=['mnist', 'cifar10', 'olivetti_faces', 'digits', 'bike_sharing', 'sgemm_product'], type=str)
    parser.add_argument('--data_root', default='./data', type=str)
    parser.add_argument('--N', default=1000, type=int)
    # Model
    parser.add_argument('--hidden_dim', default=1000, type=int)
    # Optimizer
    parser.add_argument('--lr', nargs='+', default=[1.0], type=float)
    parser.add_argument('--momentum', nargs='+', default=[0.0], type=float)
    # Training params
    parser.add_argument('--n_steps', default=10000, type=int)
    parser.add_argument('--batch_size', nargs='+', default=[10], type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--seed', default=42, type=int)
    # Number of runs
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

        aggregator = zip
    else:
        aggregator = itertools.product  

    if args.dataset in ['olivetti_faces', 'digits']:
        train_inputs, train_targets, *dummy = \
            get_sklearn_data(dataset=args.dataset, N=args.N, data_root=args.data_root, seed=args.seed, device=device)
    elif args.dataset in ['bike_sharing', 'sgemm_product']:
         train_inputs, train_targets, *dummy = \
             get_uci_data(dataset=args.dataset, N=args.N, data_root=args.data_root, seed=args.seed, device=device)
    else:
        train_inputs, train_targets, *dummy = \
            get_image_dataset(dataset=args.dataset, N=args.N, data_root=args.data_root, seed=args.seed, device=device)
    # make experiment dir if needed
    os.makedirs(f'{args.save_dir}', exist_ok=True)

    loss_curves = {}

    # init model note that only 1 class since we predict single number)
    model = NTKTwoLayerMLP(
        in_dim=train_inputs.shape[-1], 
        hidden_dim=args.hidden_dim, 
        num_classes=1,
        activation='relu'
    ).to(device)
    # functional version of the model
    fmodel, params = make_functional(model)
    # get empirical NTK
    K_emp = empirical_ntk(fmodel, params, train_inputs, train_inputs)[..., 0, 0]
    with torch.no_grad():
        eigv, U = torch.linalg.eigh(K_emp)
        # sort in descending order
        eigv, U = torch.flip(eigv, dims=(0,)), torch.flip(U, dims=(1,))
    # scale NTK
    K_emp *= (args.N / eigv.max().item())
    eigv  /= eigv.max()
    # get initial prediction
    with torch.no_grad():
        d_f = model(train_inputs).view(-1) - train_targets
    # get diag elements of covariance (for Mean Field)
    c_diag = (d_f @ U) ** 2

    data = {'c_diag': c_diag.cpu(), 'eigv' : eigv.cpu()}

    # train loop
    for B, lr, momentum in aggregator(args.batch_size, args.lr, args.momentum):
        params = {'batch_size' : B, 'lr' : lr, 'momentum': momentum}
        loss_curves_params = {}
        # set training params
        batch_fn = lambda step: B
        alpha_fn = lambda step: lr
        beta_fn  = lambda step: momentum
        # log step
        print(format_params(params), flush=True)
        ### Train Neural Network

        # copy model
        model_copy = deepcopy(model)

        nn_optimizer = torch.optim.SGD(model_copy.parameters(), lr=lr, momentum=momentum)
        loss_curves_params["mlp"] = train(
            args.n_steps, 
            model_copy, 
            nn_optimizer, 
            train_inputs, 
            train_targets,
            B,
            val_inputs=None, 
            val_targets=None
        )['train/loss']
        ### Train Mean Field

        state = {'C' : deepcopy(c_diag), 'J': torch.zeros_like(c_diag), 'P': torch.zeros_like(c_diag)}
        state, loss_curve, _ = train_spectral_diagonal(
            args.n_steps, state, eigv, alpha_fn, beta_fn, batch_fn,  
            tau=1.0, track_diag_err=False, 
        )
        loss_curves_params["mean_field"] = loss_curve

        ### Train Gauss

        state = {'C' : deepcopy(c_diag), 'J': torch.zeros_like(c_diag), 'P': torch.zeros_like(c_diag)}
        state, loss_curve, _ = train_spectral_diagonal(
            args.n_steps, state, eigv, alpha_fn, beta_fn, batch_fn,  
            tau=-1.0, track_diag_err=False, 
        )
        loss_curves_params["gauss"] = loss_curve

        ### Train Linearized
        ln_optimizer = SGD(alpha_fn=alpha_fn, beta_fn=beta_fn)
        # make state again
        state = {'d_f' : d_f.clone(), 'p_f': torch.zeros(args.N, dtype=torch.float32, device=device)}

        state, loss_curve, _ = train_linearized(
            ln_optimizer, state, K_emp, n_steps=args.n_steps, batch_size=B, track_diag_err=False)
        loss_curves_params["linearized"] = loss_curve

        loss_curves[(B, lr, momentum)] = loss_curves_params

    print('Training completed!')
    # save data
    torch.save(loss_curves, f'{args.save_dir}/loss_curves_4_regimes.pth')
    torch.save(data, f'{args.save_dir}/data.pth')
