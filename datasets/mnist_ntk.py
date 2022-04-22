import torch 

from torchvision.datasets import MNIST


def NTK_2_layer(X, sigma_w=1, sigma_b=1):
    _, d_in = X.shape
    # 1st layer
    Sigma_1 = (sigma_w ** 2 / d_in) * X @ X.T + sigma_b ** 2  
    Theta_1 = Sigma_1
    # 2nd layer
    norm   = torch.sqrt(torch.diag(Sigma_1))
    # clamp for numerical stability
    angle  = torch.arccos(torch.clamp(Sigma_1 / (norm[:, None] * norm[None, :]), max=1.0))
    angle -= angle.diagonal()
    
    J_0 = (torch.pi - angle) / (2 * torch.pi)
    J_1 = (torch.sin(angle) + (torch.pi - angle) * torch.cos(angle)) / (2 * torch.pi)
    
    Sigma_2 = (sigma_w ** 2) * (Theta_1 * J_0 + J_1)
    Theta_2 = Sigma_2

    return Theta_2


def generate_mnist_ntk_data(size: int, data_root: str = './data'):
    # get dataset
    train_dataset = MNIST(root=data_root, train=True, download=True)
    # generate rand size samples
    rand_idx = torch.randperm(60000)[:size]
    # select samples
    X = train_dataset.data[rand_idx].reshape(-1, 28 * 28).to(torch.float32)
    y = train_dataset.targets[rand_idx].to(torch.float32)
    # normalize
    X_mean, X_std = X.mean(), X.std()
    X = (X - X_mean) / X_std
    # get NTK
    K = NTK_2_layer(X)
    # renormalize NTK
    mult_factor = size / torch.linalg.eigvalsh(K).max()
    K *= mult_factor
    # return data
    return K, y