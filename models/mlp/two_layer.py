import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import NTKLinear


class NTKTwoLayerMLP(nn.Module):
    
    def __init__(self, in_dim: int = 784, hidden_dim: int = 1000, num_classes: int = 10, activation='relu'):
        super().__init__()
        self.fc1 = NTKLinear(in_dim, hidden_dim)
        self.fc2 = NTKLinear(hidden_dim, num_classes, bias=False)
        self.act = getattr(F, activation)
        
    def forward(self, x: torch.Tensor):
        '''
        Args: 
            x - torch.Tensor
                flattened tensor of images if shape (batch_size, image_h * image_w)
        Returns:
            x - torch.Tensor
                logits of predicted classes of shape (batch_size, num_classes)
        '''
        # flatten input data (if needed)
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
        # compute logits
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
    