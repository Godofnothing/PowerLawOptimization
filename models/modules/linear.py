import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NTKLinear(nn.Module):
    
    def __init__(self, in_features, out_features, sigma_w=1.0, sigma_b=1.0, bias=True):
        super().__init__()
        # save params
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        # register weights
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty((out_features,)))
        else:
            self.bias = None
        # init parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=self.sigma_w)
        if self.bias is not None:
            nn.init.normal_(self.bias, mean=0.0, std=self.sigma_b)
        
    def forward(self, x: torch.Tensor):
        return F.linear(x, (1 / math.sqrt(self.in_features)) * self.weight, self.bias)
    