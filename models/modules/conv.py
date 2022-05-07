import math
import torch.nn as nn

from typing import Union
from torch import Tensor
from torch.nn.common_types import _size_2_t


class NTKConv2d(nn.Conv2d):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        sigma_w: float = 1.0,
        sigma_b: float = 1.0, 
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ) -> None:
        # init std for weight and bias
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        # base init
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype
        )

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=self.sigma_w)
        if self.bias is not None:
            nn.init.normal_(self.bias, mean=0.0, std=self.sigma_b)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(
            input, (1 / math.sqrt(self.in_channels * self.kernel_size[0] * self.kernel_size[1])) * self.weight, self.bias)
    