import torch
from abc import abstractmethod


__all__ = ['Optimizer']


class Optimizer:

    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def step(self, step: int, state: dict, K: torch.Tensor, batch_idx: torch.Tensor):
        pass
