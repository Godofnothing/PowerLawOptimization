from abc import abstractmethod


__all__ = ['Schedule']


class Schedule:

    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def __call__(self, step: int) -> float:
        pass
    