from .base import Schedule


__all__ = ['JacobiScheduleA', 'JacobiScheduleB']


class JacobiScheduleA(Schedule):

    def __init__(self, lr: float, a: float=1.0, b: float = 1.0) -> None:
        super().__init__()
        self.lr = lr
        self.a = a
        self.b = b

    def __call__(self, step: int) -> float:
        n, lr, a, b = step, self.lr, self.a, self.b
        return lr * (2 * n + a + b + 1) * (2 * n + a + b + 2) / (2 * (n + a + 1) * (n + a + b + 1))


class JacobiScheduleB(Schedule):

    def __init__(self, a: float=1.0, b: float = 1.0) -> None:
        super().__init__()
        self.a = a
        self.b = b

    def __call__(self, step: int) -> float:
        n, a, b = step, self.a, self.b
        return n * (n + b) * (2 * n + a + b + 2) / ((n + a + 1) * (n + a + b + 1) * (2 * n + a + b))
