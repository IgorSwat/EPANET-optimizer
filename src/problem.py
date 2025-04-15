import numpy as np

from abc import ABC, abstractmethod


# -----------------
# Problem interface
# -----------------

class Problem(ABC):

    def __init__(self, dim: int, lb: np.ndarray, ub: np.ndarray):
        super().__init__()

        self.dim = dim
        self.lb = lb
        self.ub = ub

    @abstractmethod
    def evaluate(self, solution: np.ndarray) -> float:
        pass