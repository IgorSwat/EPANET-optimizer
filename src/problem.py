import numpy as np

from abc import ABC, abstractmethod
from epyt import epanet
from typing import override


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


# -------------------------
# EPANET simulation problem
# -------------------------

class EpanetProblem(Problem):

    def __init__(self, dim, lb, ub, model: epanet, time_hrs: int):
        super().__init__(dim, lb, ub)

        self.network = model
        self.time_hrs = time_hrs
    
    @override
    def evaluate(self, solution):
        ''' NOTE: This is a temporary version 

            In this version we use pipe diameter as parameters and pressure at last junction as objective fitness function
        '''

        # Update pipe diameters
        pipe_indices = self.network.getLinkPipeIndex()
        self.network.setLinkDiameter(pipe_indices, solution.astype(int))

        self.network.setTimeSimulationDuration(self.time_hrs * 3600)

        # Run complete simulation
        self.network.solveCompleteHydraulics()
        results = self.network.getComputedTimeSeries().Pressure

        # Some dumb heuristic
        # - Maximizes mean pressure at 7th junction
        mean = np.mean(results[:, -1])

        return -mean
