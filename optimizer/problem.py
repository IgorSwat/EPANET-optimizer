import numpy as np
import pandas as pd
import warnings

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

    def __init__(self, dim, lb, ub, model: epanet, time_hrs: int, measured_df: pd.DataFrame):
        super().__init__(dim, lb, ub)

        self.network = model
        self.time_hrs = time_hrs
        self.measured_df = measured_df
    
    @override
    def evaluate(self, solution):
        pipe_indices = self.network.getLinkPipeIndex()

        # Apply new roughness coefficients
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            self.network.setLinkRoughnessCoeff(pipe_indices, solution)

            # Configure simulation duration
            self.network.setTimeSimulationDuration(self.time_hrs * 3600)

            # Run hydraulics
            self.network.solveCompleteHydraulics()

            ts = self.network.getComputedTimeSeries()
        
            pressure_data = ts.Pressure[:-1]
            expected_rows = len(self.measured_df.index)

            if pressure_data.shape[0] != expected_rows:
                return float('inf')

            # Build DataFrame of simulated pressures
            node_ids = self.network.getNodeNameID()
            sim_df = pd.DataFrame(ts.Pressure[:-1], columns=node_ids, index=self.measured_df.index)

            # Select only measured nodes and align
            sim_sel = sim_df[self.measured_df.columns]

            # Compute Mean Squared Error
            diff = sim_sel.values - self.measured_df.values
            mse = float(np.mean(diff ** 2))

            return mse