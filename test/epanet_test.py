import numpy as np
import pandas as pd
import time

from epyt import epanet
from src.problem import EpanetProblem
from src.wso import Optimizer

def epanet_test(model_filepath: str) -> None:
    # Step 1 - load network from file
    network = epanet(model_filepath)

    # Step 2 - set simulation time duration
    time_hrs = 24                                           # One full day
    network.setTimeSimulationDuration(time_hrs * 3600)      # Needs to be in seconds

    # Step 3 - run a complete hydraulics simulation and obtain results
    network.solveCompleteHydraulics()
    results = network.getComputedTimeSeries()

    # Step 4 - save results into CSV files
    pd.DataFrame(results.Pressure).to_csv("pressure_24h.csv")
    pd.DataFrame(results.Flow).to_csv("flow_24h.csv")

    # Step 5 - close network model
    network.unload()


def epanet_wso_test(model_filepath: str) -> None:
    network = epanet(model_filepath)

    dim = network.getLinkPipeCount()
    lb = np.full(shape=(dim), fill_value=4)
    ub = np.full(shape=(dim), fill_value=20)

    problem = EpanetProblem(dim, lb, ub, network, 24)

    optimizer = Optimizer()
    no_sharks = 50
    steps = 100

    diameters_best, loss_best = optimizer.optimize(problem, no_sharks=no_sharks, steps=steps)
    print("Optimal fitness:", loss_best)
    print("Optimal solution:", diameters_best, end="\n\n")

    network.solveCompleteHydraulics()
    results = network.getComputedTimeSeries()
    pd.DataFrame(results.Pressure).to_csv("pressure_24h.csv")

    network.unload()