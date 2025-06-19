from .finput import read_pressure_timeseries
from .problem import EpanetProblem
from .wso import Optimizer

import argparse
import glob
import numpy as np
import os
import shutil
import sys
import yaml

from epyt import epanet


if __name__ == "__main__":
    # Step 1 - load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Step 2 - parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--sharks", type=int, default=50, help="Shark population size")
    parser.add_argument("--steps", type=int, default=10, help="Number of optimer iterations")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--verbose", action="store_true", help="Enable additional logging")
    parser.add_argument("--logging_freq", type=int, default=10, help="Interval between logs")
    args = parser.parse_args()

    # Step 3 - load EPANET model
    # - Depending on the argument value we either load an example small network, or real bigger network
    try:
        # model_filepath = config["paths"]["models"]["example"]
        model_filepath = config["paths"]["models"]["target"]
        tmp_filepath = config["paths"]["other"]["tmp"]

        network = epanet(model_filepath)
    except Exception as e:
        print(f"ERROR: {e}")
        exit(1)
    
    print(f"[ Succesfully loaded model from {model_filepath} ]")

    # Step 4 - Determine roughness value bounds
    # - By default, we use min and max from already existing roughness values from the model as bounds
    pipe_indices = network.getLinkPipeIndex()
    n_pipes = len(pipe_indices)        # Number of pipes (= dimensionality)
    
    roughness_values = np.array([network.getLinkRoughnessCoeff(i) for i in pipe_indices])
    min_rough = max(1e-3, roughness_values.min() / 2)       # Increase bound by 2 times
    max_rough = roughness_values.max() * 2

    print("\nInitial solution:", roughness_values, end="\n\n")

    # Create bounds vectors
    lb = np.full(shape=(n_pipes,), fill_value=min_rough)
    ub = np.full(shape=(n_pipes,), fill_value=max_rough)

    # Step 5 - create an objective function
    # - We use pressure values from P.txt and we define loss as MSE with respect to these values
    df_pressure = read_pressure_timeseries(config["paths"]["data"]["pressure"])

    print(f"[ Succesfully loaded data from {config["paths"]["data"]["pressure"]} ]")

    # Now create a Problem instance
    problem = EpanetProblem(dim=n_pipes,
                            lb=lb, ub=ub,
                            model=network,
                            time_hrs=24,
                            measured_df=df_pressure)

    # Step 6 - run WSO
    no_sharks = args.sharks
    no_steps = args.steps
    no_workers = args.workers
    print(args.verbose)

    # NOTE: You can adjust number of parallel workers
    optimizer = Optimizer(model_filepath, tmp_filepath, no_workers=no_workers)

    # Run the optimization process
    print(f"[ Optimization started (no_sharks={no_sharks}, steps={no_steps})]")
    roughness_best, loss_best = optimizer.optimize(problem, no_sharks=no_sharks, steps=no_steps, 
                                                   verbose=args.verbose, logging_freq=args.logging_freq)
    print("[ Optimization finished ]")

    print("\nOptimal fitness:", loss_best)
    print("Optimal solution:", [f"{x:.3f}" for x in roughness_best], end="\n\n")

    # Step 7 - remove temporary directions
    for path in glob.glob("tmp/worker_*"):
        if os.path.isdir(path):
            shutil.rmtree(path)

    # Step 8 - save optimization history to a file
    with open(config["paths"]["output"]["optim_history"], "w") as f:
        for item in optimizer.best_fitness_history:
            f.write(f"{item}\n")

    # Step 9 - save and quit EPANET model
    network.setLinkRoughnessCoeff(pipe_indices, roughness_best)
    network.saveInputFile(config["paths"]["output"]["model"])

    network.unload()