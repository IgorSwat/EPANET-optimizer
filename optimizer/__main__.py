from .finput import read_pressure_timeseries
from .problem import EpanetProblem
from .wso import Optimizer

import numpy as np
import sys
import yaml

from epyt import epanet


if __name__ == "__main__":
    # Step 1 - load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Step 2 - load EPANET model
    # - Depending on the argument value we either load an example small network, or real bigger network
    try:
        model_filepath = config["paths"]["models"]["example"] if "example" in sys.argv else config["paths"]["models"]["target"]

        network = epanet(model_filepath)
    except Exception as e:
        print(f"ERROR: {e}")
        exit(1)
    
    print(f"[ Succesfully loaded model from {model_filepath} ]")

    # Step 3 - Determine roughness value bounds
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

    # Step 4 - create an objective function
    # - We use pressure values from P.txt and we define loss as MSE with respect to these values
    df_pressure = read_pressure_timeseries(config["paths"]["data"]["pressure"])

    print(f"[ Succesfully loaded data from {config["paths"]["data"]["pressure"]} ]")

    # Now create a Problem instance
    problem = EpanetProblem(dim=n_pipes,
                            lb=lb, ub=ub,
                            model=network,
                            time_hrs=24,
                            measured_df=df_pressure)

    # Test on some sample data
    # print("MSE:", problem.evaluate(ub))
    # print(df_pressure.index)

    # Step 5 - run WSO
    no_sharks = int(sys.argv[2])
    steps = int(sys.argv[3])

    optimizer = Optimizer()

    # Run the optimization process
    print(f"[ Optimization started (no_sharks={no_sharks}, steps={steps})]")
    roughness_best, loss_best = optimizer.optimize(problem, no_sharks=no_sharks, steps=steps, verbose=True)
    print("[ Optimization finished ]")

    print("\nOptimal fitness:", loss_best)
    print("Optimal solution:", [f"{x:.3f}" for x in roughness_best], end="\n\n")

    # Step 7 - save and quit EPANET model
    network.setLinkRoughnessCoeff(pipe_indices, roughness_best)
    network.saveInputFile(config["paths"]["models"]["output"])

    network.unload()