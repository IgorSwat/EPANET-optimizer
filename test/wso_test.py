import numpy as np
import pyswarms as ps
import random

from src.wso import Optimizer
from src.problem import Problem
from typing import override


# ---------------
# Sample problems
# ---------------

class RosenbrockProblem(Problem):
    def __init__(self, dim, lb, ub):
        super().__init__(dim, lb, ub)

    @override
    def evaluate(self, x):
        return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    

class RastriginProblem(Problem):
    def __init__(self, dim, lb, ub):
        super().__init__(dim, lb, ub)

    @override
    def evaluate(self, x):
        return 10 * len(x) + np.sum((x-2)**2 - 10 * np.cos(2 * np.pi * (x-2)))
    

# --------
# WSO test
# --------

def wso_test() -> None:
    no_sharks = 30  # Population size
    itemax = 100      # Number of iterations
    dim = 5           # Dimension of the problem

    # Define WSO
    wso = Optimizer()

    # random.seed(410375)
    # np.random.seed(410375)

    # Define test problems
    test_problems = [
        RosenbrockProblem(dim, np.full(shape=(dim), fill_value=-10), np.full(shape=(dim), fill_value=10)),
        RastriginProblem(dim, np.full(shape=(dim), fill_value=-5.12), np.full(shape=(dim), fill_value=5.12))
    ]

    no_tests = 100
    errors = [0.0, 0.0]
    verbose = False

    for i in range(no_tests):
        for j, problem in enumerate(test_problems):
            if verbose:
                print(f"{problem.__class__.__name__} test:")
                print("-" * 20)
            gbest, fmin = wso.optimize(problem, no_sharks=no_sharks, steps=itemax)
            if verbose:
                print("Optimal fitness:", fmin)
                print("Optimal solution:", gbest, end="\n\n")
            errors[j] += fmin
    print("Rosenbrock average error:", errors[0] / no_tests)
    print("Rastrigin average error:", errors[1] / no_tests)


# ---------------
# WSO vs PSO test
# ---------------

def wso_vs_pso_test() -> None:
    dimensions = 2

    def rastrigin(x):
        return 10 * dimensions + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)
    
    no_trials = 1000
    pso_avg_loss, wso_avg_loss = 0.0, 0.0
    
    for i in range(no_trials):
        # ===== PSO algorithm =====
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        optimizer = ps.single.GlobalBestPSO(n_particles=30, dimensions=dimensions, options=options)
        best_cost, best_pos = optimizer.optimize(rastrigin, iters=100, verbose=False)
        pso_avg_loss += best_cost

        # ===== WSO algorithm =====
        lower_bounds = np.full(dimensions, -5.12)
        upper_bounds = np.full(dimensions, 5.12)
        wso = Optimizer()
        best_pos, best_cost = wso.optimize(
            RastriginProblem(dimensions, lower_bounds, upper_bounds),
            no_sharks=30, steps=100
        )
        wso_avg_loss += best_cost

    pso_avg_loss /= no_trials
    wso_avg_loss /= no_trials

    print("PSO average loss ({}D):".format(dimensions), pso_avg_loss)
    print("WSO average loss ({}D):".format(dimensions), wso_avg_loss)