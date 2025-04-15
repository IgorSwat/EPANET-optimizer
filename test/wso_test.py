import numpy as np
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
        return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    

# --------
# WSO test
# --------

def wso_test() -> None:
    no_sharks = 100  # Population size
    itemax = 500      # Number of iterations
    dim = 10           # Dimension of the problem

    # Define WSO
    wso = Optimizer()

    # random.seed(410375)
    # np.random.seed(410375)

    # Define test problems
    test_problems = [
        RosenbrockProblem(dim, np.full(shape=(dim), fill_value=-10), np.full(shape=(dim), fill_value=10)),
        RastriginProblem(dim, np.full(shape=(dim), fill_value=-5.12), np.full(shape=(dim), fill_value=5.12))
    ]

    no_tests = 10
    total_error = 0.0
    for i in range(no_tests):
        for problem in test_problems:
            print(f"{problem.__class__.__name__} test:")
            print("-" * 20)
            gbest, fmin = wso.optimize(problem, no_sharks=no_sharks, steps=itemax)
            print("Optimal fitness:", fmin)
            print("Optimal solution:", gbest, end="\n\n")
            total_error += fmin
    print("Average error:", total_error / no_tests)