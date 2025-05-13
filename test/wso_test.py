import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyswarms as ps
import random
import time

from optimizer.wso import Optimizer
from optimizer.problem import Problem
from typing import override
from tqdm import tqdm


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


# ----------------
# Helper functions
# ----------------

def make_pso_callable(problem):
    """
    Zwraca funkcję f(X), gdzie X ma kształt (n_particles, dim),
    a wynik to wektor kosztów długości n_particles.
    """
    def _f(X: np.ndarray) -> np.ndarray:
        return np.array([problem.evaluate(x) for x in X])
    return _f

def run_single_trial(dimensions: int,
                     problem_name: str,
                     pso_opts: dict,
                     wso_opts: dict,
                     bounds: tuple,
                     seed: int) -> dict:
    """
    Jeden trial PSO vs WSO dla zadanego problemu.
    problem_name: 'rastrigin' lub 'rosenbrock'
    """
    np.random.seed(seed)
    # Tworzymy instancję problemu
    if problem_name.lower() == 'rastrigin':
        problem = RastriginProblem(dimensions, bounds[0], bounds[1])
    elif problem_name.lower() == 'rosenbrock':
        problem = RosenbrockProblem(dimensions, bounds[0], bounds[1])
    else:
        raise ValueError("Nieznany problem: " + problem_name)

    # --- PSO ---
    pso = ps.single.GlobalBestPSO(
        n_particles=pso_opts['n_particles'],
        dimensions=dimensions,
        options=pso_opts['options'],
        bounds=bounds
    )
    f_pso = make_pso_callable(problem)
    t0 = time.time()
    best_cost_pso, best_pos_pso = pso.optimize(f_pso,
                                               iters=pso_opts['iters'],
                                               verbose=False)
    t_pso = time.time() - t0

    # --- WSO ---
    wso = Optimizer()
    t0 = time.time()
    best_pos_wso, best_cost_wso = wso.optimize(
        problem,
        no_sharks=wso_opts['no_sharks'],
        steps=wso_opts['steps']
    )
    t_wso = time.time() - t0

    return {
        'problem': problem_name,
        'pso_cost': best_cost_pso,
        'wso_cost': best_cost_wso,
        'pso_time': t_pso,
        'wso_time': t_wso
    }

def benchmark(dimensions: int = 2,
              no_trials: int = 100,
              problems: list = None,
              pso_opts: dict = None,
              wso_opts: dict = None):
    """
    Uruchamia benchmark dla listy problemów.
    problems: lista stringów, np. ['rastrigin','rosenbrock']
    """
    if problems is None:
        problems = ['rastrigin', 'rosenbrock']
    if pso_opts is None:
        pso_opts = {
            'n_particles': 30,
            'options': {'c1': 0.5, 'c2': 0.3, 'w': 0.9},
            'iters': 100
        }
    if wso_opts is None:
        wso_opts = {
            'no_sharks': 30,
            'steps': 100
        }

    lower_bounds = np.full(dimensions, -5.12)
    upper_bounds = np.full(dimensions,  5.12)
    bounds = (lower_bounds, upper_bounds)

    records = []
    for problem in problems:
        for i in tqdm(range(no_trials), desc=f"Benchmark {problem}"):
            res = run_single_trial(dimensions, problem, pso_opts, wso_opts, bounds, seed=i)
            records.append({
                'problem': problem,
                'trial': i,
                'algorithm': 'PSO',
                'cost': res['pso_cost'],
                'time_s': res['pso_time']
            })
            records.append({
                'problem': problem,
                'trial': i,
                'algorithm': 'WSO',
                'cost': res['wso_cost'],
                'time_s': res['wso_time']
            })

    return pd.DataFrame(records)

def plot_comparison_bars(df):
    """
    Dla każdego problemu rysuje:
     1) słupek średniego kosztu z błędem = odch.std
     2) słupek średniego czasu z błędem = odch.std
    df musi mieć kolumny ['problem','algorithm','cost','time_s'].
    """
    for problem, subdf in df.groupby('problem'):
        # Statystyki kosztu
        cost_stats = subdf.groupby('algorithm')['cost'].agg(['mean','std'])
        algos = cost_stats.index.tolist()
        means = cost_stats['mean'].values
        errs  = cost_stats['std'].values

        x = np.arange(len(algos))
        plt.figure()
        plt.bar(x, means, yerr=errs, capsize=5)
        plt.xticks(x, algos)
        plt.ylabel("Średni koszt")
        plt.title(f"Średni koszt końcowy ({problem})")
        plt.show()

        # Statystyki czasu
        time_stats = subdf.groupby('algorithm')['time_s'].agg(['mean','std'])
        means_t = time_stats['mean'].values
        errs_t  = time_stats['std'].values

        plt.figure()
        plt.bar(x, means_t, yerr=errs_t, capsize=5)
        plt.xticks(x, algos)
        plt.ylabel("Średni czas [s]")
        plt.title(f"Średni czas wykonania ({problem})")
        plt.show()


# --------
# WSO test
# --------

def wso_test() -> None:
    no_sharks = 100  # Population size
    itemax = 1000      # Number of iterations
    dim = 100           # Dimension of the problem

    # Define WSO
    wso = Optimizer()

    # random.seed(410375)
    # np.random.seed(410375)

    # Define test problems
    test_problems = [
        RosenbrockProblem(dim, np.full(shape=(dim), fill_value=-10), np.full(shape=(dim), fill_value=10)),
        RastriginProblem(dim, np.full(shape=(dim), fill_value=-5.12), np.full(shape=(dim), fill_value=5.12))
    ]

    no_tests = 1
    errors = [0.0, 0.0]
    verbose = True

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
    DIM = 2
    TRIALS = 200

    df = benchmark(dimensions=DIM, no_trials=TRIALS)
    plot_comparison_bars(df)