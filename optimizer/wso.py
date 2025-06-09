from .problem import Problem

import numpy as np
import math
import random
import os

from concurrent.futures import ProcessPoolExecutor, as_completed


# ----------------------------------
# Helper functions - worker handling
# ----------------------------------

_worker = None
_worker_dir = None

def evaluate_with_local_worker(solution, model_path, time_hrs, measured_df, dim, lb, ub):
    global _worker, _worker_dir

    if _worker is None:
        import shutil
        from .worker import EpanetWorker

        # Use Process ID as an unique identifier
        pid = os.getpid()
        base_dir = os.getcwd()
        _worker_dir = os.path.join(base_dir, f"tmp/worker_{pid}")
        
        os.makedirs(_worker_dir, exist_ok=True)

        _worker = EpanetWorker(_worker_dir, "../../" + model_path, time_hrs, measured_df, dim, lb, ub)

    result = _worker(solution)

    return result


# ------------------------------------
# White Shark Optimizer implementation
# ------------------------------------

# Main WSO mechanism
# - Directly connected with a given problem by taking evaluator, dimentionality and parameter ranges as input
# - We assume that parameters are integers / floats in form of numpy array
class Optimizer:

    def __init__(self, model_filepath: str, no_workers: int = 4):
        # Initialize hyperparameters - according to WSO paper
        self.p_min = 0.5
        self.p_max = 1.5
        self.tau = 4.125
        self.mu = 2 / abs(2 - self.tau - np.sqrt(self.tau ** 2 - 4 * self.tau))
        self.f_min = 0.07
        self.f_max = 0.75
        self.a0 = 6.25
        self.a1 = 100.0
        self.a2 = 0.0005

        # Other optimization parameters
        self.model_filepath = model_filepath
        self.no_workers = no_workers
    

    def optimize(self, problem: Problem, no_sharks: int = 10, steps: int = 10,
                 verbose: bool = False) -> tuple[np.ndarray, float]:
        ''' Performs WSO to find a solution that minimizes problem.evaluate() function values 
        
            Returns a pair of (best_solution, best_solution_eval)
        '''

        # Step 1 - Generate initial population with respect dimensionality
        # - W for shark positions
        # - v for shark velocities
        W = np.random.uniform(problem.lb, problem.ub, (no_sharks, problem.dim))
        v = np.zeros_like(W)      # zeros_like() automatically copies dimensionality of an array

        # Step 2 - Evaluate initial population fitness
        # - Iterates over population ranks (position of a single shark) and creates 1D fitness vector
        fitness = np.array([problem.evaluate(pos) for pos in W])
        fitness_min = np.min(fitness)
        W_best = W.copy()                   # 2D matrix
        W_gbest = W[np.argmin(fitness)]     # 1D vector

        # Verbose setup
        next_log = 0.1

        # Main WSO loop
        with ProcessPoolExecutor(max_workers=self.no_workers) as executor:
            for k in range(1, steps + 1):
                # Calculate adaptive parameters
                p1 = self.p_max + (self.p_max - self.p_min) * np.exp(-(4 * k / steps)**2)
                p2 = self.p_min + (self.p_max - self.p_min) * np.exp(-(4 * k / steps)**2)
                mv = 1 / (self.a0 + np.exp((steps / 2.0 - k) / self.a1))
                s_s = abs(1 - np.exp(-self.a2 * k / steps))

                # Step 3 - update shark velocities
                # - NOTE: Can be additionally vectorized
                nu = np.random.randint(0, no_sharks, no_sharks)
                for i in range(no_sharks):
                    c1 = random.random()
                    c2 = random.random()
                    v[i, :] = self.mu * (v[i, :] + p1 * c1 * (W_gbest - W[i, :]) + p2 * c2 * (W_best[nu[i], :] - W[i, :]))
                
                # Step 4 - update positions with wavy motion or random allocation
                f = self.f_min + (self.f_max - self.f_min) / (self.f_max + self.f_min)
                for i in range(no_sharks):
                    a = W[i, :] > problem.ub
                    b = W[i, :] < problem.lb
                    w0 = np.logical_xor(a, b)
                    if random.random() < mv:
                        W[i][w0] = problem.ub[w0] * a[w0] + problem.lb[w0] * b[w0]
                    else:
                        W[i, :] += v[i, :] / f

                # Step 5 - school movement update
                for i in range(no_sharks):
                    # TODO: Is this thing even correct?
                    if random.random() <= s_s:
                        D = np.abs(np.random.rand() * (W_gbest - W[i, :]))  
                        if i == 0:
                            sgn = np.sign(np.random.rand(problem.dim) - 0.5)
                            W[i, :] = W_gbest + np.random.rand(problem.dim) * D * sgn
                        else:
                            sgn = np.sign(np.random.rand(problem.dim) - 0.5)
                            tmp = W_gbest + np.random.rand(problem.dim) * D * sgn
                            W[i, :] = (W[i, :] + tmp) / (2 * random.random())

                # Step 6 - return the sharks to the original solution space
                W = np.clip(W, problem.lb, problem.ub)

                # Step 7 - evaluate and update best positions
                # - Multithreading enabled
                future_to_shark = {
                    executor.submit(
                        evaluate_with_local_worker,
                        W[i, :],
                        self.model_filepath,
                        problem.time_hrs,
                        problem.measured_df,
                        problem.dim,
                        problem.lb,
                        problem.ub
                    ): i for i in range(no_sharks)
                }

                for future in as_completed(future_to_shark):
                    i = future_to_shark[future]
                    try:
                        fit = future.result()
                        if fit < fitness[i]:
                            W_best[i, :] = W[i, :]
                            fitness[i] = fit
                        if fitness[i] < fitness_min:
                            fitness_min = fitness[i]
                            W_gbest = W_best[i].copy()
                    except Exception as exc:
                        print(f'Exception generated by shark {i}: {exc}')
                
                # Verbose logging
                if verbose:
                    progress = k / steps
                    if progress >= next_log or k == steps:
                        print(f"Progress: {int(progress*100)}% | Best fitness: {fitness_min:.4f}")
                        next_log += 0.1
        
        return W_gbest, fitness_min
