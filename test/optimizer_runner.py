import argparse
import time
import numpy as np
from optimizer.wso import Optimizer
from optimizer.problem import Problem
from typing_extensions import override
from test.wso_test import RosenbrockProblem, RastriginProblem, BentCigarProblem, LevyProblem, SchwefelProblem, ZakharovProblem
# ------------------------------------------------------------------
# ShiftedProblem Wrapper and Factory
# ------------------------------------------------------------------
class ShiftedProblem(Problem):
    def __init__(self, base_problem: Problem, shifted_lb: float = -80, shifted_ub: float = 80, target: np.ndarray = None):
        """
        Wraps a base problem so that its optimum is mapped to a random target point in the shifted
        domain. If no target is provided, it is generated uniformly at random from (shifted_lb, shifted_ub)^d.
        
        The transformation is:
              x_original = x - (target - base_problem.optimal_point())
        so that when x equals the target, x_original is the base problem’s optimum.
        """
        self.base_problem = base_problem
        self.dim = base_problem.dim
        if target is None:
            target = np.random.uniform(shifted_lb, shifted_ub, self.dim)
        self.target = target
        self.shift = self.target - self.base_problem.optimal_point()
        self.lb = np.full(self.dim, shifted_lb)
        self.ub = np.full(self.dim, shifted_ub)
    
    def evaluate(self, x):
        # Transform from shifted coordinates back to the original domain.
        x_original = x - self.shift
        return self.base_problem.evaluate(x_original)
    
    def optimal_point(self):
        # The optimum in the shifted domain is now the random target.
        return self.target

def create_shifted_problem(problem_cls, d: int, base_lb: float, base_ub: float,
                             shifted_lb: float = -80, shifted_ub: float = 80, target: np.ndarray = None) -> ShiftedProblem:
    """
    Constructs a base problem using the provided class with base bounds and wraps it as a ShiftedProblem.
    The search space for the optimizer becomes [shifted_lb, shifted_ub]^d, and the optimum is placed
    at a randomly selected target (unless one is explicitly provided).
    """
    lb = np.full(d, base_lb)
    ub = np.full(d, base_ub)
    base_problem = problem_cls(d, lb, ub)
    return ShiftedProblem(base_problem, shifted_lb, shifted_ub, target)


class OptimizerRunner:
    @staticmethod
    def runs(x, n=100, max_iterations=100, d=10):
        """
        Recursively runs tests on all function configurations x times.
        """
        if x <= 0:
            return {"status": "ok"}
        for func_config in OptimizerRunner.function_configs(d):
            OptimizerRunner.test_for_function(func_config, n, max_iterations, d)
        return OptimizerRunner.runs(x - 1, n, max_iterations, d)

    @staticmethod
    def test(n=100, max_iterations=100, d=10):
        """
        Runs one test on all configured functions and prints results.
        """
        for func_config in OptimizerRunner.function_configs(d):
            OptimizerRunner.test_for_function(func_config, n, max_iterations, d)

    @staticmethod
    def test_for_function(func_config, n=100, max_iterations=100, d=10):
        """
        Uses the shifted problem factory to create a problem instance, performs optimization,
        and prints one formatted result line.
        """
        name = func_config["name"]
        base_lb = func_config["l"]
        base_ub = func_config["u"]
        base_cls = func_config["cls"]

        # Create a shifted problem using the factory.
        problem = create_shifted_problem(base_cls, d, base_lb, base_ub)

        start_time = time.time()
        optimizer = Optimizer()
        best_pos, best_cost = optimizer.optimize(problem, no_sharks=n, steps=max_iterations)
        end_time = time.time()

        elapsed_ms = int((end_time - start_time) * 1000)
        optim_n = Optimizer.n() if hasattr(Optimizer, 'n') else n
        print(f"PY {max_iterations} {optim_n} {d} {name} {elapsed_ms} {OptimizerRunner.format_solution(best_cost)}")

    @staticmethod
    def run(n=100, max_iterations=100, d=10):
        """
        Runs a detailed optimization on the Rosenbrock function (shifted version) and prints
        the true solution (which is now the random target), elapsed time, the optimizer solution, and the MAE.
        """
        shifted_lb, shifted_ub = -80, 80
        # No target is provided so that the optimum will be randomly placed in (-80,80)^d.
        problem = create_shifted_problem(RosenbrockProblem, d, -100.0, 100.0, shifted_lb, shifted_ub)

        # In our shifted problem, the optimum is the random target.
        true_solution_shifted = problem.optimal_point()
        true_val = problem.evaluate(true_solution_shifted)
        print("TRUE SOLUTION: {} at {}".format(
            OptimizerRunner.format_solution(true_val),
            OptimizerRunner.format_solution(true_solution_shifted)
        ))

        start_time = time.time()
        optimizer = Optimizer()
        best_pos, best_cost = optimizer.optimize(problem, no_sharks=n, steps=max_iterations)
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000.0

        print("WSO Elapsed time: {:.2f} ms".format(elapsed_ms))
        print("WSO SOLUTION: {} at {}".format(
            OptimizerRunner.format_solution(best_cost),
            OptimizerRunner.format_solution(best_pos)
        ))

    @staticmethod
    def format_solution(solution):
        """
        Formats a scalar or vector solution as a space-separated string.
        """
        if np.isscalar(solution):
            return str(solution)
        flat = np.array(solution).flatten()
        return " ".join(map(str, flat))

    @staticmethod
    def function_configs(d):
        """
        Returns the list of function configurations.
        Each dict includes the problem name, its class, and the base lower/upper bounds,
        which are used to construct the shifted problem.
        """
        return [
            {"name": "Rosenbrock", "cls": RosenbrockProblem, "l": -100, "u": 100},
            {"name": "Rastrigin",  "cls": RastriginProblem,  "l": -100, "u": 100},
            {"name": "Levy",       "cls": LevyProblem,       "l": -100, "u": 100},
            {"name": "Zakharov",   "cls": ZakharovProblem,   "l": -100, "u": 100},
            {"name": "Schwefel",   "cls": SchwefelProblem,   "l": -500, "u": 500},
            {"name": "BentCigar",  "cls": BentCigarProblem,  "l": -100, "u": 100},
        ]

# ------------------------------------------------------------------
# CLI Argument Parsing
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Optimizer Runner CLI")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands: runs, test, run")

    # Subparser for the "runs" command: recursive tests.
    parser_runs = subparsers.add_parser("runs", help="Recursively run tests on all functions x times")
    parser_runs.add_argument("x", type=int, help="Number of recursive runs")
    parser_runs.add_argument("n", type=int, nargs="?", default=100,
                             help="Population size / # sharks (default: 100)")
    parser_runs.add_argument("max_iterations", type=int, nargs="?", default=100,
                             help="Number of iterations/steps (default: 100)")
    parser_runs.add_argument("d", type=int, nargs="?", default=10,
                             help="Dimensions (default: 10)")

    # Subparser for the "test" command: one test per function.
    parser_test = subparsers.add_parser("test", help="Run a single test on all functions")
    parser_test.add_argument("n", type=int, nargs="?", default=100,
                             help="Population size / # sharks (default: 100)")
    parser_test.add_argument("max_iterations", type=int, nargs="?", default=100,
                             help="Number of iterations/steps (default: 100)")
    parser_test.add_argument("d", type=int, nargs="?", default=10,
                             help="Dimensions (default: 10)")

    # Subparser for the "run" command: a detailed run using Rosenbrock.
    parser_run = subparsers.add_parser("run", help="Run a detailed optimization on Rosenbrock")
    parser_run.add_argument("n", type=int, nargs="?", default=100,
                            help="Population size / # sharks (default: 100)")
    parser_run.add_argument("max_iterations", type=int, nargs="?", default=100,
                            help="Number of iterations/steps (default: 100)")
    parser_run.add_argument("d", type=int, nargs="?", default=10,
                            help="Dimensions (default: 10)")

    args = parser.parse_args()

    if args.command == "runs":
        OptimizerRunner.runs(args.x, args.n, args.max_iterations, args.d)
    elif args.command == "test":
        OptimizerRunner.test(args.n, args.max_iterations, args.d)
    elif args.command == "run":
        OptimizerRunner.run(args.n, args.max_iterations, args.d)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
