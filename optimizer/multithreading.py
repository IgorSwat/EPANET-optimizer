import os
import shutil


# -------------------
# EPANET worker class
# -------------------

# A wrapper for multithreading EPANET evaluation
# - Each wrapper has it's own Epanet model instance to prevent any pickle errors
class EpanetWorker:
    def __init__(self, worker_model_path, time_hrs, measured_df, dim, lb, ub, work_dir):
        from .problem import EpanetProblem
        from epyt import epanet

        self.work_dir = work_dir

        self.problem = EpanetProblem(
            dim=dim,
            lb=lb,
            ub=ub,
            model=epanet(worker_model_path),
            model_filepath=worker_model_path,
            time_hrs=time_hrs,
            measured_df=measured_df
        )

        os.chdir(self.work_dir)

    def __call__(self, solution):
        return self.problem.evaluate(solution)
    

# Helper function - instantiating workers
def create_worker(*args, **kwargs):
    return EpanetWorker(*args, **kwargs)