import os
import shutil
import threading


# -------------------
# EPANET worker class
# -------------------

# A wrapper for multithreading EPANET evaluation
# - Each wrapper has it's own Epanet model instance to prevent any pickle errors
class EpanetWorker:

    # A global lock
    _lock = threading.Lock()

    def __init__(self, work_dir, worker_model_path, time_hrs, measured_df, dim, lb, ub):
        from .problem import EpanetProblem
        from epyt import epanet

        with EpanetWorker._lock:
            self.old_dir = os.getcwd()
            self.work_dir = work_dir

            os.chdir(self.work_dir)

            self.problem = EpanetProblem(
                dim=dim,
                lb=lb,
                ub=ub,
                model=epanet(worker_model_path),
                time_hrs=time_hrs,
                measured_df=measured_df
            )
    
    def __del__(self):
        try:
            os.chdir(self.old_dir)
            shutil.rmtree(self.work_dir)
        except Exception as e:
            print(f"Error during removal of temporal direction: {e}")

    def __call__(self, solution):
        return self.problem.evaluate(solution)