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

            self.model = epanet(worker_model_path)

            self.problem = EpanetProblem(
                dim=dim,
                lb=lb,
                ub=ub,
                model=self.model,
                time_hrs=time_hrs,
                measured_df=measured_df.copy()
            )
    
    def __call__(self, solution):
        return self.problem.evaluate(solution)
    
    # NOTE: Very important to close loaded EPANET model at the end of object's lifetime
    def __del__(self):
        self.model.unload()
    

# ----------------------------------
# Helper functions - worker handling
# ----------------------------------

_worker = None
_worker_dir = None

def evaluate_with_local_worker(solution, model_path, tmp_path, time_hrs, measured_df, dim, lb, ub):
    global _worker, _worker_dir

    if _worker is None:
        from .worker import EpanetWorker

        # Use Process ID as an unique identifier
        pid = os.getpid()
        base_dir = os.getcwd()
        _worker_dir = os.path.join(base_dir, tmp_path, f"worker_{pid}")
        
        os.makedirs(_worker_dir, exist_ok=True)

        model_filename = os.path.basename(model_path)

        worker_model_path = os.path.join(_worker_dir, model_filename)

        shutil.copy(model_path, worker_model_path)

        _worker = EpanetWorker(work_dir=_worker_dir, worker_model_path=worker_model_path, time_hrs=time_hrs, 
                               measured_df=measured_df, dim=dim, lb=lb, ub=ub)

    result = _worker(solution)

    return result