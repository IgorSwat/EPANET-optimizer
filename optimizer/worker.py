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
    
    def __call__(self, solution):
        return self.problem.evaluate(solution)
    

# ----------------------------------
# Helper functions - worker handling
# ----------------------------------

_worker = None
_worker_dir = None

def evaluate_with_local_worker(solution, model_path, time_hrs, measured_df, dim, lb, ub):
    global _worker, _worker_dir

    if _worker is None:
        from .worker import EpanetWorker

        # Use Process ID as an unique identifier
        pid = os.getpid()
        base_dir = os.getcwd()
        _worker_dir = os.path.join(base_dir, f"tmp/worker_{pid}")
        
        os.makedirs(_worker_dir, exist_ok=True)

        # 1. Uzyskaj samą nazwę pliku z oryginalnej ścieżki
        model_filename = os.path.basename(model_path)

        # 2. Stwórz pełną ścieżkę docelową dla pliku w katalogu tymczasowym
        worker_model_path = os.path.join(_worker_dir, model_filename)

        # 3. Skopiuj plik modelu do katalogu tymczasowego pracownika
        shutil.copy(model_path, worker_model_path)

        _worker = EpanetWorker(_worker_dir, worker_model_path, time_hrs, measured_df, dim, lb, ub)

    result = _worker(solution)

    return result