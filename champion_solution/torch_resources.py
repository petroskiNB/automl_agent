"""Resource policy for the experimental entry point (set before ML imports)."""
import os


def configure(threads=None):
    threads = int(threads if threads is not None else os.getenv("AUTOML_CPU_THREADS", "2"))
    if not 1 <= threads <= 8:
        raise ValueError("CPU threads must be in 1..8; this is a per-training-process allocation")
    os.environ["AUTOML_CPU_THREADS"] = str(threads)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[key] = str(threads)
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
    return threads


def configure_torch(threads=None):
    threads = configure(threads)
    import torch
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    return threads
