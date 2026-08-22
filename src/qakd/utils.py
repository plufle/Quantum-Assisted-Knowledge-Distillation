import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Make a training run reproducible across reruns on the same stack.

    QAKD comparisons are three matched seeds, so unseeded CUDA kernels can turn a
    method comparison into hardware noise.  Set this before the first CUDA
    operation in every train entry point.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Required by cuBLAS when deterministic GEMM algorithms are enabled.  `setdefault`
    # preserves an explicit value supplied by an experiment launcher.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
    except ImportError:
        pass
