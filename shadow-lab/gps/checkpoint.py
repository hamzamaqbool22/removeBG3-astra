"""Restricted loader for legacy GPS checkpoints containing NumPy score metadata."""
import numpy as np
from numpy._core.multiarray import scalar
import torch


def load_checkpoint(path):
    # NumPy 1.x stored this scalar constructor under numpy.core; NumPy 2.x
    # moved it to numpy._core. Older checkpoints also carry concrete dtype
    # objects (often the validation score's float64), not only tensor weights.
    numeric_dtypes = [type(np.dtype(code)) for code in
                      ('bool', 'int8', 'int16', 'int32', 'int64',
                       'uint8', 'uint16', 'uint32', 'uint64',
                       'float16', 'float32', 'float64', 'complex64', 'complex128')]
    allowed = [np.dtype, *numeric_dtypes,
               (scalar, 'numpy.core.multiarray.scalar'),
               (scalar, 'numpy._core.multiarray.scalar')]
    # Scope the allowlist to this load. Never fall back to unrestricted pickle.
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location='cpu', weights_only=True)
