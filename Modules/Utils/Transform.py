import numpy as np


def Transform(K):
    """Transform K by  I - D1/2KD1/2."""

    # Shape
    n, d = K.shape

    # Compute D
    D_12 = np.diagflat(np.sum(K, axis=1))

    # Centering of the kernel
    new_K = np.identity(n) - D_12 @ K @ D_12

    return new_K
