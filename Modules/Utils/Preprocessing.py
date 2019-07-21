import numpy as np
from numba import jit


@jit
def Preprocessing(K):
    """Remove the mean of K."""

    # Shape
    n, d = K.shape

    # Building of the matrix I - U
    i_u_left = np.identity(n) - np.ones((n, n)) / n
    i_u_right = np.identity(d) - np.ones((d, d)) / d

    # Centering of the kernel
    new_K = np.dot(i_u_left, np.dot(K, i_u_right))

    return new_K
