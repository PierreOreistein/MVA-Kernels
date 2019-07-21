import numpy as np
from numba import jit


@jit
def LinearKernel(X, Y, fit_intercept=False):
    """Compute the K matrix in the case of the linear kernel."""

    return np.dot(X, Y.T)
