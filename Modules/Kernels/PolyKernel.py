import numpy as np
from numba import njit


def PolyKernel(X, Y, k=2, add_ones=False):
    """Compute the K matrix in the case of the linear kernel."""

    # Shape of X
    n, _ = np.shape(X)
    d, _ = np.shape(Y)

    # Convert X and Y
    X = np.array(X, dtype=np.float)
    Y = np.array(Y, dtype=np.float)

    @njit
    def subPolyKernel(X, Y):
        """Apply the dot product to X and Y."""

        return np.dot(X, Y.T)

    # Count the dot product
    result = subPolyKernel(X, Y)

    # Test if add ones
    if add_ones:
        # Compute results
        result = (np.array(result) + np.ones((n, d))) ** k

    else:
        # Compute results
        result = np.array(result) ** k

    return result
