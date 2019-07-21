import numpy as np


def NoAugmentation(X, y):
    """Just convert the datasets as arrays."""

    return np.array(X), np.array(y)
