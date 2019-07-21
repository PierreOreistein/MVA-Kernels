import numpy as np


def Dropout(K, p=0.15):
    """Dropout p * 100 % of the martix K."""

    # Shape
    n, d = K.shape

    # Building of the dropout matrix
    dropout_mat = np.random.choice([0, 1], size=(n, d), p=[p/2, 1 - p/2])

    # Centering of the kernel
    new_K = np.where(dropout_mat > 0, K, 0)
    new_K = np.where(dropout_mat.T > 0, new_K, 0)

    return new_K
