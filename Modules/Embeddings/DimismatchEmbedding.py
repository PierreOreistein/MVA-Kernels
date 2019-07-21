import itertools
import pandas as pd
import numpy as np
from numba import jit


def DimismatchEmbedding(X, d=[2, 5, 6], m=2, seed=42):
    """Implementation of the DimismatchKernel."""

    # Shape
    n, = np.array(X).shape

    # Convert X as a DataFrame
    X_df = pd.DataFrame(X, columns=["seq"])
    X_df["ascii"] = X_df["seq"].apply(lambda x: list(x))
    X_converted = X_df["ascii"].apply(lambda x: [ord(l) for l in x]).values
    X_converted = np.array(X_converted.tolist(), dtype=np.int8)

    # Letters
    letters = ["A", "C", "G", "T"]
    letters_converted = [ord(l) for l in letters]

    # Resulting array
    new_X = None

    # Loop over the dimensions of the sequences
    for k in d:

        # Cartesian product
        lists = [letters_converted] * k
        k_mer_l = np.array([elt for elt in itertools.product(*lists)])

        # Test if d_k is too big
        if len(k_mer_l) > 4096:

            # Initialisation of the seed
            np.random.seed(seed)

            # Select 10e6 columns randomly
            k_mer_l = k_mer_l[np.random.randint(0, len(k_mer_l), int(4096))]

        # Shape
        d_k = len(k_mer_l)

        @jit(nopython=True, parallel=True)
        def subDimismatchEmbedding(X_conv=X_converted, k_mer=k_mer_l):
            """Compute the DimismacthEmbedding on for this list of k_mer."""

            # Resulting array
            new_X_k = np.zeros((n, d_k))

            # Computation of the embedding of X
            for j in range(d_k):
                for i in range(n):

                    # Extract x_i_l and k_mer_j
                    k_mer_j = k_mer[j]
                    x_i_l = X_conv[i]

                    # Compute rho(x_i, k_mer_j)
                    rho = 0
                    len_i = len(x_i_l)

                    for l in range(len_i - k):

                        x_il = x_i_l[l:(l + k)]

                        # Computation of gamma_k_m
                        gamma = 0

                        for p in range(k - 1):

                            gamma += int((x_il[p] == k_mer_j[p]) *
                                         (x_il[p + 1] == k_mer_j[p + 1]))

                        if gamma >= (k - m):
                            rho += gamma

                    # Update resulting array
                    new_X_k[i, j] = rho

            return new_X_k

        # Compute the embedding for the given list of k_mer_l
        new_X_k = subDimismatchEmbedding()

        # Update new_X
        if new_X is None:
            new_X = new_X_k
        else:
            new_X = np.hstack((new_X, new_X_k))

    # Return data
    return new_X
