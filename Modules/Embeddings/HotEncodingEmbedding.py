import itertools
import pandas as pd
import numpy as np
from numba import jit
from tqdm import tqdm


def HotEncodingEmbedding(X, d=[5, 6, 7]):
    """Implementation of the HotEncoding Embedding with DimismatchEmbedding."""

    # Shape
    n = np.shape(X)[0]

    # Convert X as a DataFrame to deal with numba
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
    for k in tqdm(d):

        # Cartesian product
        lists = [letters_converted] * k
        k_mer_l = np.array([elt for elt in itertools.product(*lists)])

        # Shape
        d_k = len(k_mer_l)

        # Resulting array
        new_X_k = np.zeros((n, len(X[0]) - k, d_k), dtype=np.bool_)

        @jit(nopython=True, parallel=True)
        def subHotEncodingEmbedding(new_X_k, X_conv=X_converted, k_mer=k_mer_l):
            """Compute the DimismacthEmbedding on for this list of k_mer."""

            # Computation of the embedding of X
            for i in range(n):

                # Extract the sequence i
                x_i = X_conv[i]
                len_i = len(x_i)

                for l in range(len_i - k + 1):

                    # Extract x_il
                    x_il = x_i[l:(l + k)]

                    # Extract indices
                    for j in range(d_k):

                        # Extract k_mer_j
                        k_mer_j = k_mer[j]

                        # Computation of gamma_k_m
                        matchs = 0

                        for p in range(k):

                            matchs += int(x_il[p] == k_mer_j[p])

                        if matchs >= k:
                            new_X_k[i, l, j] = 1

            return new_X_k

        # Compute the embedding for the given list of k_mer_l
        new_X_k = subHotEncodingEmbedding(new_X_k)

        # Update new_X
        if new_X is None:
            new_X = new_X_k.reshape((n, -1))
        else:
            new_X = np.hstack((new_X, new_X_k.reshape((n, -1))))
            new_X = new_X.astype(np.bool_)

    # Return data
    return new_X
