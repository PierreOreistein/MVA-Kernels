import itertools
import pandas as pd
import numpy as np
from numba import jit
from tqdm import tqdm


def WeightedDegreeEmbedding(X, d=[5, 6, 7], m=0):
    """Implementation of the WeightedDegree Embedding with DimismatchEmbedding."""

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

        @jit(nopython=True, parallel=True)
        def subDimismatchEmbedding(X_conv=X_converted, k_mer=k_mer_l):
            """Compute the DimismacthEmbedding on for this list of k_mer."""

            # Resulting array
            new_X_k = []

            # Computation of the embedding of X
            for i in range(n):

                # Extract the sequence i
                x_i_l = X_conv[i]
                len_i = len(x_i_l)

                # Saving array of x_i
                x_i_save = []

                for l in range(len_i - k):

                    # Saving array for l
                    x_i_l_save = []

                    # Extract x_il
                    x_il = x_i_l[l:(l + k)]

                    for j in range(d_k):

                        # Extract k_mer_j
                        k_mer_j = k_mer[j]

                        # Computation of gamma_k_m
                        gamma = 0

                        for p in range(k - 1):

                            gamma += int((x_il[p] == k_mer_j[p]) *
                                         (x_il[p + 1] == k_mer_j[p + 1]))

                        if gamma >= (k - m - 1):
                            x_i_l_save.append(j)

                    # Update x_i_save
                    x_i_save.append(x_i_l_save)

                # Update resulting array
                new_X_k.append(x_i_save)

            return new_X_k

        # Compute the embedding for the given list of k_mer_l
        new_X_k = subDimismatchEmbedding()

        # Update new_X
        if new_X is None:
            new_X = new_X_k
        else:
            for i in range(n):
                new_X[i].extend(new_X_k[i])

    # Return data
    return new_X
