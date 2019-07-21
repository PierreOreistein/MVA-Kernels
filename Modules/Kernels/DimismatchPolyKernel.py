import itertools
from numba import jit
import numpy as np


def DimismatchPolyKernel(X, Y, m=1, d_l=[5], sigma=1, k=2, add_ones=False):
    """Compute the K matrix in the case of the linear kernel."""

    # Shape of X
    n = len(X)
    d = len(Y)

    # Convert X and Y as array
    X = np.array(X)
    Y = np.array(Y)

    # Resulting array
    result = np.zeros((n, d))

    # Letters
    letters = ["A", "C", "G", "T"]
    letters_converted = [ord(l) for l in letters]

    # Loop over the dimensions of the sequences
    for k in d_l:

        # Cartesian product
        lists = [letters_converted] * k
        k_mer_list = np.array([elt for elt in itertools.product(*lists)])

        # Shape
        d_k = len(k_mer_list)

        # Loop over the x_i
        for i in range(n):

            @jit(nopython=True, parallel=True)
            def subDimismatch(result, X, Y, i=i, k_mer_list=k_mer_list):
                """Compute the DimismacthEmbedding on for this list of k_mer."""

                for j in range(d):

                    # Extract the sequence i and j of X and Y
                    x_i = X[i]
                    y_j = Y[j]

                    # Length of the sequence
                    length = len(x_i)

                    # Loop over all the sequences
                    for l_1 in range(length - k + 1):

                        # Extract x_i_l
                        x_i_l = x_i[l_1:(l_1 + k)]

                        for l_2 in range(length - k + 1):

                            # Extract y_j_l
                            y_j_l = y_j[l_2:(l_2 + k)]

                            # Initialisation of the Hamming distance
                            hamming_dist = 0

                            # Loop over all k_mers
                            for j in range(d_k):

                                # Extract k_mer
                                k_mer = k_mer_list[j]

                                # Computation of matchs
                                matchs_x = 0
                                matchs_y = 0
                                for p in range(k):
                                    matchs_x += int((x_i_l[p] == k_mer[p]) *\
                                                    (x_i_l[p + 1] == k_mer[p + 1]))
                                    matchs_y += int((y_j_l[p] == k_mer[p]) *
                                                    (y_j_l[p + 1] == k_mer[p + 1]))

                                if (matchs_x >= (k - m)) and (matchs_y < (k - m)):
                                    hamming_dist += 1
                                if (matchs_x < (k - m)) and (matchs_y >= (k - m)):
                                    hamming_dist += 1

                            # Compute the score of K_0[i, j]
                            expo = k * np.exp(- 1 / (2 * sigma ** 2 * k) * (hamming_dist) ** 2)

                            # Update K[i, j]
                            result[i, j] += expo

                return result

        # Update result
        result = subDimismatch(result, X, Y)

    # Compute result
    if add_ones:

        # Shape of X
        n, _ = np.shape(X)
        d, _ = np.shape(Y)

        # Compute results
        result = (np.dot(X, Y.T) + np.ones((n, d))) ** k

    else:
        result = np.dot(X, Y.T) ** k

    return result
