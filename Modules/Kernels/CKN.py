from numba import jit
import numpy as np


def CKN(X, Y, sigma=1, d_l=[5, 6, 7]):
    """Compute the K matrix in the case of the Convolutional Kernel Network."""

    # Shape of X
    n = len(X)
    d = len(Y)

    # Resulting array
    result = np.zeros((n, d))

    @jit(nopython=True)
    def subCKN(result, X, Y):
        """Compute the DimismacthEmbedding on for this list of k_mer."""

        # Computation of the embedding of X
        for i in range(n):
            for j in range(d):

                # Extract the sequence i and j of X and Y
                x_i = X[i]
                y_j = Y[j]

                # Length of the sequence
                length = len(x_i)

                # Loop over all the sequences
                for l_1 in range(length):
                    for l_2 in range(length):

                        # Initialisation of the Hamming distance
                        hamming_dist = 0

                        # Loop over all possible k_mer
                        for d_k in d_l:

                            # Test of indices
                            if (l_1 + d_k <= length) and (l_2 + d_k <= length):

                                # Extract x_i_l
                                x_i_l = x_i[l_1:(l_1 + d_k)]
                                y_j_l = y_j[l_2:(l_2 + d_k)]

                                # Computation of matchs
                                matchs = 0

                                for p in range(d_k):
                                    matchs += int(x_i_l[p] == y_j_l[p])

                                if matchs >= d_k:
                                    hamming_dist += 1

                            # Compute the score of K_0[i, j]
                            expo = np.exp(1 / (sigma ** 2) *
                                          (hamming_dist / d_k - 1))

                            # Update K[i, j]
                            result[i, j] += d_k * expo / (n * d)

            return result

        # Update result
        result = subCKN(result, X, Y)

    return result
