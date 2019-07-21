import numpy as np
from tqdm import tqdm


def SpectrumKernel(X, Y, d_l=[5, 12]):
    """Compute the K matrix in the case of the Spectrum kernel."""

    # Shape of X
    n = len(X)
    d = len(Y)

    # Length of the sequences
    len_seq = len(X[0])

    # Resulting array
    result = np.zeros((n, d))

    # Loop over all the sequences
    for i in tqdm(range(n)):
        for j in range(d):

            # Extract the sequence i and j of X and Y
            x_i = X[i]
            y_j = Y[j]

            # Loop over all the size of k_mers
            for d_k in d_l:

                # Extract all sub list of size d_k
                kmers_x = [x_i[l:(l+d_k)] for l in range(len_seq - d_k + 1)]

                # Extract k_mers and counts associated
                kmers_x, counts_x = np.unique(kmers_x, return_counts=True,
                                              axis=0)

                # Loop over all these k_mers
                for l, kmer in enumerate(kmers_x):

                    # Count the occurrences of kmer in y_j
                    counts_y_j_kmer = y_j.count(kmer)

                    # Update result[i, j]
                    result[i, j] += counts_x[l] * counts_y_j_kmer

    return result
