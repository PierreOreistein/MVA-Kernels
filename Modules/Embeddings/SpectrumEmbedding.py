import numpy as np
import pandas as pd
from tqdm import tqdm


def extractKmers(X, k=12):
    """Extract the kmers of length k of X."""

    # Length of the sequences
    len_seq = len(X[0])

    # Initialisation of the set of kmers
    k_mers = set()

    # Loop over the sequences of X
    for x_i in X:

        # Loop over the sequence
        for l in range(len_seq - k + 1):

            # Extract k_mer of x_i
            k_mer_i = x_i[l:(l + k)]

            # Update k_mers
            k_mers.update([k_mer_i])

    return list(k_mers)


def SpectrumEmbedding(X, d_l=[5, 12], train=True, X_train=None):
    """Count the number of times a letter is present."""

    # Shape
    n, = np.array(X).shape

    # Extract all kmers of dimension in d_l
    kmers = set()
    for k in d_l:

        # Update kmers
        if train:
            kmers.update(extractKmers(X, k=k))
        else:
            kmers.update(extractKmers(X_train, k=k))

    # Convert kmers as a list
    kmers = list(kmers)

    # Convert X as a dataFrame
    X_df = pd.DataFrame(X, columns=["seq"])

    # Columns to extract
    result = []

    # Loop over all kmers
    for kmer in tqdm(kmers):

        # Add a new_column
        result.append(X_df["seq"].str.count(kmer).values)

    # Convert result as a numpy array and transpose
    result = np.transpose(np.array(result, dtype=np.uint8))

    return result
