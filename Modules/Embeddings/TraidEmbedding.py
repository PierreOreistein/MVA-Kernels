import numpy as np


def TraidEmbedding(X):
    """Count the number of times a letter is present."""

    # Length of the sequences
    len_seq = len(X[0])

    # Loop over each sequences in X
    result = []

    # Clusters
    converter = {"A": "A", "C": "A", "G": "A", "T": "T"}

    # All combinations
    combinations = ["AAA", "AAT", "ATA", "ATT", "TAA", "TAT", "TTA", "TTT"]

    for x_i in X:

        # Extract all sub list of size d_k
        sub_list_i = [x_i[l:(l+3)] for l in range(len_seq - 3 + 1)]

        # Convert each sub_list
        sub_list_i_converted = []

        for sub_list in sub_list_i:

            sub_list_i_converted.append("".join([converter[l] for l in sub_list]))

        # Count the frequency
        x_i_embedded = []
        for comb in combinations:

            # Update x_i_embedded
            x_i_embedded.append(sub_list_i.count(comb))

        # Append result
        result.append(x_i_embedded)

    return np.array(result)
