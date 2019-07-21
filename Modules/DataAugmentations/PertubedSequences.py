import numpy as np


def PertubedSequences(X, y, p=0.05, n=2, add_compl=False):
    """Add the complementory sequence to the dataset."""

    if add_compl:

        # Complementary table
        complementary_table = {"A": "T",
                               "T": "A",
                               "G": "C",
                               "C": "G"}

        # New array
        new_X = []
        new_y = []

        # Equivalent sequences
        for i, x_i in enumerate(X):

            # Extract x_i
            compl_x_i = ''.join([complementary_table[letter] for letter in x_i])

            # Append new_X and new_y
            new_X.extend([x_i, compl_x_i])
            new_y.extend([y[i], y[i]])

        # Convert as array
        X = np.array(new_X)
        y = np.array(new_y)

    # DNA letters
    letters = ["A", "T", "G", "C"]

    # Length of the sequences
    len_seq = len(X[0])

    # New array
    new_X = []
    new_y = []

    # Equivalent sequences
    for i, x_i in enumerate(X):

        # Append new_X and new_y
        new_X.append(x_i)
        new_y.append(y[i])

        # Compute x_i splited
        x_i_splitted = np.array(list(x_i))

        # Number of pertubated sequences to generate
        for k in range(n):

            # Saving array of the new sequence
            new_x_i = []

            # boolean
            booleans = np.random.binomial(1, p, len_seq)

            # Update new_x_i
            new_x_i = np.where(booleans > 0, np.random.choice(letters), x_i_splitted)
            new_x_i = ''.join((list(new_x_i)))

            # Append new_X and new_y
            new_X.append(new_x_i)
            new_y.append(y[i])

    # Convert as array
    new_X = np.array(new_X)
    new_y = np.array(new_y)

    return new_X, new_y
