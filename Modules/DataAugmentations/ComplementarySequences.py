import numpy as np


def ComplementarySequences(X, y):
    """Add the complementory sequence to the dataset."""

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
    new_X = np.array(new_X)
    new_y = np.array(new_y)

    return new_X, new_y
