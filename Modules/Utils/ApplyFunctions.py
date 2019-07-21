import numpy as np


def ApplyFunctions(df_dct, data_aug, embedding):
    """Apply the DataAugmentation and Embedding function to df_dct."""

    # Resulting dict
    result_dct = {}

    # Loop over the three datasets
    for i in range(3):

        # Extraction of the dataset
        X_train_i = df_dct[i][0]["seq"].values
        y_train_i = df_dct[i][2]["Bound"].values

        # Shape of data
        n = np.shape(X_train_i)[0]

        # Initialise the seed
        np.random.seed(42)

        # Shuffle the data
        randomised_idx = np.random.permutation(np.arange(n))
        randomised_X = X_train_i[randomised_idx]
        randomised_y = y_train_i[randomised_idx]

        # Data augmentation
        new_X_train_i, new_y_train_i = data_aug.call(randomised_X,
                                                     randomised_y)

        # Embedding of X_train and X_test
        new_X_train_i = embedding.call(new_X_train_i, train=True)

        # Save the resulting datasets
        result_dct[i] = {0: new_X_train_i, 2: new_y_train_i}

    return result_dct
