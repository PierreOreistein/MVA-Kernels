import pandas as pd


def ImportData(file_path_x, file_path_y, suffix="", header=None, sep=" "):
    """Import the data in file_name and return them as a Panda DataFrame."""

    # Dictionnary containing all the dataset required
    df_dict = {}

    for k in range(3):
        # Extraction of the training set , the testing set and the labels of
        # the training set
        X_train_df = pd.read_csv(file_path_x + "Xtr" + str(k) + suffix +
                                 ".csv", header=header, sep=sep)
        X_test_df = pd.read_csv(file_path_x + "Xte" + str(k) + suffix + ".csv",
                                header=header,
                                sep=sep)
        Y_train_df = pd.read_csv(file_path_y + "Ytr" + str(k) + ".csv")

        # Adding of these datasets to df_dict
        df_dict[k] = [X_train_df, X_test_df, Y_train_df]

    return df_dict
