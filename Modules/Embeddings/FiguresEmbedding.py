import pandas as pd
import numpy as np


def FiguresEmbedding(X):
    """Transform the letter as figures."""

    # Convert X as a DataFrame to deal with numba
    X_df = pd.DataFrame(X, columns=["seq"])
    X_df["ascii"] = X_df["seq"].apply(lambda x: list(x))
    X_converted = X_df["ascii"].apply(lambda x: [ord(l) for l in x]).values
    X_converted = np.array(X_converted.tolist())

    return X_converted
