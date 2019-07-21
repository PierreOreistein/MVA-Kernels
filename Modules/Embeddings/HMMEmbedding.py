import itertools


def HMMEmbedding(df):
    """Compute the probability to pass from a letter to another."""

    # Shape
    n, d = df.shape

    # Copy data
    data_df = df.copy()

    # Letters
    letters = ["A", "C", "G", "T"]

    # Cartesian product
    lists = [letters] * 2
    cp = [elt for elt in itertools.product(*lists)]

    # Dict of probability
    proba_dict = {}
    total = 0

    for seq in cp:
        # Convert as a string
        str_seq = ''.join(seq)

        # Count
        proba_dict[str_seq] = data_df["seq"].apply(lambda x: x.count(str_seq)).sum()
        proba_dict[str_seq] /= (n * len(data_df["seq"].iloc[0]))
        total += proba_dict[str_seq]

    # Renormalise the probabilities
    proba_dict = {k: v / total for k, v in proba_dict.iteritems()}

    # Fisher score vector for each sequence

    return proba_dict
