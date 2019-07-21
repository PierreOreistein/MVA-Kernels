from joblib import delayed
from joblib import Parallel
import numpy as np
from tqdm import tqdm


def Prediction(best_parameters_dct, df_dict, n_jobs=-1):
    """Predication of the model for the best parameters.

       Take as argument a list of the best parameters for the three datasets.
       """

    # Extract the best model
    data_aug = best_parameters_dct["Data Augmentation"]["Function"]
    embedding = best_parameters_dct["Embedding"]["Function"]
    model = best_parameters_dct["Model"]["Function"]

    # Array of prediction
    pred = np.zeros((3 * 1000, 2), dtype=int)

    def subPredictions(k, df_dct=df_dict):
        """Compute predictions for a given dataset."""

        # Extraction of the data
        X_train_k = df_dct[k][0]["seq"].values
        y_train_k = df_dct[k][2]["Bound"].values
        X_test_k = df_dct[k][1]["seq"].values

        # Data augmentation
        X_train_k, y_train_k = data_aug.call(X_train_k, y_train_k)

        # Embedding
        X_train_k = embedding.call(X_train_k, train=True)

        # Training of the model
        model.fit(X_train_k, y_train_k)

        # Compute average score
        score = model.score(X_train_k, y_train_k) / 3

        # Prediction of test data
        X_test_k = embedding.call(X_test_k, train=False, X_train=df_dct[k][0]["seq"].values)
        y_pred_k = model.predict(X_test_k)

        return y_pred_k, score

    # Paralleisation of the cv
    if n_jobs != 1:
        preds_score = Parallel(n_jobs=n_jobs)(delayed(subPredictions)(k)
                                              for k in tqdm(range(3)))

    elif n_jobs == 1:
        preds_score = [subPredictions(k) for k in range(3)]

    # Initialisation of average score
    final_score = 0

    # Loop to extract the predictions and scores
    for k in range(3):

        # Update pred
        pred[1000 * k: 1000 * (k + 1), 0] = df_dict[k][1]["Id"].values
        pred[1000 * k: 1000 * (k + 1), 1] = preds_score[k][0].reshape(-1)

        # Update final_score
        final_score += preds_score[k][1]

    # Display average score
    print(final_score)

    return pred
