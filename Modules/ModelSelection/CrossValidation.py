from joblib import delayed
from joblib import Parallel
import numpy as np
import time


def CrossValidation(df_dct, model, cv=5, n_jobs=-1, return_int=True):
    """Apply a cross validation to the model."""

    # Average of the score
    all_scores = []

    # Loop over the three datasets
    for i in range(3):

        start = time.time()

        # Extraction of the dataset
        X_train_i = df_dct[i][0]
        y_train_i = df_dct[i][2]

        # Shape of data
        n = np.shape(X_train_i)[0]
        step = n // cv

        def oneFold(k, cv=cv):
            """Execute one fold of the cv."""

            # Index for the training set and testing set
            if k == cv - 1:
                idx_test = np.arange(k * step, n)
            else:
                idx_test = np.arange(k * step, (k + 1) * step)
            idx_train = np.delete(np.arange(0, n), idx_test)

            # Extract the kth X_train and X_test batch
            X_train_k = [X_train_i[i] for i in idx_train]
            y_train_k = np.array([y_train_i[i] for i in idx_train])
            X_test_k = [X_train_i[i] for i in idx_test]
            y_test_k = np.array([y_train_i[i] for i in idx_test])

            # Fitting of the model on this batch
            model.fit(X_train_k, y_train_k)

            # Compute the score for this fold
            score_i_k = model.score(X_test_k, y_test_k)
            print("Score i k", score_i_k)

            return score_i_k

        # Paralleisation of the cv
        if n_jobs != 1:
            scores_i = Parallel(n_jobs=n_jobs)(delayed(oneFold)(k) for k in range(cv))

        elif n_jobs == 1:
            scores_i = [oneFold(k) for k in range(cv)]

        # Update the result
        all_scores.append(scores_i)

        # Display the time required
        print("Time of the cross-validation: {:4f}, Score: {:4f}".format(
              time.time() - start, np.mean(scores_i)))

        # Break if score too low
        if (np.mean(scores_i)) < 0.62 and (i == 0):
            all_scores.append([0 for j in range(cv)])
            all_scores.append([0 for j in range(cv)])
            break

    if return_int:
        means = [np.mean(scores) for scores in all_scores]
        return np.mean(means)

    else:
        return all_scores
