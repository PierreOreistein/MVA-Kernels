
def Normalisation(K, K_train=None, K_test=None):
    """Remove the mean of K."""

    def subNormalisation(K):
        """Compute the normalisation of K."""

        # Shape
        n, d = K.shape

        # Resulting array
        new_K = K.copy()

        # Apply normalisation
        if n == d:
            for i in range(n):
                for j in range(d):
                    new_K[i, j] /= (K[i, i] * K[j, j]) ** (1/2)

        else:
            for i in range(n):
                for j in range(d):

                    if (K_train[i, i] * K_test[j, j] < 10e-2):
                        new_K[i, j] == 0
                    else:
                        new_K[i, j] /= (K_train[i, i] * K_test[j, j]) ** (1/2)

        return new_K

    # Compute new_K
    new_K = subNormalisation(K)

    return new_K
