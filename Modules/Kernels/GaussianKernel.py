import numpy as np
from scipy.spatial.distance import cdist


def gaussianDiff(X, xp, sigma):
    """Compute the gaussian of the difference between a dataset X and xp."""

    # Reshape the array
    xp = xp.reshape((1, -1))

    # Compute the L2 norm of the difference
    norm = np.linalg.norm(X - xp, axis=1)

    return np.exp(- norm ** 2 / (2 * sigma ** 2))


def GaussianKernel(X, Y, nb=10, mode=1, sigma=1, fit_intercept=False):
    """Compute the matrix K for the gaussian kernel."""

    def B(X_train, nb=10, mode=1, sigma=1):
        """Choice of the centroids."""

        # Shape of the data
        nx, dx = np.shape(X_train)

        if mode == 1:  # Random centroids
            min_x, max_x = np.amin(X_train, axis=0), np.amax(X_train, axis=0)
            matb = np.random.rand(nb, dx)
            matb = (max_x - min_x) * matb + min_x

        else:  # Centroids taken among the data points
            indices = range(nx)
            np.random.seed(42)
            indices = np.random.permutation(indices)
            matb = X_train[indices[:nb], :]

        return matb

    # Add a row of one if fit_intercept
    if fit_intercept:

        # X
        n_x, d_x = np.shape(X)
        X = np.vstack((X, np.ones((1, d_x))))

        # Y
        n_y, d_y = np.shape(Y)
        Y = np.vstack((Y, np.ones((1, d_y))))

    # Extract the centroids
    B_mat = B(X, nb=nb, mode=mode, sigma=sigma)

    # Extract the shape of the matrices given in argument
    nb, db = np.shape(B_mat)
    nx, dx = np.shape(X)
    ny, dy = np.shape(Y)
    K_X = np.zeros((nx, nb))
    K_Y = np.zeros((ny, nb))

    for i in range(nb):
        o = B_mat[i, :]
        K_X[:, i] = gaussianDiff(X, o, sigma).reshape(-1)
        K_Y[:, i] = gaussianDiff(Y, o, sigma).reshape(-1)

    return np.dot(K_X, K_Y.T)


def GaussianKernelBIS(X, Y, sigma=None):
    """Compute the gaussian kernel."""

    # Extract shapes
    nx, dx = np.shape(X)
    ny, dy = np.shape(Y)

    # Definition of gamma
    if sigma is None:
        gamma = 1.0 / (dx * np.std(X))
    else:
        gamma = 1.0 / (2.0 * sigma ** 2)

    # Check the dimensions
    assert dx == dy

    norms = cdist(X, Y, metric='sqeuclidean')
    res = -np.exp(gamma * norms)

    return res
