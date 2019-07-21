import numpy as np


def sigmoid(x):
    """Return the value of the sigmoid in x."""

    x_clipped = np.clip(x, -100, 100)
    # if np.min(x) < -10:
    #     return np.where(x < -10, 5 * 1e-5, 1 / (1 + np.exp(-x)))

    return 1 / (1 + np.exp(-x_clipped))


class KernelLogisticRegression(object):
    def __init__(self, kernel=None, preprocessing=None, normalisation=None,
                 dropout=None, maxi=False, lamda=1, max_iter=15, informations=True):
        """Initialisation of the Perceptron class"""

        # Name of the class
        self.name = "KernelLogisticRegression"

        # Hyperparameter for the regulariser in the optimisation
        self.lamda = lamda

        # Kernel
        self.kernel = kernel

        # Gradient Descent parameters
        self.max_iter = max_iter
        self.informations = informations

        # Function to preprocess the data
        self.preprocessing = preprocessing
        self.normalisation = normalisation
        self.dropout = dropout
        self.maxi = maxi

    def fit(self, data_train, labels, alpha_init=[]):
        """Fitting of the model."""

        # Computation of K
        self.K_train_func = lambda data: self.kernel.call(data_train, data)
        self.K_train = self.K_train_func(data_train)

        # Preprocessing of K_train
        if self.preprocessing is not(None):
            self.K_train = self.preprocessing(self.K_train)
        if self.normalisation is not(None):
            self.K_train = self.normalisation(self.K_train)

        # Bijection between values of y and {-1, 1}
        self.Fromlabels = {min(labels): -1, max(labels): 1}
        self.Tolabels = {-1: min(labels), 1: max(labels)}
        self.y = np.array([self.Fromlabels[y_i] for y_i in labels]).reshape((-1, 1))

        # Drop complementary y
        if self.maxi:
            idx_even = np.arange(len(data_train) // 2) * 2
            self.y = self.y[idx_even]

        # Initialisation of the vector of weights
        if len(alpha_init) == 0:
            self.alpha = np.zeros((self.K_train.shape[1], 1))
            # self.alpha =  np.zeros((self.K_train.shape[1], 1)) + 10e-4
        else:
            self.alpha = alpha_init

        # Saving of the previous iterations of the value of alpha and f
        self.histo_alpha = []
        self.histo_f = []

        # Peform the gradient descent
        self.gradient_decent()

    def predict(self, data_test, average_size=3):
        """Prediction of the label for the given data."""

        # Computation of K
        K_pre = self.K_train_func(data_test)

        # Preprocessing of K_test
        if self.preprocessing is not(None):
            K_pre = self.preprocessing(K_pre)

        if self.normalisation is not(None):
            K_test = self.kernel.call(data_test, data_test)
            K_test = self.normalisation(self.preprocessing(K_test))
            K_pre = self.normalisation(K_pre, K_train=self.K_train,
                                       K_test=K_test)

        # Computation of the average w over the last average_size iteration
        alpha_predict = np.mean(self.histo_alpha[-average_size:, :], axis=0)
        alpha_predict = alpha_predict.reshape(-1)

        # Prediction
        y_pred = np.where(sigmoid(alpha_predict.T.dot(K_pre)) >= 0.5, 1, -1)
        label_pred = np.array([self.Tolabels[y_i] for y_i in y_pred])
        label_pred = label_pred.reshape((-1, 1))

        return label_pred

    def predict_proba(self, data_test, average_size=3):
        """Predict the probability to belong to the class 1."""

        # Computation of K
        K_test = self.K_train_func(data_test)

        # Computation of the average w over the last average_size iteration
        alpha_predict = np.mean(self.histo_alpha[-average_size:, :], axis=0)
        alpha_predict = alpha_predict.reshape(-1)
        proba_predict = sigmoid(alpha_predict.T.dot(K_test))

        return proba_predict

    def score(self, data_test, labels_test):
        """Compute the accuracy."""

        # Drop complementary y
        if self.maxi:
            idx_even = np.arange(len(labels_test) // 2) * 2
            labels_test = labels_test[idx_even]

        # Compute predictions
        y_pred = self.predict(data_test)

        return np.mean(y_pred == labels_test.reshape((-1, 1)))

    def loss(self):
        """Compute the loss."""

        # Dropout
        if self.dropout is not None:
            K_grad = self.dropout(self.K_train)
        else:
            K_grad = self.K_train

        # Compute Loss
        loss = np.log(1 + np.exp(-self.y * np.dot(K_grad, self.alpha)))
        loss = np.mean(loss)
        loss += self.lamda / 2 * np.dot(self.alpha.T, np.dot(K_grad,
                                                             self.alpha))

        return loss

    def grad_loss(self):
        """Compute the gradient of the loss."""

        # Shape
        n, _ = np.shape(self.K_train)

        # Dropout
        if self.dropout is not None:
            K_grad = self.dropout(self.K_train)
        else:
            K_grad = self.K_train

        # Compute gradient Loss
        grad = 1 / n * np.dot(np.dot(K_grad, np.diagflat(self.P)), self.y)
        grad += self.lamda * np.dot(K_grad, self.alpha)

        return grad.reshape((-1, 1))

    def gradient_decent(self):
        """Execution of the gradient descent"""

        # New iteration of the gradient
        if self.informations:
            print("\n----------")

        # Shape of K
        n, _ = np.shape(self.K_train)

        # Initialisation of the iterator and P
        ite = 0
        self.P = np.random.rand(n, 1) + 10e5

        while ite < self.max_iter:  # and np.linalg.norm(self.grad_loss()) > 10e-15:

            # Dropout
            if self.dropout is not None:
                K_grad = self.dropout(self.K_train)
            else:
                K_grad = self.K_train

            # Update the parameters of the WKRR problem
            self.m = np.dot(K_grad, self.alpha)
            self.P = -sigmoid(-self.y * self.m)
            self.W = sigmoid(self.m) * sigmoid(-self.m)
            self.z = self.m - self.y * self.P / self.W
            grad = self.grad_loss()

            # Solve the WKRR problem
            # W_sqrt = np.sqrt(np.diagflat(self.W))
            # inv = np.linalg.inv(np.dot(W_sqrt, np.dot(self.K_train, W_sqrt))+\
            #                     n * self.lamda * np.identity(n))
            # self.alpha = np.dot(W_sqrt, np.dot(np.dot(inv, W_sqrt), self.z))

            # Solve the WKRR problem thanks to the inversion matrix lemma
            W_diag = np.diagflat(self.W)
            inv = np.linalg.inv(np.dot(W_diag, K_grad) +
                                (n * self.lamda + 10e-5) * np.identity(n))
            self.alpha = np.dot(inv, np.dot(W_diag, self.z))

            # Saving of the newly computed w
            self.histo_alpha.append(self.alpha.reshape(-1))
            self.histo_f.append(self.loss())

            # Display the progress of the gradient descent
            if self.informations:  # ite % 10 == 0 and*
                norm = np.linalg.norm(grad)
                print("Iterations done: {}, Loss: {}, Gradient: {}".format(ite,
                                                                           self.histo_f[-1],
                                                                           norm))

            # Update of the iterator
            ite += 1

        # Update convert histo_f and histo_alpha as array
        self.histo_f = np.array(self.histo_f).reshape((-1, 1))
        self.histo_alpha = np.array(self.histo_alpha).reshape((-1, n))
