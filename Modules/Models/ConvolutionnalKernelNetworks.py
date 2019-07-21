import numpy as np


def sigmoid(x):
    """Return the value of the sigmoid in x."""

    x_clipped = np.clip(x, -100, 100)
    # if np.min(x) < -10:
    #     return np.where(x < -10, 5 * 1e-5, 1 / (1 + np.exp(-x)))

    return 1 / (1 + np.exp(-x_clipped))


class ConvolutionnalKernelNetworks(object):
    def __init__(self, learning_rate=1, lamda=10e-4, max_iter=5,
                 informations=True, kernel=None):
        """Initialisation of the Perceptron class"""

        # Name of the class
        self.name = "Convolutionnal Kernel Networks"

        # Hyperparameter for the regulariser in the optimisation
        self.lamda = lamda

        # Gradient Descent parameters
        self.max_iter = max_iter
        self.informations = informations

        # Function to preprocess the data
        self.learning_rate = learning_rate

    def fit(self, data_train, labels, w_init=[]):
        """Fitting of the model."""

        # Save data train
        self.X = np.array(data_train)

        # Bijection between values of y and {-1, 1}
        self.Fromlabels = {min(labels): -1, max(labels): 1}
        self.Tolabels = {-1: min(labels), 1: max(labels)}
        self.y = np.array([self.Fromlabels[y_i] for y_i in labels]).reshape((-1, 1))

        # Initialisation of the vector of weights
        if len(w_init) == 0:
            self.w = np.zeros((np.shape(data_train)[1], 1))
        else:
            self.w = w_init

        # Saving of the previous iterations of the value of alpha and f
        self.histo_w = []
        self.histo_f = []

        # Peform the gradient descent
        self.gradient_decent()

    def predict(self, data_test, average_size=3):
        """Prediction of the label for the given data."""

        # Computation of the average w over the last average_size iteration
        w_predict = np.mean(self.histo_w[-average_size:, :], axis=0)
        w_predict = w_predict.reshape((-1, 1))

        # Prediction
        y_pred = np.where(dot(data_test, w_predict) > 0, 1, -1)
        labels_pred = np.array([self.Tolabels[y_i] for y_i in y_pred])
        labels_pred = labels_pred.reshape((-1, 1))

        return labels_pred

    def score(self, data_test, labels_test):
        """Compute the accuracy."""

        # Compute predictions
        y_pred = self.predict(data_test)

        return np.mean(y_pred == labels_test.reshape((-1, 1)))

    def loss(self, data):
        """Compute the loss."""

        # Compute Loss
        loss = np.log(1 + np.exp(-self.y * np.dot(data, self.w)))
        loss = np.mean(loss)
        loss += self.lamda * np.linalg.norm(self.w)

        return loss

    def grad_loss(self):
        """Compute the gradient of the loss."""

        # Compute gradient Loss
        grad = np.mean(sigmoid(- self.y * np.dot(self.X, self.w)) * (-self.y * self.X), axis=0)
        grad = grad.reshape((-1, 1))
        grad += 2 * self.lamda * self.w

        return grad.reshape((-1, 1))

    def hess_loss(self):
        """Compute the hessian of the loss."""

        # Shape X
        n = len(self.X)

        # Compute gradient Loss
        sigmoid_1 = sigmoid(- self.y * np.dot(self.X, self.w))
        sigmoid_2 = sigmoid(self.y * np.dot(self.X, self.w))
        sqrt = np.sqrt(sigmoid_1 * sigmoid_2)
        hess = 1 / n * np.dot((sqrt *  self.X).T, sqrt * self.X)
        hess += 2 * self.lamda

        return hess

    def gradient_decent(self):
        """Execution of the gradient descent"""

        # New iteration of the gradient
        if self.informations:
            print("\n----------")

        # Initialisation of the iterator and P
        ite = 0

        # Number of anchors points
        p_z = np.shape(self.X)[1]
        print("Fisrt loss: ", self.loss(self.X))
        print("First grad: ", np.linalg.norm(self.grad_loss()))
        while ite < self.max_iter:  # and np.linalg.norm(self.grad_loss()) > 10e-15:

            # Update w thanks to a Newton algorithm
            grad = self.grad_loss()
            hess = self.hess_loss()
            inv_hess = np.linalg.inv(hess + 10e-9 * np.identity(p_z))
            self.w = self.w + self.learning_rate * np.dot(inv_hess, grad)
            self.w = self.w.reshape((-1, 1))

            # Saving of the newly computed w
            self.histo_w.append(self.w)
            self.histo_f.append(self.loss(self.X))

            # Display the progress of the gradient descent
            if self.informations:  # ite % 10 == 0 and*
                norm = np.linalg.norm(grad)
                print("Iterations done: {}, Loss: {}, Gradient: {}".format(ite,
                                                                           self.histo_f[-1],
                                                                           norm))
                print("Norm of w", np.linalg.norm(self.w), np.shape(self.w))

            # Update of the iterator
            ite += 1

        # Update convert histo_f and histo_alpha as array
        self.histo_f = np.array(self.histo_f).reshape((-1, 1))
        self.histo_w = np.array(self.histo_w).reshape((-1, n))
