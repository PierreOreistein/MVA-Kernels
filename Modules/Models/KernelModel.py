import numpy as np


class KernelModel:
        def __init__(self, kernel=None, informations=True, preprocessing=None):
            """Abstract Model Class"""

            # Definition of the name of the model
            self.name = "KernelModel"

            # Instantiation of the kernel
            self.kernel = kernel

            # Function to preprocess the data
            self.preprocessing = preprocessing

            # Parameter of the gradient Descent
            self.informations = informations

        def score(self, data_test, labels_test):
            """Compute the accuracy."""

            return np.mean(self.predict(data_test) == labels_test.reshape((-1, 1)))

        def fit(self, data_train, labels, alpha_init=[]):
            """Fitting of the model."""
            pass

        def predict(self, data_test, average_size=3):
            """Prediction of the label for the given data."""
            pass

        def predict_proba(self, data_test, average_size=3):
            """Predict the probability to belong to the class 1."""
            pass

        def loss(self):
            """Compute the loss."""
            pass

        def grad_loss(self):
            """Compute the gradient of the loss."""
            pass
