import numpy as np
from Modules.Models.KernelModel import KernelModel


class KernelSVM(KernelModel):
    def __init__(self, kernel=None,
                 informations=True, lamda=1.0, max_iter=10000, tol=1e-6,
                 preprocessing=None, normalisation=None):
        # Instantiation of the super class
        super().__init__(informations=informations, kernel=kernel)

        # Name of the class
        self.name = "KernelSVM"

        # Hyperparameter for the regulariser in the optimisation
        self.lamba = lamda

        # Gradient Descent parameters
        self.max_iter = int(max_iter)
        self.tol = tol
        self.proj_cst = 0.1
        self.preprocessing = preprocessing
        self.normalisation = normalisation

    def fit(self, data_train, labels, alpha_init=[]):
        """
        ASSUMPTION: CONSTANT LEARNING RATE!!!!

        :param data_train:
        :param labels:
        :param alpha_init:
        :return:
        """

        # Computation of K
        self.K_train_func = lambda data: self.kernel.call(data_train, data)
        self.K_train = self.K_train_func(data_train)

        # Preprocessing of K_train
        if self.preprocessing is not(None):
            self.K_train = self.preprocessing(self.K_train)
        if self.normalisation is not(None):
            self.K_train = self.normalisation(self.K_train)

        if self.informations:
            print('K train mean: {}; std: {}'.format(self.K_train.mean(), self.K_train.std()))

        # Bijection between values of y and {-1, 1}
        self.Fromlabels = {min(labels): -1, max(labels): 1}
        self.Tolabels = {-1: min(labels), 1: max(labels)}
        self.y_train = np.array([self.Fromlabels[y_i] for y_i in labels]).reshape((-1, 1))

        # Initialisation of alpha
        self.n_examples = len(self.K_train)
        if alpha_init == []:
            self.alpha = np.zeros(self.n_examples)
        else:
            self.alpha = np.array(alpha_init)

        # Initialisation of alpha_t-1 and ite
        ite = 0
        self.alpha_previous = 100 * np.ones(self.n_examples)

        # Condition to stop the gradient descent
        diff_alphas = np.abs(self.alpha.reshape(-1) -\
                             self.alpha_previous.reshape(-1)).max()

        # Projected gradient procedure
        while (ite < self.max_iter) and (diff_alphas < self.tol) :

            # Compute the gradient
            grad = self.gradient()
            grad_norm = np.abs(grad).max()

            # Compute the current loss
            loss = self.trainLoss()

            # Solve best alpha and project them
            invert = np.identity(self.n_examples)
            try:
                alpha_star = np.linalg.solve(
                    a=self.K_train + 1e-7 * invert,
                    b=self.y_train.reshape(-1))
            except np.linalg.LinAlgError:
                print(self.K_train.mean(), self.K_train.std())
                raise
            proj_alpha = self.project_alpha(alpha_star)

            # alpha_n = proj_y_n
            self.alpha_previous = self.alpha
            self.alpha = self.alpha + self.proj_cst * (proj_alpha.reshape(-1) -\
                                                       self.alpha)

            if self.informations and iter % 1 == 0:
                try:
                    print("Iterations done: {}, Loss: {:.4f}, Gradient: {}".format(iter,
                                                                                   loss,
                                                                                   grad_norm))
                except:
                    print(loss, grad_norm)

        self.alpha = self.project_alpha(self.alpha)
        return self

    def project_alpha(self, alpha):
        """
        Projects alpha such that alpha is admissible:

        0.0 <= y_i x alpha_i <= 1.0 / (2 x lamda x n)

        :param alpha:
        :return:
        """

        # Compute y_i x \alpha_i
        prod_sign = np.sign(alpha.reshape(-1) * self.y_train.reshape(-1))
        alpha_sign = np.sign(alpha.reshape(-1))

        # Useless operation but shows that necessarily proj_alpha=0
        # a_max = 0.1 * np.zeros(self.n_examples)
        # a_max[prod_sign < 0.0] = 0.0
        #
        # # Clip to margin C
        # a_max[prod_sign > 0.0] = 1.0 / (2.0 * self.lamba * self.n_examples)

        a_min = np.zeros(self.n_examples)
        a_max = np.zeros(self.n_examples)

        a_min[(prod_sign > 0.0) & (alpha_sign > 0.0)] = 0.0
        a_max[(prod_sign > 0.0) & (alpha_sign > 0.0)] = 1.0 / (2.0 * self.lamba * self.n_examples)

        a_min[(prod_sign > 0.0) & (alpha_sign < 0.0)] = -1.0 / (2.0 * self.lamba * self.n_examples)
        a_max[(prod_sign > 0.0) & (alpha_sign < 0.0)] = 0.0

        return np.clip(alpha, a_min=a_min, a_max=a_max)

    def trainLoss(self):
        """Compute the loss for the training set."""
        return self.alpha.T.dot(self.y_train) -\
                0.5 * self.alpha.T.dot(self.K_train.dot(self.alpha))

    def gradient(self):
        """Compute the gradient for the current self.alpha."""
        return self.K_train.dot(self.alpha) - self.y_train

    def predict(self, data_test, average_size=3):
        """Predict a class for dataset."""

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

        # Extract alpha_predict
        alpha_predict = self.alpha.reshape(-1)

        # Prediction
        y_pred = np.where(alpha_predict.T.dot(K_pre) >= 0.0, 1, -1)
        label_pred = np.array([self.Tolabels[y_i] for y_i in y_pred])
        label_pred = label_pred.reshape((-1, 1))

        return label_pred
