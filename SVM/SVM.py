# Conventional Machine Learning Algorithms
# Class of "SVM".
# Author: Qixun Qu
# Create on: 2018/03/23
# Modify on: 2018/03/24

#     ,,,         ,,,
#   ;"   ';     ;'   ",
#   ;  @.ss$$$$$$s.@  ;
#   `s$$$$$$$$$$$$$$$'
#   $$$$$$$$$$$$$$$$$$
#  $$$$P""Y$$$Y""W$$$$$
#  $$$$  p"$$$"q  $$$$$
#  $$$$  .$$$$$.  $$$$'
#   $$$DaU$$O$$DaU$$$'
#    '$$$$'.^.'$$$$'
#       '&$$$$$&'


import numpy as np


class SVM(object):

    def __init__(self, C=1.0,
                 kernel="rbf", degree=3,
                 gamma="auto", coef0=1.0,
                 tol=1e-3, max_iter=-1,
                 random_state=None):
        '''__INIT__

            Instance initialization.

        '''

        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state

        self.b = None
        self.F = None
        self.N = None
        self.alphas = None
        self.errors = None
        self.error_rates = None

        self.X = None
        self.y = None

        return

    def _initialize(self, X_train, y_train):
        '''_I

            Initialize variables to optimize SVM.

        '''

        self.X = X_train
        self.y = y_train

        self.b = 0.0
        self.N, self.F = X_train.shape
        self.alphas = np.zeros(self.N)
        self.errors = self._E(index="all")
        self.error_rates = []

        if self.gamma == "auto":
            self.gamma = 1.0 / self.F
        return

    def _E(self, index="all"):
        '''_E

            Computer error.

        '''

        if index == "all":
            X, y = self.X, self.y
        else:
            X, y = self.X[index], self.y[index]

        return self._G(X, y, X) - y

    def _G(self, X_train, y_train, X_test):
        '''_G

            Decision function

        '''

        pred = np.dot(self.alphas * y_train,
                      self._K(X_train, X_test)) + self.b

        return pred

    def _K(self, x1, x2):
        '''_K

            Compute features by kernels.

        '''

        if self.kernel == "linear":
            return np.dot(x1, x2.T)
        elif self.kernel == "poly":
            return (np.dot(x1, x2.T) + self.coef0) ** self.degree
        elif self.kernel == "sigmoid":
            return np.tanh(self.gamma * np.dot(x1, x2.T) + self.coef0)
        elif self.kernel == "rbf":
            deno = 2 * self.gamma ** 2
            x1_ndim, x2_ndim = np.ndim(x1), np.ndim(x2)
            if x1_ndim == 1 and x2_ndim == 1:
                return np.exp(-np.linalg.norm(x1 - x2) / deno)
            elif (x1_ndim > 1 and x2_ndim == 1) or \
                 (x1_ndim == 1 and x2_ndim > 1):
                return np.exp(-np.linalg.norm(x1 - x2, axis=1) / deno)
            elif x1_ndim > 1 and x2_ndim > 1:
                return np.exp(-np.linalg.norm(
                    x1[:, np.newaxis] - x2[np.newaxis, :], axis=2) / deno)
        else:
            print(self.kernel + " is not valid.")
            raise SystemExit
        return

    def fit(self, X_train, y_train,
            X_test, y_test,
            verbose=True,
            vb_num=10):

        '''FIT
        '''

        self._initialize(X_train, y_train)

        return
