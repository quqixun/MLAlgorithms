# Conventional Machine Learning Algorithms
# Class of "SVM".
# Author: Qixun Qu
# Create on: 2018/03/23
# Modify on: 2018/03/31

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


class SVC(object):

    def __init__(self, C=1.0,
                 kernel="rbf", degree=3,
                 gamma="auto", coef0=1.0,
                 tol=1e-3, max_iter=-1,
                 epsilon=1e-3,
                 random_state=None,
                 verbose=True,
                 vb_num=10):
        '''__INIT__

            Instance initialization.

        '''

        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.random_state = random_state

        self.b = None
        self.F = None
        self.N = None
        self.alphas = None
        self.E = None
        self.Obj = None

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

        if self.gamma == "auto":
            self.gamma = 1.0 / self.F

        self.alphas = np.zeros(self.N)
        self.E = self._E()
        self.Obj = self._O()

        return

    def _O(self, alphas=None):
        '''_O

            Objective function.

        '''

        if alphas is None:
            alphas = self.alphas
        ao = alphas ** 2
        yo = self.y ** 2
        ko = self._K(self.X, self.X)
        obj = 0.5 * np.sum(ao * yo * ko) - np.sum(alphas)
        return obj

    def _E(self, index="all"):
        '''_E

            Compute error.

        '''

        if index == "all":
            X, y = self.X, self.y
        else:
            X, y = self.X[index], self.y[index]

        return self._G(X=X) - y
        # return loss

    def _G(self, index=None, X=None):
        '''_G

            Decision function

        '''

        if index is not None:
            if index == "all":
                X = self.X
            else:
                X = self.X[index]
        else:
            if X is None:
                X = self.X

        pred = np.dot(self.alphas * self.y,
                      self._K(self.X, X)) + self.b

        return pred

    def _K(self, x1, x2):
        '''_K

            Kernel functions.

        '''

        if self.kernel == "linear":
            return np.dot(x1, x2.T) + self.coef0
        elif self.kernel == "poly":
            return (np.dot(x1, x2.T) + self.coef0) ** self.degree
        elif self.kernel == "sigmoid":
            return np.tanh(self.gamma * np.dot(x1, x2.T) + self.coef0)
        elif self.kernel == "rbf":
            deno = 2 * (self.gamma ** 2)
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

    def _take_step(self, i1, i2):
        a1_old = self.alphas[i1]
        a2_old = self.alphas[i2]
        y1, y2 = self.y[i1], self.y[i2]
        x1, x2 = self.X[i1], self.X[i2]

        if y1 == y2:
            L = max(0, a2_old + a1_old - self.C)
            H = min(self.C, a2_old + a1_old)
        else:
            L = max(0, a2_old - a1_old)
            H = min(self.C, self.C + a2_old - a1_old)

        E1, E2 = self.E[i1], self.E[i2]
        eta = self._K(x1, x1) + self._K(x2, x2) - 2 * self._K(x1, x2)
        a2_new_unc = y2 * (E1 - E2) / eta + a2_old

        if a2_new_unc > H:
            a2_new = H
        elif a2_new_unc < L:
            a2_new = L
        else:
            a2_new = a2_new_unc

        a1_new = y1 * y2 * (a2_old - a2_new) + a1_old
        self.alphas[i1] = a1_new
        self.alphas[i2] = a2_new

        b1_new = self.b - E1 - \
            y1 * self._K(x1, x1) * (a1_new - a1_old) - \
            y2 * self._K(x2, x1) * (a2_new - a2_old)
        b2_new = self.b - E2 - \
            y1 * self._K(x1, x2) * (a1_new - a1_old) - \
            y2 * self._K(x2, x2) * (a2_new - a2_old)

        if ((a1_new > 0) and (a1_new < self.C)):
            self.b = b1_new
        elif ((a2_new > 0) and (a2_new < self.C)):
            self.b = b2_new
        else:
            self.b = (b1_new + b2_new) / 2.0

        self.E[i1] = self._E(i1)
        self.E[i2] = self._E(i2)

        print(self.alphas, self.b)
        return

    def _examine_example(self, i2):
        return

    def fit(self, X_train, y_train):

        '''FIT
        '''

        self._initialize(X_train, y_train)

        return
