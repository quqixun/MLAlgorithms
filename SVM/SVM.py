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
                 kernel="rbf", degree=2,
                 sigma="auto", coef0=1.0,
                 tol=1e-3, epsilon=1e-3,
                 random_state=None):
        '''__INIT__

            Instance initialization.

        '''

        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.sigma = sigma
        self.coef0 = coef0
        self.tol = tol
        self.epsilon = epsilon
        self.random_state = random_state

        self.b = None
        self.F = None
        self.N = None
        self.alphas = None
        self.E = None

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

        if self.sigma == "auto":
            self.sigma = 1.0 / self.F

        self.alphas = np.zeros(self.N)
        self.E = self._E()

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
        obj = np.sum(alphas) - 0.5 * np.sum(ao * ko * yo)
        return obj

    def _E(self, index=None):
        '''_E

            Compute error.

        '''

        if index is None:
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
            X = self.X[index]
        else:
            if X is None:
                X = self.X

        pred = np.dot(self.alphas * self.y,
                      self._K(self.X, X)) - self.b

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
            return np.tanh(self.sigma * np.dot(x1, x2.T) + self.coef0)
        elif self.kernel == "rbf":
            deno = 2 * (self.sigma ** 2)
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

        if i1 == i2:
            return 0

        a1_old = self.alphas[i1]
        a2_old = self.alphas[i2]
        y1, y2 = self.y[i1], self.y[i2]
        x1, x2 = self.X[i1], self.X[i2]
        E1, E2 = self.E[i1], self.E[i2]
        s = y1 * y2

        if y1 == y2:
            L = max(0, a2_old + a1_old - self.C)
            H = min(self.C, a2_old + a1_old)
        else:
            L = max(0, a2_old - a1_old)
            H = min(self.C, self.C + a2_old - a1_old)

        if L == H:
            return 0

        k11 = self._K(x1, x1)
        k12 = self._K(x1, x2)
        k22 = self._K(x2, x2)
        eta = 2 * k12 - k11 - k22

        if eta < 0:
            a2_new = a2_old - y2 * (E1 - E2) / eta
            if a2_new > H:
                a2_new = H
            elif a2_new < L:
                a2_new = L
        else:
            alphas_temp = np.copy(self.alphas)
            alphas_temp[i2] = L
            L_obj = self._O(alphas=alphas_temp)
            alphas_temp[i2] = H
            H_obj = self._O(alphas=alphas_temp)

            if L_obj > (H_obj + self.epsilon):
                a2_new = L
            elif L_obj < (H_obj - self.epsilon):
                a2_new = H
            else:
                a2_new = a2_old

        if a2_new < 1e-8 and a2_new > 0:
            a2_new = 0.0
        elif a2_new > (self.C - 1e-8) and a2_new < self.C:
            a2_new = self.C

        if (np.abs(a2_new - a2_old) <
           self.epsilon * (a2_new + a2_old + self.epsilon)):
            return 0

        a1_new = a1_old + s * (a2_old - a2_new)

        b1_new = (self.b + E1 +
                  y1 * k11 * (a1_new - a1_old) +
                  y2 * k12 * (a2_new - a2_old))
        b2_new = (self.b + E2 +
                  y1 * k12 * (a1_new - a1_old) +
                  y2 * k22 * (a2_new - a2_old))

        if a1_new > 0 and a1_new < self.C:
            b_new = b1_new
        elif a2_new > 0 and a2_new < self.C:
            b_new = b2_new
        else:
            b_new = (b1_new + b2_new) / 2.0

        self.alphas[i1] = a1_new
        self.alphas[i2] = a2_new

        self.E = self._E()

        if 0.0 < a1_new < self.C:
            self.E[i1] = 0.0

        if 0.0 < a2_new < self.C:
            self.E[i2] = 0.0

        # self.E[i1] = self._E(i1)
        # self.E[i2] = self._E(i2)
        # self.E = self._E()

        # non_opt = [i for i in range(self.N) if i != i1 and i != i2]
        # self.E[non_opt] = self.E[non_opt] + \
        #     y1 * (a1_new - a1_old) * self._K(x1, self.X[non_opt]) + \
        #     y2 * (a2_new - a2_old) * self._K(x2, self.X[non_opt]) + \
        #     self.b - b_new

        self.b = b_new

        return 1

    def _examine_example(self, i2):

        y2 = self.y[i2]
        a2 = self.alphas[i2]
        E2 = self.E[i2]
        r2 = E2 * y2

        if ((r2 < -self.tol and a2 < self.C) or
           (r2 > self.tol and a2 > 0)):
            n0nC_list = np.where((self.alphas != 0) &
                                 (self.alphas != self.C))[0]
            if len(n0nC_list) > 1:
                if self.E[i2] > 0:
                    i1 = np.argmin(self.E)
                else:
                    i1 = np.argmax(self.E)
                if self._take_step(i1, i2):
                    return 1

            rnd_n0nC_list = np.random.permutation(n0nC_list)
            for i1 in rnd_n0nC_list:
                if self._take_step(i1, i2):
                    return 1

            rnd_all_list = np.random.permutation(self.N)
            for i1 in rnd_all_list:
                if self._take_step(i1, i2):
                    return 1
        return 0

    def fit(self, X_train, y_train):

        '''FIT
        '''

        self._initialize(X_train, y_train)

        num_changed = 0
        examine_all = 1

        np.random.seed(self.random_state)
        while num_changed > 0 or examine_all:
            num_changed = 0
            if examine_all:
                for i2 in range(self.N):
                    num_changed += self._examine_example(i2)
            else:
                i2_list = np.where((self.alphas != 0) &
                                   (self.alphas != self.C))[0]
                for i2 in i2_list:
                    num_changed += self._examine_example(i2)

            if examine_all == 1:
                examine_all = 0
            elif num_changed == 0:
                examine_all = 1

        return

    def predict(self, X_test):
        pred = self._G(X=X_test)
        pred = (pred >= 0) * 2 - 1
        return pred
