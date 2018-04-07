# Conventional Machine Learning Algorithms
# Class of "SVC".
# Author: Qixun Qu
# Create on: 2018/03/23
# Modify on: 2018/04/07

# References:
# [1] Sequential Minimal Optimization:
#     A Fast Algorithm for Training Support Vector Machines.
#     John C. Platt, 1998. See the pseudo-code in Section 2.5.
# [2] Implementing a Support Vector Machine using
#     Sequential Minimal Optimization and Python 3.5.
#     From: http://jonchar.net/notebooks/SVM/
#     I copy the code to generate RBF kernel.

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


from __future__ import division
from __future__ import print_function


import numpy as np


class SVC(object):

    def __init__(self, C=1.0,
                 kernel="rbf", degree=2,
                 gamma="auto", coef0=1.0,
                 tol=1e-3, epsilon=1e-3,
                 random_state=None):
        '''__INIT__

            Instance initialization.

            Inputs:
            -------

            - C : float or int, default is 1.0, penalty parameter.
            - kernel : string, default is "rbf", kernel type,
                       select one kernel among
                       ["linear", "poly", "sigmoid", "rbf"].
            - degree : int, default is 2, degree of the polynomial
                       kernel function (kernel = "poly").
            - gamma : float, default is "auto", parameter for RBF
                      and sigmoid kernel function. If gamma is "auto",
                      gamma equals to 1 / n_features.
            - coef0 : float or int, default is 1.0, bias term for
                      linear, polynomial and sigmoid kernel functions.
            - tol : float, default is 1e-3, the tolerance of inaccuracies
                    around the KKT conditions.
            - epslion : float, default is 1e-3, tolerance for updating
                        Lagrange muliplier.
            - randome_state : int or None, default is None, seed for
                              reproducing results.

        '''

        # Initialize parameters
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.epsilon = epsilon
        self.random_state = random_state

        # Declear other variables that would
        # be used in optimization process
        self.X = None       # Features array of training samples
        self.y = None       # Labels of training samples

        self.b = None       # Threshold
        self.F = None       # Number of features
        self.N = None       # Number of training samples
        self.E = None       # Errors cache
        self.alphas = None  # Lagrange multipliers

        return

    def _initialize(self, X_train, y_train):
        '''_initialize

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

        return

    def _O(self, alphas=None):
        '''_O

            Objective function.

        '''

        if alphas is None:
            alphas = self.alphas

        obj = np.sum(alphas) - 0.5 * np.sum(alphas ** 2 *
                                            self._K(self.X, self.X) *
                                            self.y ** 2)
        return obj

    def _E(self):
        '''_E

            Compute error.

        '''

        return self._G(X=self.X) - self.y

    def _G(self, X=None):
        '''_G

            Decision function

        '''

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

        if i1 == i2:
            return 0

        a1_old = self.alphas[i1]
        a2_old = self.alphas[i2]
        y1, y2 = self.y[i1], self.y[i2]
        x1, x2 = self.X[i1], self.X[i2]
        E1, E2 = self.E[i1], self.E[i2]
        s = y1 * y2

        # Compute L and H
        if y1 == y2:
            L = max(0, a2_old + a1_old - self.C)
            H = min(self.C, a2_old + a1_old)
        else:
            L = max(0, a2_old - a1_old)
            H = min(self.C, self.C + a2_old - a1_old)

        if L == H:
            return 0

        # Compute eta
        k11 = self._K(x1, x1)
        k12 = self._K(x1, x2)
        k22 = self._K(x2, x2)
        eta = k11 + k22 - 2 * k12

        if eta > 0:
            a2_new = a2_old + y2 * (E1 - E2) / eta
            if a2_new > H:
                a2_new = H
            elif a2_new < L:
                a2_new = L
        else:
            alphas_temp = np.copy(self.alphas)
            alphas_temp[i2] = L
            # Objective function at a2=L
            L_obj = self._O(alphas=alphas_temp)
            alphas_temp[i2] = H
            # Objective function at a2=H
            H_obj = self._O(alphas=alphas_temp)

            if L_obj < (H_obj - self.epsilon):
                a2_new = L
            elif L_obj > (H_obj + self.epsilon):
                a2_new = H
            else:
                a2_new = a2_old

        if (np.abs(a2_new - a2_old) <
           self.epsilon * (a2_new + a2_old + self.epsilon)):
            return 0

        a1_new = a1_old + s * (a2_old - a2_new)

        # Update threshold
        b1_new = (self.b + E1 +
                  y1 * k11 * (a1_new - a1_old) +
                  y2 * k12 * (a2_new - a2_old))
        b2_new = (self.b + E2 +
                  y1 * k12 * (a1_new - a1_old) +
                  y2 * k22 * (a2_new - a2_old))

        if 0 < a1_new < self.C:
            self.b = b1_new
        elif 0 < a2_new < self.C:
            self.b = b2_new
        else:
            self.b = (b1_new + b2_new) / 2.0

        # Store new Lagrange multipliers
        self.alphas[i1] = a1_new
        self.alphas[i2] = a2_new

        # Update error cache using new alphas
        self.E = self._E()

        return 1

    def _examine_example(self, i2):

        y2 = self.y[i2]
        a2 = self.alphas[i2]
        E2 = self.E[i2]
        r2 = E2 * y2

        if ((r2 < -self.tol and a2 < self.C) or
           (r2 > self.tol and a2 > 0)):
            # Indices of examples which is not 0 and not C
            n0nC_list = np.where((self.alphas != 0) &
                                 (self.alphas != self.C))[0]
            if len(n0nC_list) > 1:
                if self.E[i2] > 0:
                    i1 = np.argmin(self.E)
                else:
                    i1 = np.argmax(self.E)
                if self._take_step(i1, i2):
                    return 1

            # Loop over all non-0 and non-C alpha,
            # starting at a random point
            rnd_n0nC_list = np.random.permutation(n0nC_list)
            for i1 in rnd_n0nC_list:
                if self._take_step(i1, i2):
                    return 1

            # Loop over all possible i1, starting at a random point
            rnd_all_list = np.random.permutation(self.N)
            for i1 in rnd_all_list:
                if self._take_step(i1, i2):
                    return 1
        return 0

    def fit(self, X_train, y_train):

        '''FIT
        '''

        self._initialize(X_train, y_train)

        # Main routine
        num_changed = 0
        examine_all = 1

        np.random.seed(self.random_state)
        while num_changed > 0 or examine_all:
            num_changed = 0
            if examine_all:
                # Loop i2 over all training examples
                for i2 in range(self.N):
                    num_changed += self._examine_example(i2)
            else:
                # Loop i2 over examples where alpha
                # is not 0 and not C
                i2_list = np.where((self.alphas != 0) &
                                   (self.alphas != self.C))[0]
                for i2 in i2_list:
                    num_changed += self._examine_example(i2)

            if examine_all == 1:
                examine_all = 0
            elif num_changed == 0:
                examine_all = 1

        return

    def predict(self, X_test, sign=True):
        pred = self._G(X=X_test)
        if sign:
            pred = np.sign(pred)
        return pred
