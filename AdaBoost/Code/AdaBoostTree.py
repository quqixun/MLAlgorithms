
# Conventional Machine Learning Algorithms
# Class of "AdaBoostTree".
# Author: Qixun Qu
# Create on: 2018/03/13
# Modify on: 2018/03/14

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


from __future__ import print_function


import copy
import numpy as np
from tqdm import *
import matplotlib.pyplot as plt


class AdaBoostTree(object):

    def __init__(self, M, clf):
        '''__INIT__

            Initialization of the object to indicate the
            number of iteration and the classifier.

            Inputs:
            -------

            - M : int, the number of iteration.
            - clf : object of sklearn.tree, the classifier.

        '''

        # Set the number of iteration
        self.M = M

        # Set the initial classifier
        self.init_clf = clf

        # A list to store the fitted classifier
        # of each iteration
        self.clfs = None

        # A list to store alpha of each iteration
        self.alphas = None

        # Lists to store error rates of training set
        # and test set in each iteration
        self.test_errs = None
        self.train_errs = None

        return

    def fit(self,
            X_train, Y_train,
            X_test, Y_test,
            verbose=True,
            vb_num=10):
        self.clfs = []
        self.alphas = []
        self.test_errs = []
        self.train_errs = []

        train_num, test_num = len(X_train), len(X_test)
        weights = np.ones(train_num) / train_num
        ws_pred_train = np.zeros(train_num)
        ws_pred_test = np.zeros(test_num)

        print("\nAdaBoostTree - {} Iterations".format(self.M))
        iter_range = range(self.M)
        if not verbose:
            iter_range = tqdm(iter_range)

        for m in iter_range:
            clf = copy.copy(self.init_clf)
            clf.fit(X_train, Y_train, sample_weight=weights)
            self.clfs.append(clf)

            Y_pred_train = clf.predict(X_train)
            Y_miss = [int(y) for y in (Y_pred_train != Y_train)]

            err = np.dot(weights, Y_miss)
            alpha = np.log((1 - err) / err) / 2.0
            self.alphas.append(alpha)

            exp = [np.exp(-1 * alpha * Y_train[i] * Y_pred_train[i])
                   for i in range(train_num)]
            Z = np.dot(weights, exp)
            weights = [w / Z * e for w, e in zip(weights, exp)]

            ws_pred_train, train_error_rate = \
                self._update_pred(ws_pred_train, Y_pred_train, Y_train, alpha)
            self.train_errs.append(train_error_rate)

            Y_pred_test = clf.predict(X_test)
            ws_pred_test, test_error_rate = \
                self._update_pred(ws_pred_test, Y_pred_test, Y_test, alpha)
            self.test_errs.append(test_error_rate)

            if (m + 1) % vb_num != 0 and m != 0:
                continue

            if verbose:
                print("Iteration {}".format(m + 1))
                self._print_metrics(train_error_rate, test_error_rate)

        if not verbose:
            self._print_metrics(train_error_rate, test_error_rate)

        return

    def predict(self, X):
        pred = np.zeros(len(X))
        for clf, alpha in zip(self.clfs, self.alphas):
            Y_pred = clf.predict(X)
            pred += alpha * Y_pred
        return np.sign(pred)

    def plot_curve(self):
        x = np.arange(self.M) + 1
        plt.figure()
        plt.plot(x, self.train_errs, label="Train")
        plt.plot(x, self.test_errs, label="Test")
        plt.xlim((0, self.M))
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylabel("Error Rate", fontsize=16)
        plt.xlabel("Iteration", fontsize=16)
        plt.legend(loc=1, fontsize=14)
        plt.grid("on", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.show()
        return

    def _update_pred(self, ws_pred, Y_pred, Y_true, alpha):
        ws_pred += alpha * Y_pred
        error_rate = self._get_error_rate(Y_true, np.sign(ws_pred))
        return ws_pred, error_rate

    def _get_error_rate(self, Y_true, Y_pred):
        return sum(Y_pred != Y_true) / float(len(Y_pred))

    def _print_metrics(self, train_error_rate, test_error_rate):
        print("Training Error Rate: {0:.6f}".format(train_error_rate),
              "Testing Error Rate: {0:.6f}".format(test_error_rate))
        return
