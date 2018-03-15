# Conventional Machine Learning Algorithms
# Class of "AdaBoostTree".
# Author: Qixun Qu
# Create on: 2018/03/13
# Modify on: 2018/03/16

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
        '''FIT

            Build the classifier based on input data.
            In each iteration, fitted classifier and alpha will be
            stored for predicting new data; error rates of training
            set and test set will be stored for plotting learning
            curves.

            Inputs:
            -------

            - X_train : array with shape [n_samples, n_features],
                        features array of training data.
            - Y_train : array with shape [n_samples],
                        labels array of training data.
            - X_test : array with shape [n_samples, n_features],
                       features array of test data.
            - Y_test : array with shape [n_samples],
                       labels array of test data.
            - verbose : boolean, the symbol to determine whether
                        show logs while training process.
            - vb_num: int, the interval of iteration to display logs.

        '''

        # Initialize variables as empty list
        self.clfs = []
        self.alphas = []
        self.test_errs = []
        self.train_errs = []

        # Obtain the number of samples in both
        # training set and test set
        train_num, test_num = len(X_train), len(X_test)

        # Initialize weights of samples
        # In 1st iteration, all samples in training set
        # share the same weight
        weights = np.ones(train_num) / train_num

        # Initialize arrays to store and update predictions
        # of training set and test set in each iteration
        ws_pred_train = np.zeros(train_num)
        ws_pred_test = np.zeros(test_num)

        print("\nAdaBoostTree - {} Iterations".format(self.M))

        # Create an iterable object according to
        # the number of iteration
        iter_range = range(self.M)
        if not verbose:
            # A progress bar would be shown if no verbose
            iter_range = tqdm(iter_range)

        for m in iter_range:
            # Create a new tree classifier by copying
            # the initial classifier
            clf = copy.copy(self.init_clf)

            # Train the classifier based on training set
            clf.fit(X_train, Y_train, sample_weight=weights)
            # Add the fitted classifier into the list of classifiers
            self.clfs.append(clf)

            # Compute the prediction of training set
            Y_pred_train = clf.predict(X_train)
            # Obtain a binary list that indicates which labels in
            # prediction are not euqal to the truth. In Y_miss,
            # 1 means "miss" (not equal), 0 means "hit" (equal)
            Y_miss = [int(y) for y in (Y_pred_train != Y_train)]

            # Compute error as Equation (1)
            err = np.dot(weights, Y_miss)
            # Compute alpha as Equation (2)
            alpha = np.log((1 - err) / err) / 2.0
            # Add alpha into the list of alphas
            self.alphas.append(alpha)

            # Compute exponential part in Equations (4) and (5)
            exp = [np.exp(-1 * alpha * Y_train[i] * Y_pred_train[i])
                   for i in range(train_num)]
            # Compute the regularization term as Equation (5)
            Z = np.dot(weights, exp)
            # Upate weights for next iteration as Equation (3)
            weights = [w / Z * e for w, e in zip(weights, exp)]

            # Update prediction of training set
            # and compute error rate
            ws_pred_train, train_error_rate = \
                self._update_pred(ws_pred_train, Y_pred_train, Y_train, alpha)
            self.train_errs.append(train_error_rate)

            # Predict labels for test set
            Y_pred_test = clf.predict(X_test)
            # Update prediction of test set
            # and compute error rate
            ws_pred_test, test_error_rate = \
                self._update_pred(ws_pred_test, Y_pred_test, Y_test, alpha)
            self.test_errs.append(test_error_rate)

            if verbose:
                # Determine whether print logs according to vb_num
                if (m + 1) % vb_num != 0 and m != 0:
                    continue

                # Print logs of training iteration
                print("Iteration {}".format(m + 1))
                self._print_metrics(train_error_rate, test_error_rate)

        if not verbose:
            # Print logs of las iteration
            self._print_metrics(train_error_rate, test_error_rate)

        return

    def predict(self, X):
        '''PREDICT

            Predict labels for the input features array.

            Input:
            ------

            - x : array with shape [n_samples, n_features],
                  features array of samples to be predicted.

            Output:
            -------

            - predicted labels with shape [n_samples]

        '''

        # Initialize an array to store and update prediction
        pred = np.zeros(len(X))

        # In ith iteration, do prediction by the ith classifier
        # and update prediction by the ith alpha
        for clf, alpha in zip(self.clfs, self.alphas):
            pred += alpha * clf.predict(X)

        # Compute and return the binary prediction
        return np.sign(pred)

    def plot_curve(self):
        '''PLOT_CURVES

            Plot error rates of training set
            and test set of each iteration.

        '''

        # Generate coordinate of X axis
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
        '''_UPDATE_PRED

            Private method to update predictions in each
            iteration of training process. Also, get error
            rate of each iteration.

            Inputs:
            -------

            - ws_pred : array with shape [n_samples],
                        old predictions updated from (i-1)th iteration.
            - Y_pred : array with shape [n_samples],
                       new predictions of ith iteration.
            - Y_true : array with shape [n_samples], ground truth.
            - alpha : float, computed in ith iteration.

            Outputs:
            --------

            - ws_pred : updated predictions of ith iteration.
            - error_rate : error rate of ith iteration.

        '''

        # Update predictions as Equation (6)
        ws_pred += alpha * Y_pred
        # Compute error rate of final classifier
        # in ith iteration as Euqation (7)
        error_rate = self._get_error_rate(Y_true, np.sign(ws_pred))

        return ws_pred, error_rate

    def _get_error_rate(self, Y_true, Y_pred):
        '''_GET_ERROR_RATE

            Private method to compute error rate.

            Inputs:
            -------

            - Y_true : array with shape [n_samples], ground truth.
            - Y_pred : array with shape [n_samples],
                       new predictions of ith iteration.

        '''

        # Compute and return error rate
        return sum(Y_pred != Y_true) / float(len(Y_pred))

    def _print_metrics(self, train_error_rate, test_error_rate):
        '''_PRINT_METRICS

            Private method to print logs while training process.

            Inputs:
            -------

            - train_error_rate : float, error rate of training set.
            - test_error_rate : float, error rate of test set.

        '''

        print("Training Error Rate: {0:.6f}".format(train_error_rate),
              "Testing Error Rate: {0:.6f}".format(test_error_rate))

        return
