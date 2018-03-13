import numpy as np
import matplotlib.pyplot as plt


class AdaBoostTree(object):

    def __init__(self, M, clf):
        self.M = M
        self.clf = clf
        self.clfs = None
        self.errs = None
        self.alphas = None

        return

    def fit(self, X_train, Y_train):
        self.clfs = []
        self.errs = []
        self.alphas = []

        num = len(X_train)
        weights = np.ones(num) / num
        pred_train = np.zeros(num)

        for m in range(self.M):
            self.clf.fit(X_train, Y_train,
                         sample_weight=weights)
            self.clfs.append(self.clf)

            Y_pred = self.clf.predict(X_train)
            Y_miss = [int(y) for y in (Y_pred != Y_train)]

            err = np.dot(weights, Y_miss)
            self.errors.append(err)

            alpha = np.log((1 - err) / err) / 2.0
            self.alphas.append(alpha)

            exp = [np.exp(-1 * alpha * Y_train[i] * Y_pred[i]) for i in range(num)]
            Z = np.dot(weights, exp)
            weights = [w / Z * e for w, e in zip(weights, exp)]

            pred_train += alpha * pred_train

        return

    def predict(self, X_test, Y_test):
        return

    def plot_curve(self):
        return
