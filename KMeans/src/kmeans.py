# Conventional Machine Learning Algorithms
# Class of "KMeans".
# Author: Qixun Qu
# Create on: 2018/05/05
# Modify on: 2018/05/07

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
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class KMeans(object):

    def __init__(self, k=None, init="kmmeans++", tol=1e-4,
                 max_iters=1000, random_state=None):
        '''__INIT__
        '''

        self.k = k
        self.init = init
        self.tol = tol
        self.max_iters = max_iters
        self.random_state = random_state

        self.X = None
        self.n_features = None
        self.init_centers = None
        self.centers = None
        self.old_centers = None
        self.cluster = None

        return

    def _init_centers(self, X):
        '''_INIT_CENTERS

            Initialize centers by two methods:
            -1- randomly select points from dataset;
            -2- apply algorithm "kmeans++".

        '''

        centers = []
        if self.init == "random":
            init_idx = np.random.choice(len(X), self.k,
                                        replace=False)
            centers = X[init_idx, :]
        elif self.init == "kmeans++":
            first_idx = np.random.choice(len(X), 1)[0]
            centers.append(X[first_idx, :])
            for i in range(1, self.k):
                ds = np.array([min([np.linalg.norm((x - c))
                                    for c in centers])
                               for x in X])
                probs = ds / np.sum(ds)
                cum_probs = np.cumsum(probs)
                thresh = np.random.rand()
                idx = np.where(cum_probs >= thresh)[0][0]
                centers.append(X[idx, :])

        return np.array(centers)

    def _fit_initialize(self, X):
        '''_FIT_INITIALIZE
        '''

        self.X = X
        self.n_features = X.shape[1]
        self.init_centers = self._init_centers(X)
        self.centers = np.copy(self.init_centers)
        self.old_centers = np.copy(self.centers)
        self.clusters = np.array([-1] * len(X))

        return

    def _should_stop(self):
        '''_SHOULD_STOP
        '''

        dist = np.linalg.norm(self.old_centers - self.centers)
        print("The distance between old centers and new centers: {0:.6f}".format(dist))
        if dist < self.tol:
            return True
        else:
            return False

    def _fit_cluster(self):
        '''_FIT_CLUSTER
        '''

        for i in range(self.max_iters):
            print("Iteration {}:".format(i))
            for j, x in enumerate(self.X):
                xds = [np.linalg.norm(x - c) for c in self.old_centers]
                self.clusters[j] = xds.index(min(xds))
            for k in range(self.k):
                idx = np.where(self.clusters == k)
                self.centers[k] = np.mean(self.X[idx, :], axis=1)
            if self._should_stop():
                break
            self.old_centers = np.copy(self.centers)

        return

    def fit(self, X):
        '''FIT
        '''

        np.random.seed(seed=self.random_state)
        self._fit_initialize(X)
        self._fit_cluster()

        return

    def predict(self, X):
        '''PREDICT
        '''

        clusters = np.array([-1] * len(X))
        for j, x in enumerate(X):
            xds = [np.linalg.norm(x - c) for c in self.old_centers]
            clusters[j] = xds.index(min(xds))

        return clusters

    def plot_clusters(self, X=None, clusters=None):
        '''PLOT_CLUSTERS
        '''

        if (self.n_features != 2) and (self.n_features != 3):
            print("Only plot points in 2D or 3D.")
            return

        if X is None and clusters is None:
            X, clusters = self.X, self.clusters

        fig = plt.figure(figsize=(12, 5))

        if self.n_features == 2:
            plt.subplot(121)
            plt.title("Initialization", fontsize=14)
            plt.plot(X[:, 0], X[:, 1], ".", ms=10, alpha=0.5)
            plt.plot(self.init_centers[:, 0], self.init_centers[:, 1], "r*",
                     ms=10, markeredgewidth=1, markeredgecolor="k")
            plt.xlabel("Feature 1", fontsize=14)
            plt.ylabel("Feature 2", fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.subplot(122)
            plt.title("Clustering Result", fontsize=14)
            for k in range(self.k):
                c = cm.hsv(k / self.k)
                idx = np.where(clusters == k)
                plt.plot(X[idx, 0], X[idx, 1], ".", ms=10, alpha=0.5, color=c)
                plt.plot(self.centers[k, 0], self.centers[k, 1], "*", ms=10,
                         color=c, markeredgewidth=1, markeredgecolor="k")
            plt.xlabel("Feature 1", fontsize=14)
            plt.ylabel("Feature 2", fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
        else:  # self.n_features = 3
            ax1 = fig.add_subplot(121, projection='3d')
            plt.title("Initialization", fontsize=14)
            ax1.scatter(X[:, 0], X[:, 1], X[:, 2], s=10, alpha=0.5)
            ax1.scatter(self.init_centers[:, 0], self.init_centers[:, 1], self.init_centers[:, 2],
                        marker="*", c="r", s=100, edgecolor="k")
            ax1.set_xlabel("Feature 1", fontsize=14)
            ax1.set_ylabel("Feature 2", fontsize=14)
            ax1.set_zlabel("Feature 3", fontsize=14)
            ax2 = fig.add_subplot(122, projection='3d')
            plt.title("Clustering Result", fontsize=14)
            for k in range(self.k):
                c = cm.hsv(k / self.k)
                idx = np.where(clusters == k)
                ax2.scatter(X[idx, 0], X[idx, 1], X[idx, 2], c=c, s=10, alpha=0.5)
                ax2.scatter(self.centers[k, 0], self.centers[k, 1], self.centers[k, 2],
                            marker="*", c=c, s=100, edgecolor="k")
            ax2.set_xlabel("Feature 1", fontsize=14)
            ax2.set_ylabel("Feature 2", fontsize=14)
            ax2.set_zlabel("Feature 3", fontsize=14)

        plt.tight_layout()
        plt.show()
