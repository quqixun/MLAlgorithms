# Conventional Machine Learning Algorithms
# Class of "KMeans".
# Author: Qixun Qu
# Create on: 2018/05/05
# Modify on: 2018/05/10

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

    def __init__(self, k, init="kmmeans++", tol=1e-4,
                 max_iters=1000, random_state=None):
        '''__INIT__

            Initialize an instance to do KMeans Cluster.

            Inputs:
            -------

            - k : int, the number of clusters.
            - init : string, one of ["random", "kmeans++"],
                     the method to initialize centers before
                     the first iteration, default is "kmeans++".
            - tol : float, the threshold to stop iteration.
                    If the distance between new centers and old
                    centers is smaller than this threshold, stop
                    running. Default is 1e-4.
            - max_iters : int, the maximum number of iterations.
            - random_state : int, seed for reproducing result.

        '''

        self.k = k                        # the number of clusters
        self.init = init                  # initialization method
        self.tol = tol                    # threshold to stop
        self.max_iters = max_iters        # maximum number of iterations
        self.random_state = random_state  # seed for reproducing results

        self.X = None             # training data
        self.cluster = None       # array of clusters
        self.centers = None       # array of new centers
        self.old_centers = None   # array of old centers in previous iteration
        self.init_centers = None  # array of initial centers

        return

    def _init_centers(self, X):
        '''_INIT_CENTERS

            Initialize centers by two methods:
            -1- randomly select points from dataset;
            -2- apply algorithm "kmeans++", see
                http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf
                for more information.

            Input:
            ------

            - X : numpy array of training data in shape
                  [n_samples, n_features].

            Output:
            -------

            - initial centers in shape [self.k, n_features].

        '''

        # Initialize
        centers = []

        if self.init == "random":
            # Randomply select unique points as initial centers
            init_idx = np.random.choice(len(X), self.k,
                                        replace=False)
            centers = X[init_idx, :]
        elif self.init == "kmeans++":
            # Apply kmeans++ to select initial centers
            # The first center is randomly selected
            first_idx = np.random.choice(len(X), 1)[0]
            centers.append(X[first_idx, :])
            # Select the other centers
            for i in range(1, self.k):
                # Compute the minimum distance between
                # the sample and the nearest center
                ds = np.array([min([np.linalg.norm(x - c) ** 2
                                    for c in centers])
                               for x in X])
                # Convert to probability
                probs = ds / np.sum(ds)
                # Compute comulative sum of probabilities
                # as weighted probabilitys
                cum_probs = np.cumsum(probs)
                # Choose one new data point at random as a
                # new center, using the weighted probabilities
                thresh = np.random.rand()
                idx = np.where(cum_probs >= thresh)[0][0]
                centers.append(X[idx, :])

        return np.array(centers)

    def _fit_initialize(self, X):
        '''_FIT_INITIALIZE
        '''

        self.X = X
        self.clusters = np.array([-1] * len(X))
        self.init_centers = self._init_centers(X)
        self.centers = np.copy(self.init_centers)
        self.old_centers = np.copy(self.centers)

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

    def plot_clusters(self, X=None, clusters=None,
                      axis_labels=None):
        '''PLOT_CLUSTERS

            Plot clustering result.

        '''

        if X is None and clusters is None:
            X, clusters = self.X, self.clusters

        n_features = X.shape[1]
        if (n_features != 2) and (n_features != 3):
            print("Only plot points in 2D or 3D.")
            return

        if axis_labels is None:
            axis_labels = ["Feature 1", "Feature 2"]
            if n_features == 3:
                axis_labels.append("Feature 3")

        fig = plt.figure(figsize=(12, 5))

        if n_features == 2:
            plt.subplot(121)
            plt.title("Initialization", fontsize=14)
            plt.plot(X[:, 0], X[:, 1], ".", ms=10, alpha=0.5)
            plt.plot(self.init_centers[:, 0], self.init_centers[:, 1], "r*",
                     ms=10, markeredgewidth=1, markeredgecolor="k")
            plt.xlabel(axis_labels[0], fontsize=14)
            plt.ylabel(axis_labels[1], fontsize=14)
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
            plt.xlabel(axis_labels[0], fontsize=14)
            plt.ylabel(axis_labels[1], fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
        else:  # n_features = 3
            ax1 = fig.add_subplot(121, projection='3d')
            plt.title("Initialization", fontsize=14)
            ax1.scatter(X[:, 0], X[:, 1], X[:, 2], s=10, alpha=0.5)
            ax1.scatter(self.init_centers[:, 0], self.init_centers[:, 1], self.init_centers[:, 2],
                        marker="*", c="r", s=100, edgecolor="k")
            ax1.set_xlabel(axis_labels[0], fontsize=14)
            ax1.set_ylabel(axis_labels[1], fontsize=14)
            ax1.set_zlabel(axis_labels[2], fontsize=14)
            ax2 = fig.add_subplot(122, projection='3d')
            plt.title("Clustering Result", fontsize=14)
            for k in range(self.k):
                c = cm.hsv(k / self.k)
                idx = np.where(clusters == k)
                ax2.scatter(X[idx, 0], X[idx, 1], X[idx, 2], c=c, s=10, alpha=0.5)
                ax2.scatter(self.centers[k, 0], self.centers[k, 1], self.centers[k, 2],
                            marker="*", c=c, s=100, edgecolor="k")
            ax2.set_xlabel(axis_labels[0], fontsize=14)
            ax2.set_ylabel(axis_labels[1], fontsize=14)
            ax2.set_zlabel(axis_labels[2], fontsize=14)

        plt.tight_layout()
        plt.show()

        return
