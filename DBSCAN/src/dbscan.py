# Conventional Machine Learning Algorithms
# Class of "DBSCAN".
# Author: Qixun Qu
# Create on: 2018/06/13
# Modify on: 2018/06/13

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


class DBSCAN(object):

    def __init__(self, epsilon, min_samples, random_state=None):
        '''__INIT__

            Initialize an instance of DBSCAN clustering.

            Inputs:
            -------

            - epsilon: float, maximum radius of the neighborhood.
            - min_samples: int, the number of samples in a neighborhood
                           to be core points.
            - random_state : int, seed for reproducing results.

        '''

        self.epsilon = epsilon
        self.min_samples = min_samples
        self.random_state = random_state

        self.X = None        # training data
        self.clusters = None  # array of clusters

        return

    def _fit_initialize(self, X):
        '''_FIT_INITIALIZE

            Initialize variables before fitting.

            Input:
            ------

            - X : numpy array of training data in shape
                  [n_samples, n_features].

        '''

        # Training data
        self.X = X
        # Assign each sample's cluster to -1
        # -1 means undefined
        # 0 means noise
        self.clusters = np.array([-1] * len(X))

        return

    def _find_neighbors(self, point):
        '''_FIND_NEIGHBORS
        '''

        def distance(p1, p2):
            return np.linalg.norm(p1 - p2)

        neighbors = []
        for i in range(len(self.X)):
            if distance(self.X[i], point) <= self.epsilon:
                neighbors.append(i)

        return neighbors

    def _fit_cluster(self):
        '''_FIT_CLUSTER

            Fit DBSCAN clustering.

        '''

        # Cluster counter
        C = 0

        for p in range(len(self.X)):
            point1 = self.X[p]

            if self.clusters[p] != -1:
                continue

            N1 = self._find_neighbors(point1)
            if len(N1) < self.min_samples:
                self.clusters[p] = 0
                continue

            C += 1
            self.clusters[p] = C

            N1.pop(N1.index(p))
            q = 0
            while q < len(N1):
                i = N1[q]
                point2 = self.X[i]
                if self.clusters[i] == 0:
                    self.clusters[i] == C
                elif self.clusters[i] == -1:
                    self.clusters[i] = C
                    N2 = self._find_neighbors(point2)
                    if len(N2) >= self.min_samples:
                        N2.pop(N2.index(i))
                        N1 = N1 + N2
                q += 1

        return

    def fit_predict(self, X):
        '''FIT

            Cluster given data by K-means.

            Input:
            ------

            - X : numpy array of training data in shape
                  [n_samples, n_features].

        '''

        # Set numpy seed for reproducing results
        np.random.seed(seed=self.random_state)
        # Initialize variables
        self._fit_initialize(X)
        # Apply DBSCAN clustering
        self._fit_cluster()

        return
