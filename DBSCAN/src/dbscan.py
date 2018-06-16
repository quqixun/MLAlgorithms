# Conventional Machine Learning Algorithms
# Class of "DBSCAN".
# Author: Qixun Qu
# Create on: 2018/06/13
# Modify on: 2018/06/16

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

    def __init__(self, epsilon, min_samples):
        '''__INIT__

            Initialize an instance of DBSCAN clustering.

            Inputs:
            -------

            - epsilon: float, maximum radius of the neighborhood.
            - min_samples: int, the number of samples in a neighborhood
                           to be core points.

        '''

        self.epsilon = epsilon
        self.min_samples = min_samples

        self.X = None         # training data
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

            Obtain neighboring points of given point.

            Input:
            ------

            - point: numpy ndarray, the query point.

            Output:
            -------

            - neighbors: int list, contains indices of
                         neighboring points.

        '''

        # Compute Euclidean distance
        def distance(p1, p2):
            return np.linalg.norm(p1 - p2)

        # Find all neighbors
        neighbors = []
        for i in range(len(self.X)):
            if distance(self.X[i], point) <= self.epsilon:
                neighbors.append(i)

        return neighbors

    def _fit_cluster(self):
        '''_FIT_CLUSTER

            Fit DBSCAN clustering.
            https://en.wikipedia.org/wiki/DBSCAN

        '''

        print("Runing DBSCAN ...")

        # Cluster counter
        C = 0

        for p in range(len(self.X)):
            # Obtain query point
            point1 = self.X[p]

            # The point has been processed
            # in privious inner loop
            if self.clusters[p] != -1:
                continue

            # Find neighbors of query point
            N1 = self._find_neighbors(point1)
            # If the number of neighbors is less
            # than the threshold, the point is Noise
            if len(N1) < self.min_samples:
                self.clusters[p] = 0
                continue

            # Next cluster label
            C += 1
            # Label the query point
            self.clusters[p] = C

            # Neighbors of query point
            # except itself, known as seed set
            N1.pop(N1.index(p))

            # Inner loop to label all possible
            # cluster points, beginning from the query point
            q = 0
            while q < len(N1):
                # Set seed point
                i = N1[q]
                point2 = self.X[i]

                # Change Noise to boeder point
                if self.clusters[i] == 0:
                    self.clusters[i] == C
                # If the point has not been processed
                elif self.clusters[i] == -1:
                    # Label the point
                    self.clusters[i] = C
                    # Find neighbors of this point
                    N2 = self._find_neighbors(point2)
                    # If the point is core point
                    if len(N2) >= self.min_samples:
                        # Add new neighbors to seed set
                        N2.pop(N2.index(i))
                        N1 = N1 + N2
                q += 1

        print("Done")
        return

    def fit_predict(self, X):
        '''FIT

            Cluster given data by K-means.

            Input:
            ------

            - X : numpy array of training data in shape
                  [n_samples, n_features].

        '''

        # Initialize variables
        self._fit_initialize(X)
        # Apply DBSCAN clustering
        self._fit_cluster()

        return
