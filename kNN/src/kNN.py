# Conventional Machine Learning Algorithms
# Class of "kNN".
# Author: Qixun Qu
# Create on: 2018/04/26
# Modify on: 2018/04/28

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


class Node(object):

    def __init__(self, data,
                 left=None, right=None):
        self.data = data
        self.left = left
        self.right = right
        return

    def get_left(self):
        return self.left

    def get_right(self):
        return self.right


class kNN(object):

    def __init__(self, k=10):
        '''__INIT__
        '''

        self.k = k
        self.kd_tree = None

        self.X = None
        self.y = None

        self.N = None
        self.F = None

        return

    def _initialize(self, X, y):
        '''_INITIALIZE
        '''

        self.X = X
        self.y = y

        self.N, self.F = X.shape

        return

    def _build_kd_tree(self, X=None, depth=0):
        '''_BUILD_KD_TREE
        '''

        if X is None:
            X = self.X

        if len(X) == 0:
            return None
        elif len(X) == 1:
            return Node(data=X[0])
        else:
            fth = depth % self.F
            X = X[np.argsort(X[:, fth]), :]
            m = len(X) // 2
            return Node(data=X[m],
                        left=self._build_kd_tree(X[:m], depth + 1),
                        right=self._build_kd_tree(X[m + 1:], depth + 1))

    def _search_kd_tree(self, x):
        '''_SEARCH_KD_TREE
        '''

        return

    def fit(self, X, y):
        '''FIT
        '''

        self._initialize(X, y)
        self.kdtree = self._build_kd_tree()

        return

    def predict(self, X):
        '''PREDICT
        '''

        return


def main():
    """Example usage"""
    X = np.array([(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)])
    knn = kNN()
    knn.fit(X, None)
    print(knn.kdtree.data)
    # tree = kdtree(point_list)
    # print(tree)


if __name__ == '__main__':
    main()
