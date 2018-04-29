# Conventional Machine Learning Algorithms
# Class of "kNN".
# Author: Qixun Qu
# Create on: 2018/04/26
# Modify on: 2018/04/30

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

    def __init__(self, data, label, depth,
                 left=None, right=None):
        self.data = data
        self.label = label
        self.depth = depth
        self.left = left
        self.right = right
        self.parent = None
        self._set_parent()
        return

    def _set_parent(self):
        if self.left is not None:
            self.left.parent = self
        if self.right is not None:
            self.right.parent = self
        return


class kNN(object):

    def __init__(self, k=10):
        '''__INIT__
        '''

        self.k = k
        self.kdtree_root = None

        self.X = self.y = None
        self.N = self.F = None
        self.tag = self.dis = None

        return

    def _initialize(self, X, y):
        '''_INITIALIZE
        '''

        self.X, self.y = X, y
        self.N, self.F = X.shape
        self.tag = []
        self.dis = []

        return

    def _build_kdtree(self, X=None, y=None, depth=0):
        '''_BUILD_KDTREE
        '''

        if X is None:
            X = self.X.copy()

        if y is None:
            y = self.y.copy()

        if len(X) == 0:
            return None
        elif len(X) == 1:
            return Node(data=X[0], label=y[0], depth=depth)
        else:
            fth = depth % self.F
            X = X[np.argsort(X[:, fth]), :]
            m = len(X) // 2
            return Node(data=X[m], label=y[m], depth=fth,
                        left=self._build_kdtree(X[:m], y[:m], depth + 1),
                        right=self._build_kdtree(X[m + 1:], y[m + 1:], depth + 1))

    def _distance(self, x1, x2):
        '''_DISTANCE
        '''

        return np.sum((x1 - x2) ** 2)

    def _search_leaf(self, x):
        '''_SEARCH_LEAF
        '''

        depth = 0
        leaf, node = self.kdtree_root, None
        while ((leaf.left is not None) or
               (leaf.right is not None)):
            fth = depth % self.F
            if x[fth] < leaf.data[fth]:
                node = leaf.left
            else:  # x[fth] >= leaf.data[fth]
                node = leaf.right
            if node is None:
                break
            else:
                leaf = node
                depth += 1

        return leaf, self._distance(x, leaf.data)

    def _get_brother(self, node):
        '''_GET_BROTHER
        '''

        if node == node.parent.left:
            return node.parent.right
        else:
            return node.parent.left

    def _get_vote(self):
        '''_GET_VOTE
        '''

        if self.k > len(self.tag):
            self.k = len(self.tag)

        idx = np.argsort(self.dis)[:self.k]
        self.dis = [self.dis[d] for d in idx]
        self.tag = [self.tag[d] for d in idx]

        tag_set = list(set(self.y))
        tag_num = [self.tag.count(t) for t in tag_set]
        # print(len(self.tag), tag_set, tag_num)

        self.dis, self.tag = [], []

        return tag_num.index(max(tag_num))

    def _search_kdtree(self, x):
        '''_SEARCH_KDTREE
        '''

        node, radius = self._search_leaf(x)
        while node is not None:
            dis = self._distance(x, node.data)
            if dis <= radius:
                self.dis.append(dis)
                self.tag.append(node.label)

            pnode = node.parent
            if pnode is not None:
                if (x[pnode.depth] - pnode.data[pnode.depth]) ** 2 <= radius:
                    bro = self._get_brother(node)
                    if bro is not None:
                        bro_dis = self._distance(x, bro.data)
                        self.dis.append(bro_dis)
                        self.tag.append(bro.label)
            node = pnode

        return self._get_vote()

    def fit(self, X, y):
        '''FIT
        '''

        print("Building KDTree ... ")
        self._initialize(X, y)
        self.kdtree_root = self._build_kdtree()

        return

    def predict(self, X):
        '''PREDICT
        '''

        print("Predicting ...")
        pred = []
        for x in X:
            pred.append(self._search_kdtree(x))

        return np.array(pred)
