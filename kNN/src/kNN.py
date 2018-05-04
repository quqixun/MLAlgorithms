# Conventional Machine Learning Algorithms
# Class of "kNN".
# Author: Qixun Qu
# Create on: 2018/04/26
# Modify on: 2018/05/04

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


class KDNode(object):

    def __init__(self, data, label,
                 left=None, right=None):
        '''__INIT__

            Initialization of kd-tree node.
            Each node has four variables:
            - data : features of one sample
            - label : label of one sample
            - left : left child
            - right : right child

            Inputs:
            -------

            - data : float ndarray, features of one sample.
            - label : int, label of one sample.
            - left : KDNode instance or None, left child.
            - right : KDNode instance or None, right child.

        '''

        self.data = data
        self.label = label
        self.left = left
        self.right = right

        return

    def is_leaf(self):
        '''IS_LEAF

            Check if the node is leaf, return True or False.

        '''

        return self.left is None and self.right is None


class kNN(object):

    def __init__(self, k=None):
        '''__INIT__

            Initialize a kNN classifier.

            Input:
            ------

            - k : int or None, the number of
                  nearest neighbors.

        '''

        # The number of nearest neighbors
        self.k = k
        # The root node of kd-tree
        self.kdtree_root = None

        # Training data, features and labels
        self.X, self.y = None, None
        # N is the number of taining samples,
        # F is the number of features
        self.N, self.F = None, None

        # The minimum distance which will be
        # updated while searching the kd-tree
        self.radius = None
        # The sample to by query in kd-tree
        self.query = None
        # A list stores k nearest nerghbors
        self.neighbors = None

        return

    def set_k(self, k):
        '''SET_K

            Set the number of nearest neighbors.

            Input:
            ------

            k : int, the number of nearest neighbors.

        '''

        self.k = k
        return

    def _fit_initialize(self, X, y):
        '''_FIT_INITIALIZE

            Initialize some variables before building
            the kd-tree using training samples.

            Inputs:
            -------

            X : float ndarray in shape [n_samples, n_features],
                the feature array of training samples.
            y : float ndarray in shape [n_samples, ],
                the labels of training samples.

        '''

        # Set training data
        self.X, self.y = X, y
        # Set the number of data,
        # set the number of features
        self.N, self.F = X.shape

        return

    def _build_kdtree(self, X=None, y=None, depth=0):
        '''_BUILD_KDTREE

            Build kd-tree using training data recursively.

                            Root Node      ---> Depth: 0
                               /\
                             /   \
                           /      \
                         /         \
                    Left Node  Right Node  ---> Depth: 1
                         :         :
                         :         :
                         /\
                       /   \
                     /      \
                   /         \
              Leaf Node  Leaf Node         ---> Depth: N

              Inputs:
              -------

            - X : float ndarray in shape [n_samples, n_features],
                  the number of samples is different while
                  recursively building the tree.
            - y : float ndarray in shape [n_samples, ],
                  the label list of feature X.
            - depth : int, the depth of tree, that decides which
                      feature is used to split branch.

            Output:
            -------

            - A KDNode instance.

        '''

        if X is None:
            # Iniatially use all training
            # samples' features
            X = self.X.copy()

        if y is None:
            # Initially use all training
            # samples' labels
            y = self.y.copy()

        if len(X) == 0:
            # No input data
            return None
        elif len(X) == 1:
            # Leaf node with None child node
            return KDNode(data=X[0], label=y[0])
        else:  # Other nodes
            # Get the index of feature used in one depth
            fth = depth % self.F
            # Sort the feature from low to high
            idx = np.argsort(X[:, fth])
            # Reorder the training data
            X, y = X[idx, :], y[idx]
            # Get the index of median value
            m = len(X) // 2
            # Increase depth to make tree deeper
            depth += 1
            # Generate the node
            return KDNode(data=X[m], label=y[m],
                          left=self._build_kdtree(X[:m], y[:m], depth),
                          right=self._build_kdtree(X[m + 1:], y[m + 1:], depth))

    def fit(self, X, y):
        '''FIT

            Build kd-tree using training data.

            Inputs:
            -------

            - X : float ndarray in shape [n_samples, n_features],
                  features array of training data.
            - y : float ndarray in shape [n_samples, ],
                  the label list of feature X.

        '''

        print("Building KDTree ... ")
        # Initialize variables used in fitting process
        self._fit_initialize(X, y)
        # Build the kd-tree
        self.kdtree_root = self._build_kdtree()

        return

    def _predict_initialize(self, x):
        '''_PREDICT_INITALIZE

            Initialize some variables before
            searching one sample in kd-tree.

            Input:
            ------

            - x : float ndarray in shape [n_features, ],
                  the feature list of one sample to be queryed
                  in kd-tree.

        '''

        # The minimum distance
        self.radius = np.inf
        # Queryed sample
        self.query = x
        # List of nearest nerghbors
        # In self.neighbors, each sample has
        # two elements: [node, dist],
        # dist is the distance between the node
        # and the query sample
        self.neighbors = []

        return

    def _distance(self, x1, x2):
        '''_DISTANCE

            Compute the square distance between
            two given sampels.

            Inputs:
            -------

            - x1, x2 : float ndarry or float number,
                       x1 and x2 have same dimensions.

            Output:
            -------

            - The square distance between x1 and x2.

        '''

        return np.sum((x1 - x2) ** 2)

    def _get_radius(self):
        '''_GET_RADIUS

            Obtain the largest distance between the query point
            and the last nearest neighbor in self.neighbors.
            In self.neighbors, the last nearest point is the
            last element.

        '''

        idx = -1 if self.k >= len(self.neighbors) else self.k - 1
        self.radius = self.neighbors[idx][1]

        return

    def _add_neighbor(self, node):
        '''_ADD_NEIGHBOR

            Add new point into self.neighbors and
            set self.radius with the largest distance.

            Import:
            -------

            - node : KDNode instance, the node to be added
                     into self.neighbors.

        '''

        # Compute the distance between the given node
        # and the query point
        dist = self._distance(self.query, node.data)

        # If self.neighbors is not empty
        for i, n in enumerate(self.neighbors):
            if i == self.k:
                # If k nearest neighbors has been found
                # the node will not be saved
                return
            if dist < n[1]:
                # Insert the new point into self.neighbors
                # by the order from low to high
                self.neighbors.insert(i, [node, dist])
                self._get_radius()
                return

        # If the self.neighbors is empth
        self.neighbors.append([node, dist])
        self._get_radius()

        return

    def _search_kdtree(self, node, depth=0):
        '''_SEARCH_KDTREE

            Searh a point in kd-tree recursively.
            First, get the leaf node. Then, back to
            the parent node and check if the brother
            branch should be searched. Stop when node
            is the root of the kd-tree.

            Inputs:
            -------

            - node : KDNode instance, the root node of
                     any branch or the root of kd-tree.
            - depth : int, the depth of tree, that decides which
                      feature should be compared.

        '''

        if node is None:
            # Node is not exist
            return

        if node.is_leaf():
            # Add leaf node to the list of neighbors
            self._add_neighbor(node)
            return

        # Get the index of feature used in one depth
        fth = depth % self.F

        # Initialize child nodes
        # nearer_node is the the root of the branch
        # which has the closest leaf node
        # further_node is the root of the brother branch
        # of the the nearer one
        nearer_node = None
        further_node = None

        # Get child nodes according the rule to build kd-tree
        if self.query[fth] < node.data[fth]:
            nearer_node = node.left
            further_node = node.right
        else:
            nearer_node = node.right
            further_node = node.left

        # Search in the nearer branch to get leaf node
        self._search_kdtree(nearer_node, depth + 1)
        # Add visited node in nearer branch to
        # trhe list of nerghbors
        self._add_neighbor(node)

        # If the split plane intersets with the
        # minimum hyper-sphere, the nearest point
        # maybe in the brother branch
        if (self.query[fth] - node.data[fth]) ** 2 < self.radius:
            # Search neighbors in braother branch
            self._search_kdtree(further_node, depth + 1)

        return

    def _get_vote(self):
        '''_GET_VOTE

            Apply k-nearest neighbors to vote the prediction
            of one sample. Obrain k labels of k points firstly.
            Count the number of each type of label and get the
            label which has most points.

            Output:
            -------

            - result : int, predicted label of one sample.

        '''

        # If k is larger than the number of all
        # neighbors, k will be reset
        if self.k > len(self.neighbors):
            self.k = len(self.neighbors)

        # Get labels which appear in neighbors
        labels = [n[0].label for n in self.neighbors[:self.k]]
        # Get labels appear in training data
        label_set = list(set(self.y))
        # Count the number of each type of label
        label_num = [labels.count(l) for l in label_set]
        # Get the label that has most points
        retult = label_num.index(max(label_num))

        return retult

    def predict(self, X):
        '''PREDICT

            Query input samples in kd-tree to kind
            k-nearest neighbors, which vote the final result.

            Input:
            ------

            - X : float ndarray in shape [n_samples, n_features],
                  features array of test samples.

            Output:
            -------

            - Predictions in shape [n_samples, ].

        '''

        print("Predicting (k = {}) ...".format(self.k))
        pred = []
        for x in X:
            # For each sample in test set
            # Initialize variables
            self._predict_initialize(x)
            # Find k-nearest neighbors
            self._search_kdtree(self.kdtree_root, depth=0)
            # Vote to get final prediction
            pred.append(self._get_vote())

        return np.array(pred)
