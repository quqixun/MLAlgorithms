# Conventional Machine Learning Algorithms
# Class of "NaiveBayes".
# Author: Qixun Qu
# Create on: 2018/04/23
# Modify on: 2018/04/24

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


class NaiveBayes(object):

    def __init__(self, lb=1):
        '''__INIT__
        '''

        self.lb = lb

        self.N = None
        self.F = None

        self.labels = None
        self.features = None

        self.prior_probs = None
        self.cond_probs = None
        self.post_probs = None

        self.cont_feat_idx = []
        self.cont_feat_mu = None
        self.cont_feat_sigma = None

        return

    def _initialize(self, X, y, cont_feat_idx):
        '''_INITIALIZE
        '''

        self.N, self.F = X.shape
        self.labels = set(y)
        # self.features = [set(f) for f in map(list, zip(*X))]

        self.prior_probs = {}
        self.cond_probs = {}
        self.post_probs = {}

        self.cont_feat_idx = cont_feat_idx

        # print(self.N, self.F, self.labels)

        return

    def _compute_prior_probs(self, y):
        '''_COMPUTE_PRIOR_PROBS
        '''

        for label in self.labels:
            prob = ((len(np.where(y == label)[0]) + self.lb) /
                    (self.N + len(self.labels) * self.lb))
            self.prior_probs[label] = prob

        # print(self.prior_probs)

        return

    def _compute_cond_probs(self, X, y):
        '''_COMPUTE_COND_PROBS
        '''

        self.features = {}

        return

    def _compute_post_probs(self, X):
        '''_COMPUTE_POST_PROBS
        '''

        return

    def fit(self, X, y, cont_feat_idx=[]):
        '''FIT
        '''

        self._initialize(X, y, cont_feat_idx)
        self._compute_prior_probs(y)
        self._compute_cond_probs(X, y)

        return

    def predict(self, X):
        '''PREDICT
        '''

        self._compute_post_probs(X)

        return
