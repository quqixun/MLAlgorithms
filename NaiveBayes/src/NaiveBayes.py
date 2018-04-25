# Conventional Machine Learning Algorithms
# Class of "NaiveBayes".
# Author: Qixun Qu
# Create on: 2018/04/23
# Modify on: 2018/04/25

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
        self.labels = list(set(y))

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

        for i in range(self.F):
            Xf = X[:, i]
            feat_dict = {}
            if i not in self.cont_feat_idx:
                feat_set = set(Xf)
                set_len = len(feat_set)
                for f in feat_set:
                    f_dict = {}
                    for l in self.labels:
                        f_dict[l] = ((len(np.where((Xf == f) & (y == l))[0]) + self.lb) /
                                     (len(np.where(y == l)[0]) + set_len * self.lb))
                    feat_dict[f] = f_dict
            else:
                for l in self.labels:
                    l_dict = {}
                    l_dict["mu"] = np.mean(Xf[y == l])
                    l_dict["sigma"] = np.std(Xf[y == l])
                    feat_dict[l] = l_dict

            self.cond_probs[i] = feat_dict

        return

    def _compute_post_probs(self, X):
        '''_COMPUTE_POST_PROBS
        '''

        for i in range(len(X)):
            i_dict = {}
            for l in self.labels:
                post_prob = 1
                for j in range(self.F):
                    if j not in self.cont_feat_idx:
                        post_prob *= self.cond_probs[j][X[i, j]][l]
                    else:
                        pass
                i_dict[l] = post_prob
            self.post_probs[i] = i_dict

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
        result = []
        for sd in self.post_probs:
            post_probs = self.post_probs[sd]
            preds = [post_probs[l] for l in self.labels]
            idx = preds.index(max(preds))
            result.append(self.labels[idx])

        return np.array(result)
