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

    def __init__(self, alpha=1.0):
        '''__INIT__

            Inialization of NaiveBayes Instance.

            Input:
            ------

            - alpha : float, additive smoothing parameter.
                      If it is negative, it will be set to 0.

        '''

        # The additive smoothing parameter
        self.alpha = alpha if alpha >= 0 else 0

        # Number of training sampels
        self.N = None
        # Number of features
        self.F = None

        # A list contains all labels
        self.labels = None

        # Three dictionaries that contain
        # prior probabilities
        # conditional probabilities
        # post probabilities
        self.prior_probs = None
        self.cond_probs = None
        self.post_probs = None

        # List indicates the index of feature
        # which is continuous
        self.cont_feat_idx = None

        return

    def _initialize(self, X, y, cont_feat_idx):
        '''_INITIALIZE

            Initialzie instance variables.

            Inputs:
            -------

            - X : numpy ndarray in shape [n_samples, n_features],
                  features array of training samples.
            - y : numpy ndarray in shape [n_samples, ],
                  labels list of training samples.
            - cont_feat_idx : int list or None or "all", default
                              if an empty list. It indicates the
                              indices of features which are continuous.
                              "all" means all features are continuous;
                              None means all features are discrete.

        '''

        # Set the number of samples
        # and the number of features
        self.N, self.F = X.shape

        # Extract all unique labels
        self.labels = list(set(y))

        # Create empty dictionaries
        self.prior_probs = {}
        self.cond_probs = {}
        self.post_probs = {}

        if cont_feat_idx == "all":
            # All features are continuous
            self.cont_feat_idx = list(range(self.F))
        elif cont_feat_idx is None:
            # All features are discrete
            self.cont_feat_idx = []
        else:
            # Continuous features are indicated
            # in the given list
            self.cont_feat_idx = cont_feat_idx

        return

    def _compute_prior_probs(self, y):
        '''_COMPUTE_PRIOR_PROBS

            Compute prior probabilities from
            labels of training samples.

            Input:
            ------

            - y : numpy ndarray in shape [n_samples, ],
                  the labels list of training samples.

            Result:
            -------

            The formation of self.prior_probs:
            {
                label-1: prob-1,
                   :
                lebel-n: prob-n
            }

        '''

        for label in self.labels:
            # For each label, compute probability
            # as Equation 1
            prob = ((len(np.where(y == label)[0]) + self.alpha) /
                    (self.N + len(self.labels) * self.alpha))
            self.prior_probs[label] = prob

        return

    def _compute_cond_probs(self, X, y):
        '''_COMPUTE_COND_PROBS

            Compute conditional probabilities.

            Inputs:
            -------

            - X : numpy ndarray in shape [n_samples, n_features],
                  features array of training samples.
            - y : numpy ndarray in shape [n_samples, ],
                  labels list of training samples.

            Result:
            -------

            The formation of self.cond_probs:
            {
                feat-1: { >--------------------|
                    group-1: {                 |
                        label-1: prob-111,     |
                           :                   |
                        label-n: prob-11n      |
                    },                         |
                      :                        |-->  Discrete Feature
                    group-n: {                 |
                        label-1: prob-1n1,     |
                           :                   |
                        label-n: prob-1nn      |
                    }                          |
                } >----------------------------|
                  :
                feat-n: { >--------------------|
                    label-1: {                 |
                        mu: feature's mean,    |
                        sigma: feature's std   |
                    },                         |
                      :                        |--> Continuous Feature
                    label-n: {                 |
                        mu: feature's mean,    |
                        sigma: feature's std   |
                    }                          |
                } >----------------------------|
            }

        '''

        for i in range(self.F):
            # Extract one feature over all samples
            Xf = X[:, i]
            feat_dict = {}
            if i not in self.cont_feat_idx:
                # The feature is discrete
                # Get unique groups
                feat_set = set(Xf)
                set_len = len(feat_set)
                for f in feat_set:
                    f_dict = {}
                    for l in self.labels:
                        # Compute conditianl probability as Equation 2
                        f_dict[l] = ((len(np.where((Xf == f) & (y == l))[0]) + self.alpha) /
                                     (len(np.where(y == l)[0]) + set_len * self.alpha))
                    feat_dict[f] = f_dict
            else:
                # The feature is continuous
                for l in self.labels:
                    l_dict = {}
                    # Estimate parameters of Gaussian distribution
                    # from training samples
                    l_dict["mu"] = np.mean(Xf[y == l])
                    l_dict["sigma"] = np.std(Xf[y == l])
                    feat_dict[l] = l_dict

            self.cond_probs[i] = feat_dict

        return

    def _compute_post_probs(self, X):
        '''_COMPUTE_POST_PROBS

            Compute post probabilities of given dataset.

            Input:
            ------

            - X : numpy ndarray in shape [n_samples, n_features],
                  feature array of test set to be predicted.

            Result:
            -------

            The formation of self.post_probs:
            {
                sample-1: {
                    label-1: prob-11,
                      :
                    leble-n: prob-1n
                },
                  :
                sample-n: {
                    label-1: prob-n1,
                    label-n: prob-nn
                }
            }

        '''

        for i in range(len(X)):
            # For each sample in test set
            i_dict = {}
            for l in self.labels:
                post_prob = 1
                for j in range(self.F):
                    # For each feature of one sample
                    if j not in self.cont_feat_idx:
                        # Compute as Equation 3
                        # if the feature is discrete
                        post_prob *= self.cond_probs[j][X[i, j]][l]
                    else:
                        mu = self.cond_probs[j][l]["mu"]
                        sigma = self.cond_probs[j][l]["sigma"]
                        # Compute as Equation 5
                        # if the feature is continuous
                        post_prob *= (np.exp(-(X[i, j] - mu) ** 2 / (2 * sigma ** 2)) /
                                      (np.sqrt(2 * np.pi * sigma ** 2)))
                i_dict[l] = post_prob

            self.post_probs[i] = i_dict

        return

    def fit(self, X, y, cont_feat_idx=[]):
        '''FIT

            Training Naive Bayes Classifier by the given data.
            Two steps are contained:
            -1- Compute prior-probabilities
            -2- Compute conditional probabilities

            Inputs:
            -------

            - X : numpy ndarray in shape [n_samples, n_features],
                  features array of training samples.
            - y : numpy ndarray in shape [n_samples, ],
                  labels list of training samples.
            - cont_feat_idx : int list or None or "all", default
                              if an empty list. It indicates the
                              indices of features which are continuous.
                              "all" means all features are continuous;
                              None means all features are discrete.

        '''

        print("Fitting Naive Bayes Classifier ...")
        self._initialize(X, y, cont_feat_idx)
        self._compute_prior_probs(y)
        self._compute_cond_probs(X, y)

        return

    def predict(self, X):
        '''PREDICT

            Make predictoins on given dataset.

            Input:
            ------

            - X : numpy ndarray in shape [n_samples, n_features],
                  feature array to be predicted.

            Output:
            -------

            - numpy ndarray in shape [n_samples, ],
              classification results of given samples.

        '''

        # Compute post-probabilities
        self._compute_post_probs(X)

        # Get the calssification result of each sample
        # by finding the maximum value among post-probs
        # of one sample
        result = []
        for i in self.post_probs:
            preds = list(self.post_probs[i].values())
            idx = preds.index(max(preds))
            result.append(self.labels[idx])

        return np.array(result)
