# Conventional Machine Learning Algorithms
# Helper functions for Class of "NaiveBayes".
# Author: Qixun Qu
# Create on: 2018/04/25
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


from __future__ import print_function

import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Ignore the warning caused by StandardScaler
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)


# Generate path of data directory
PARENT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
DATA_DIR = os.path.join(PARENT_DIR, "Data")


def split_dataset(X, y, test_size=0.2, random_state=None):
    '''SPLIT_DATASET

        Split dataset into train set and test set according to
        the given proportion of test set.

        Inputs:
        -------

        - X : numpy ndarray in shape [n_samples, n_features],
              features array of all input samples.
        - y : numpy ndarray or list in shape [n_samples, ],
              labels list of all input samples.
        - test_size : float, default is 0.2, the proportion
                      of test set among all samples.
        - random_state : int, seed for reproducing the split.

        Outputs:
        --------

        - X_train : features array of train set in shape
                    [n_train_samples, n_features].
        - y_train : labels list of train set in shape
                    [n_train_samples, ].
        - X_test : features array of test set in shape
                   [n_test_samples, n_features].
        - y_test : labels list of test set in shape
                   [n_test_samples, ].

    '''

    # Split dataset into train set and test set
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, stratify=y, test_size=0.2,
                         random_state=random_state)

    return X_train, y_train, X_test, y_test


def scale_dataset(X_train, X_test=None):
    '''SCALE_DATASET

        Normalize each feature of the dataset
        by subtracting feature's mean
        and deviding features's standard deviation (std).

        Inputs:
        -------

        - X_train : numpy ndarray array of training samples
                    in shape [n_train_samples, n_features].
        - X_test : optional, numpy ndarray array of testing
                   samples in shape [n_test_samples, n_features],
                   default is None.

        Outputs:
        --------

        - X_train_scaled : normalized training samples.
        - X_test_scaled : normalized testing samples
                          if X_test is not none.

    '''

    # Normalize train set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_test is None:
        return X_train_scaled
    else:
        # Normalize test set by
        # the mean and std values of train set
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled


def accuracy(y_pred, y_true):
    '''ACCURACY

        Compute accuracy of prediction.

        Inputs:
        -------

        - y_pred : Labels predicted by the model in shape [n_samples, ].
        - y_true : Real labels of dataset in shpe [n_samples, ].

        Output:
        -------

        - float, the accuray of prediction.

    '''

    return np.mean((y_pred == y_true) * 1.0)
