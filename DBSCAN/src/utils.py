# Conventional Machine Learning Algorithms
# Helper functions for Class of "KMeans".
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


import os
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

        - X : float array in shape [n_samples, n_features],
              features array of all input samples.
        - y : float list in shape [n_samples, ], labels list
              of all input samples.
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
