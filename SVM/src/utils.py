# Conventional Machine Learning Algorithms
# Helper functions for Class of "SVM".
# Author: Qixun Qu
# Create on: 2018/04/02
# Modify on: 2018/04/10

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
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs, make_circles, make_moons


# Ignore the warning caused by StandardScaler
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)


# Generate path of data directory
PARENT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
DATA_DIR = os.path.join(PARENT_DIR, "Data")


def generate_dataset(style="blob", n_samples=500, random_state=None):
    '''GENERATE_DATASET

        Generate dataset to test SVM.
        You may get three kinds of dataset through this function. Look at
        http://scikit-learn.org/stable/datasets/index.html#sample-generators
        for mare infomation.

        Inputs:
        -------

        - style : string, default is "blob", select one from
                  ["blob", "circle", "moon"] to generate different dataset.
        - n_samples : int, default is 500, the number of samples.
        - random_state: int or None, default is NOne,
                        the seed for reproducing dataset.

        Outputs:
        --------

        - X : float array in shape [n_samples, 2], features array.
        - y : float list in shape [n_samples, ], labels list.

    '''

    if style == "blob":
        # Two-cluster samples with two features
        X, y = make_blobs(n_samples=n_samples, centers=2,
                          n_features=2, random_state=random_state)
    elif style == "circle":
        # A larger circle outside, a smaller circle inside,
        # each sample has two features
        X, y = make_circles(n_samples=n_samples, noise=0.1,
                            factor=0.1, random_state=random_state)
    elif style == "moon":
        # Two interleaving half circles in 2D
        X, y = make_moons(n_samples=n_samples, noise=0.1,
                          random_state=random_state)

    # Change the negative sample's label from 0 to -1
    y[y == 0] = -1
    return X, y


def split_dataset(X, y, test_size=0.2, random_state=None):
    '''SPLIT_DATASET

        Split dataset into train set and test set according to
        the given proportion of test set.

        Inputs:
        -------

        - X : float array in shape [n_samples, n_features],
              features arrya of all input samples.
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

    # Concatenate features and labels to one array
    data = np.concatenate([X, np.reshape(y, (-1, 1))], axis=1)
    # Split dataset into train set and test set
    train, test = train_test_split(data, test_size=0.2,
                                   random_state=random_state)

    # Split features array and labels list
    X_train, y_train = train[:, :-1], train[:, -1]
    X_test, y_test = test[:, :-1], test[:, -1]
    return X_train, y_train, X_test, y_test


def scale_dataset(X_train, X_test=None):
    '''SCALE_DATASET

        Normalize each feature of the dataset
        by subtracting feature's mean
        and deviding features's standard deviation (std).

        Inputs:
        -------

        - X_train : float array of training samples
                    in shape [n_train_samples, n_features].
        - X_test : optional, float array of testing samples
                   in shape [n_test_samples, n_features].

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

        - y_pred : Labels predicted by the SVM in shape [n_samples, ].
        - y_true : Real labels of dataset in shpe [n_samples, ].

        Output:
        -------

        - float, the accuray of prediction.

    '''

    return np.mean((y_pred == y_true) * 1.0)


def plot_decision_boundary(model, X_test=None, y_test=None):
    '''PLOT_DECISION_BOUNDARY

        Plot decision boundary on given dataset.

        Inputs:
        -------

        - model : a SVC instance, trained SVM model.
        - X_test : features array of test set in shape
                   [n_test_samples, n_features], default is None.
        - y_test : labels list of test set in shape
                   [n_test_samples, ], default is None.

    '''

    # Helper to generate 100 features.
    def get_range(feat):
        x_min, x_max = X[:, feat].min(), X[:, feat].max()
        return np.linspace(x_min, x_max, 100)

    # Set the data to be plotted
    # Plot scatters of test set if it is given,
    # else plot scatters of train set
    if (X_test is not None) and (y_test is not None):
        X, y = X_test, y_test
        title = "Test Set"
    else:
        X, y = model.X, model.y
        title = "Train Set"

    # Generate 10000 sample with grid features
    x0_range, x1_range = get_range(0), get_range(1)
    X_grid = np.array(np.meshgrid(x0_range, x1_range, indexing="xy"))
    X_grid = np.transpose(X_grid, (1, 2, 0)).reshape((-1, 2))

    # Predict 10000 samples
    preds = model.predict(X_grid, sign=False).reshape((100, 100))

    # Draw decision boundary with scatters of choosen dataset
    plt.figure()
    plt.contour(x0_range, x1_range, preds, (-1, 0, 1),
                linestyles=("--", "-", "--"),
                colors=("r", "k", "b"))
    plt.scatter(X[:, 0], X[:, 1], c=y,
                lw=0, alpha=0.5, cmap="RdYlBu")
    plt.xlabel("Feature 1", fontsize=12)
    plt.ylabel("Feature 2", fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.show()

    return
