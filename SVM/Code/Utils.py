# Conventional Machine Learning Algorithms
# Helper functions for Class of "SVM".
# Author: Qixun Qu
# Create on: 2018/04/02
# Modify on: 2018/04/09

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

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs, make_circles, make_moons


def generate_dataset(style="blob",
                     n_samples=1000,
                     random_state=None):

    if style == "blob":
        X, y = make_blobs(n_samples=n_samples,
                          centers=2,
                          n_features=2,
                          random_state=random_state)
    elif style == "circle":
        X, y = make_circles(n_samples=n_samples,
                            noise=0.1,
                            factor=0.1,
                            random_state=random_state)
    elif style == "moon":
        X, y = make_moons(n_samples=n_samples,
                          noise=0.1,
                          random_state=random_state)

    y[y == 0] = -1
    return X, y


def split_dataset(X, y,
                  test_size=0.2,
                  random_state=None):
    data = np.concatenate([X, np.reshape(y, (-1, 1))], axis=1)
    train, test = train_test_split(data, test_size=0.2,
                                   random_state=random_state)

    X_train, y_train = train[:, :-1], train[:, -1]
    X_test, y_test = test[:, :-1], test[:, -1]
    return X_train, y_train, X_test, y_test


def scale_dataset(X_train, X_test=None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_test is None:
        return X_train_scaled
    else:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled


def accuracy(y_pred, y_true):
    return np.mean((y_pred == y_true) * 1.0)


def plot_decision_boundary(model, X_test=None, y_test=None):

    def get_range(feat):
        x_min, x_max = X[:, feat].min(), X[:, feat].max()
        return np.linspace(x_min, x_max, 100)

    if (X_test is not None) and (y_test is not None):
        X, y = X_test, y_test
        title = "Test Set"
    else:
        X, y = model.X, model.y
        title = "Train Set"

    x0_range, x1_range = get_range(0), get_range(1)
    X_grid = np.array(np.meshgrid(x0_range, x1_range, indexing="xy"))
    X_grid = np.transpose(X_grid, (1, 2, 0)).reshape((-1, 2))
    preds = model.predict(X_grid, sign=False).reshape((100, 100))

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
