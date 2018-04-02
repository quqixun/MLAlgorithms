# Conventional Machine Learning Algorithms
# Test script for Class of "SVM".
# Author: Qixun Qu
# Create on: 2018/03/24
# Modify on: 2018/03/24

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
from SVM import SVC
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


def scale_dataset(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def accuracy(y_pred, y_true):
    return np.mean((y_pred == y_true) * 1.0)


# Step 1

seed = 9526
X, y = generate_dataset("blob", 2000, seed)
X_train, y_train, X_test, y_test = split_dataset(X, y, 0.2, seed)
X_train_scaled, X_test_scaled = scale_dataset(X_train, X_test)

svc = SVC(C=1,
          kernel="linear",
          coef0=1.0,
          tol=1e-2,
          epsilon=1e-2)

svc.fit(X_train_scaled, y_train)
blob_pred = svc.predict(X_test_scaled)
blob_acc = accuracy(blob_pred, y_test)

print("Accuracy of Blob dataset is: ", blob_acc)
