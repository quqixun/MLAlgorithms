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


import numpy as np
from SVM import SVC
from sklearn.datasets import make_hastie_10_2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Step 1
# Generate Dataset for training and testing.
#

X, y = make_hastie_10_2(n_samples=1000,
                        random_state=9527)
data = np.concatenate([X, np.reshape(y, (-1, 1))], axis=1)
train, test = train_test_split(data, test_size=0.2)

X_train, y_train = train[:, :-1], train[:, -1]
X_test, y_test = test[:, :-1], test[:, -1]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Step 2
#

svc = SVC(C=1.0,
          kernel="linear",
          degree=2,
          sigma="auto",
          coef0=1.0,
          tol=1e-3,
          epsilon=1e-3,
          random_state=None)

svc.fit(X_train_scaled, y_train)
