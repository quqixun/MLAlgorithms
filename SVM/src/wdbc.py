# Conventional Machine Learning Algorithms
# Test script for Class of "SVM" on WDBC.
# Author: Qixun Qu
# Create on: 2018/04/09
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
import pandas as pd
from SVM import SVC
from utils import *


# Load dataset
wdbc_path = os.path.join(DATA_DIR, "WDBC", "wdbc.csv")
wdbc = pd.read_csv(wdbc_path, header=None)
wdbc_IDs = wdbc.iloc[:, 0].values
y = wdbc.iloc[:, 1].values
y[y == "M"] = 1
y[y == "B"] = -1
X = wdbc.iloc[:, 2:].values

# Basic settings
test_size = 0.2
random_state = 9527

# Split dataset into train set and test set
X_train, y_train, X_test, y_test = split_dataset(X, y, test_size,
                                                 random_state)
# Nomalize the data
X_train_scaled, X_test_scaled = scale_dataset(X_train, X_test)

# Create a clissifier using RBF kernel
svc = SVC(C=1,
          kernel="rbf", degree=3,
          coef0=1.0, gamma=1.0,
          tol=0.1, epsilon=0.1,
          random_state=random_state)

# Train and test the model
svc.fit(X_train_scaled, y_train)
wdbc_pred = svc.predict(X_test_scaled)
wdbc_acc = accuracy(wdbc_pred, y_test)
print("Accuracy of WDBC dataset is: ", wdbc_acc)
# The accuracy of test set can reach 0.9649.
