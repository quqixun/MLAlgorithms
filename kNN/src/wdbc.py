# Conventional Machine Learning Algorithms
# Test script for Class of "kNN" on WDBC.
# Author: Qixun Qu
# Create on: 2018/05/04
# Modify on: 2018/05/04

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
from kNN import kNN
from utils import *


# Load dataset
wdbc_path = os.path.join(DATA_DIR, "WDBC", "wdbc.csv")
wdbc = pd.read_csv(wdbc_path, header=None)
wdbc_IDs = wdbc.iloc[:, 0].values
y = wdbc.iloc[:, 1].values
y[y == "M"] = 1
y[y == "B"] = 0
X = wdbc.iloc[:, 2:].values

# Basic settings
test_size = 0.2
random_state = 9527

# Split dataset into train set and test set
X_train, y_train, X_test, y_test = split_dataset(X, y, test_size,
                                                 random_state)
# Nomalize the data
X_train_scaled, X_test_scaled = scale_dataset(X_train, X_test)


# Build kd-tree
knn = kNN()
knn.fit(X_train_scaled, y_train)

# Do prediction using different
# number of nearest neighbors
n, step = 30, 1
K = list(range(0, n + 1, step))
K[0] = 1

wdbc_accs = []
for k in K:
    # Predict test samples
    knn.set_k(k)
    wdbc_pred = knn.predict(X_test_scaled)
    wdbc_acc = accuracy(wdbc_pred, y_test)
    print("Accuracy of WDBC dataset is: ", wdbc_acc)
    # The result can reach 0.9737
    wdbc_accs.append(wdbc_acc)

# Plot accuracies over different k
plot_curve(K, wdbc_accs, step=5)
