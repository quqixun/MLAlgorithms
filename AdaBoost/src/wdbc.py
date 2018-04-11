# Conventional Machine Learning Algorithms
# Test script for Class of "AdaBoostTree" on WDBC.
# Author: Qixun Qu
# Create on: 2018/04/10
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
from utils import *
from AdaBoostTree import *
from sklearn.tree import DecisionTreeClassifier


PARENT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
DATA_DIR = os.path.join(PARENT_DIR, "Data")


# Load dataset
wdbc_path = os.path.join(DATA_DIR, "WDBC", "wdbc.csv")
wdbc = pd.read_csv(wdbc_path, header=None)
wdbc_IDs = wdbc.iloc[:, 0].values
y = wdbc.iloc[:, 1].values
y[y == "M"] = 1.0
y[y == "B"] = -1.0
y = y.astype(float)
X = wdbc.iloc[:, 2:].values


# Basic settings
random_state = 9527
n_samples = 10000
test_size = 0.2


# Step 1
# Generate Dataset for training and testing.
# Split dataset
X_train, y_train, X_test, y_test = split_dataset(X, y, test_size,
                                                 random_state)
# Normalize dataset
X_train_scaled, X_test_scaled = scale_dataset(X_train, X_test)


# Step 2
# Generate the basic classifier.
clf = DecisionTreeClassifier(criterion="entropy",
                             max_depth=3,
                             random_state=random_state)


# Step 3
# Test class "AdaBoostTree"
# Set the number of iterations
M = 200

# Initialize the object
abt = AdaBoostTree(M, clf, verbose=True, vb_num=10)

# Fit training data
abt.fit(X_train_scaled, y_train,
        X_test_scaled, y_test)
# abt.fit(X_train_scaled, y_train)

# Predict test data
y_pred = abt.predict(X_test_scaled)
print("Accuracy of test set:", accuracy(y_pred, y_test))
# Accuracy could reach 0.9825.

# Plot learning curves
abt.plot_curves()
