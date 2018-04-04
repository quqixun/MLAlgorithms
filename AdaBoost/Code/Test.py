# Conventional Machine Learning Algorithms
# The Script to Test Class "AdaBoostTree".
# Author: Qixun Qu
# Create on: 2018/03/13
# Modify on: 2018/04/04

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
from AdaBoostTree import *
from sklearn.datasets import make_hastie_10_2
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


seed = 9527
n_samples = 10000
test_size = 0.2

# Step 1
# Generate Dataset for training and testing.
#

X, y = make_hastie_10_2(n_samples=n_samples,
                        random_state=seed)
data = np.concatenate([X, np.reshape(y, (-1, 1))], axis=1)
train, test = train_test_split(data, test_size=test_size,
                               random_state=seed)

X_train, y_train = train[:, :-1], train[:, -1]
X_test, y_test = test[:, :-1], test[:, -1]


# Step 2
# Generate the classifier.
#

clf = DecisionTreeClassifier(criterion="entropy",
                             max_depth=3,
                             random_state=325)


# Step 3
# Test class "AdaBoostTree"
#

# Set the number of iterations
M = 200

# Initialize the object
abt = AdaBoostTree(M, clf,
                   verbose=True, vb_num=10)

# Fit training data
abt.fit(X_train, y_train, X_test, y_test)
# abt.fit(X_train, y_train)

# Predict test data
abt.predict(X_test)

# Plot learning curves
abt.plot_curve()
