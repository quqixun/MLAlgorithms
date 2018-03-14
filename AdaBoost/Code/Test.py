# Conventional Machine Learning Algorithms
# The Script to Test Class "AdaBoostTree".
# Author: Qixun Qu
# Create on: 2018/03/13
# Modify on: 2018/03/14

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


# Step 1
# Generate Dataset for training and testing.
#

X, Y = make_hastie_10_2()
data = np.concatenate([X, np.reshape(Y, (-1, 1))], axis=1)
train, test = train_test_split(data, test_size=0.2)

X_train, Y_train = train[:, :-1], train[:, -1]
X_test, Y_test = test[:, :-1], test[:, -1]


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
abt = AdaBoostTree(M, clf)

# Fit training data
abt.fit(X_train, Y_train, X_test, Y_test,
        verbose=True, vb_num=100)

# Predict test data
abt.predict(X_test)

# Plot learning curves
abt.plot_curve()
