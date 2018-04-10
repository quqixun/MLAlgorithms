# Conventional Machine Learning Algorithms
# The Script to Test Class "AdaBoostTree".
# Author: Qixun Qu
# Create on: 2018/03/13
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


from utils import *
from AdaBoostTree import *
from sklearn.datasets import make_hastie_10_2
from sklearn.tree import DecisionTreeClassifier


# Basic settings
random_state = 9527
n_samples = 10000
test_size = 0.2


# Step 1
# Generate Dataset for training and testing.
# Obtain all samples
X, y = make_hastie_10_2(n_samples=n_samples,
                        random_state=random_state)
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
# Accuracy could reach 0.9370.

# Plot learning curves
abt.plot_curve()
