# Conventional Machine Learning Algorithms
# Test Script for Class of "NaiveBayes".
# Author: Qixun Qu
# Create on: 2018/04/24
# Modify on: 2018/04/25

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


from __future__ import division
from __future__ import print_function


from utils import *
from NaiveBayes import *
from sklearn.datasets import make_hastie_10_2


# Basic settings
random_state = 9527
n_samples = 10000
test_size = 0.2


# Generate Dataset for training and testing
# Obtain all samples
X, y = make_hastie_10_2(n_samples=n_samples,
                        random_state=random_state)
# Split dataset
X_train, y_train, X_test, y_test = split_dataset(X, y, test_size,
                                                 random_state)
# Normalize dataset
X_train_scaled, X_test_scaled = scale_dataset(X_train, X_test)


# Train Gaussian Naive Bayes Classifier
nb = NaiveBayes(alpha=1)
nb.fit(X_train_scaled, y_train, cont_feat_idx="all")

# Predict test set and evaluate results
y_pred = nb.predict(X_test_scaled)
print("Accuracy of test set:", accuracy(y_pred, y_test))
# Accuracy can reach 0.9765.
