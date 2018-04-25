# Conventional Machine Learning Algorithms
# Test script for Class of "NaiveBayes" on Mushroom dataset.
# on Mushroom Dataset.
# Author: Qixun Qu
# Create on: 2018/04/25
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


from __future__ import print_function


import os
import pandas as pd
from utils import *
from NaiveBayes import *


# Basic settings
random_state = 9527
test_size = 0.2


# Load data from file
mushroom_path = os.path.join(DATA_DIR, "Mushroom", "mushroom.csv")
mushroom = pd.read_csv(mushroom_path, header=None)

# Obtain features and labels
X = mushroom.iloc[:, 1:].values
# Remove the feature which has
# too much missing values.
X = np.delete(X, 10, axis=1)
y = mushroom.iloc[:, 0].values

# Split dataset
X_train, y_train, X_test, y_test = split_dataset(X, y, test_size,
                                                 random_state)
# The dataset cannot be normalized,
# since its features are discrete


# Train Multinomial Naive Bayes Classifier
# You can change the value of alpha
nb = NaiveBayes(alpha=0)
nb.fit(X_train, y_train, cont_feat_idx=None)

# Predict test set and evaluate results
y_pred = nb.predict(X_test)
print("Accuracy of test set:", accuracy(y_pred, y_test))
# Accuracy can reach 0.9938.
