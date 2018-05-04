# Conventional Machine Learning Algorithms
# The Script to Test Class "kNN".
# Author: Qixun Qu
# Create on: 2018/04/30
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


from utils import *
from kNN import kNN
from sklearn.datasets import make_blobs


# Basic settings
random_state = 9527  # Seed
n_samples = 10000    # Number of samples
centers = 5          # Number of clusters
n_features = 5       # Number of features
test_size = 0.2      # Proportion of test set


# Generate Dataset for training and testing.
# Obtain all samples
X, y = make_blobs(n_samples=n_samples,
                  centers=centers,
                  n_features=n_features,
                  cluster_std=3,
                  random_state=random_state)
# Split dataset
X_train, y_train, X_test, y_test = split_dataset(X, y, test_size,
                                                 random_state)
# Normalize dataset
X_train_scaled, X_test_scaled = scale_dataset(X_train, X_test)


# Build kd-tree
knn = kNN()
knn.fit(X_train_scaled, y_train)

# Do prediction using different
# number of nearest neighbors
n, step = 100, 10
K = list(range(0, n + 1, step))
K[0] = 1

test_accs = []
for k in K:
    # Predict test samples
    knn.set_k(k)
    test_pred = knn.predict(X_test_scaled)
    test_acc = accuracy(test_pred, y_test)
    print("Accuracy of test dataset is: ", test_acc)
    # The result can reach 0.9725
    test_accs.append(test_acc)

# Plot accuracies over different k
plot_curve(K, test_accs)
