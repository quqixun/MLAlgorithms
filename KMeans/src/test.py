# Conventional Machine Learning Algorithms
# The Script to Test Class "kNN".
# Author: Qixun Qu
# Create on: 2018/05/06
# Modify on: 2018/05/07

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
from kmeans import KMeans
from sklearn.datasets import make_blobs


# Basic settings
random_state = 9527  # Seed
n_samples = 2000     # Number of samples
centers = 4          # Number of clusters
n_features = 2       # Number of features
test_size = 0.2      # Proportion of test set
max_iters = 100      # Maximum number of iterations


# Generate Dataset for training and testing.
# Obtain all samples
X, y = make_blobs(n_samples=n_samples,
                  centers=centers,
                  n_features=n_features,
                  cluster_std=2,
                  random_state=random_state)
# Split dataset
X_train, y_train, X_test, y_test = split_dataset(X, y, test_size,
                                                 random_state)


# Initialize and fit
km = KMeans(k=centers, init="kmeans++", tol=1e-4,
            max_iters=max_iters, random_state=random_state)
km.fit(X_train)
# Plot clustering result of training data
km.plot_clusters()

# Cluster testing data and plot result
clusters = km.predict(X_test)
km.plot_clusters(X=X_test, clusters=clusters)
