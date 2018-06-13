# Conventional Machine Learning Algorithms
# The Script to Test Class "kNN".
# Author: Qixun Qu
# Create on: 2018/06/13
# Modify on: 2018/06/13

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


from dbscan import DBSCAN
from sklearn.datasets import make_blobs


# Basic settings
random_state = 9527  # Seed
n_samples = 750      # Number of samples
centers = 3          # Number of clusters
n_features = 2       # Number of features


# Generate Dataset for training and testing.
# Obtain all samples
X, _ = make_blobs(n_samples=n_samples,
                  centers=centers,
                  n_features=n_features,
                  cluster_std=0.1,
                  random_state=random_state)


# Initialize and fit
dbs = DBSCAN(epsilon=0.1, min_samples=5,
             random_state=random_state)
dbs.fit_predict(X)
