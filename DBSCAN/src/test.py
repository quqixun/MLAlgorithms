# Conventional Machine Learning Algorithms
# The Script to Test Class "DBSCAN".
# Author: Qixun Qu
# Create on: 2018/06/13
# Modify on: 2018/06/16

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
from dbscan import DBSCAN


# Basic settings
random_state = 9527   # Seed
n_samples = 600       # Number of samples

# Settings for blobs
bcenters = 4          # Number of clusters
bn_features = 2       # Number of features
bcluster_std = 1.0    # standard deviation of the clusters

# Settings for circles
cnoise = 0.1          # standard deviation of Gaussian noise added to the data
cfactor = 0.3         # scale factor between inner and outer circles

# Setting for moons
mnoise = 0.1          # standard deviation of Gaussian noise added to the data


# Blobs
# Generate points -- "blobs"
bX, _ = generate_dataset("blobs", n_samples=n_samples,
                         centers=bcenters, n_features=bn_features,
                         cluster_std=bcluster_std, random_state=random_state)
# Cluster points by DBSCAN
bdbs = DBSCAN(epsilon=1.0, min_samples=5)
bdbs.fit_predict(bX)
# Plot clustering results
plot_clusters(bdbs)


# Circles
# Generate points -- "circles"
cX, _ = generate_dataset("circles", n_samples=n_samples,
                         noise=cnoise, factor=cfactor,
                         random_state=random_state)
# Cluster points by DBSCAN
cdbs = DBSCAN(epsilon=0.2, min_samples=5)
cdbs.fit_predict(cX)
# Plot clustering results
plot_clusters(cdbs)


# Moons
# Generate points -- "moons"
mX, _ = generate_dataset("moons", n_samples=n_samples,
                         noise=mnoise, random_state=random_state)
# Cluster points by DBSCAN
mdbs = DBSCAN(epsilon=0.15, min_samples=5)
mdbs.fit_predict(mX)
# Plot clustering results
plot_clusters(mdbs)
