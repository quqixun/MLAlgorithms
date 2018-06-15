# Conventional Machine Learning Algorithms
# Helper functions for Class of "KMeans".
# Author: Qixun Qu
# Create on: 2018/06/13
# Modify on: 2018/06/15

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


import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles, make_moons


# Ignore the warning caused by StandardScaler
# import warnings
# from sklearn.exceptions import DataConversionWarning
# warnings.filterwarnings("ignore", category=DataConversionWarning)


# Generate path of data directory
PARENT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
DATA_DIR = os.path.join(PARENT_DIR, "Data")


def generate_dataset(style="blobs",
                     n_samples=500, centers=2,
                     n_features=2, cluster_std=0.1,
                     noise=0.1, factor=0.1, random_state=None):
    '''GENERATE_DATASET

        Generate dataset to test DBSCAN.
        You may get three kinds of dataset through this function. Look at
        http://scikit-learn.org/stable/datasets/index.html#sample-generators
        for more infomation.

        Inputs:
        -------

        - style : string, default is "blobs", select one from
                  ["blobs", "circle", "moon"] to generate different dataset.
        - n_samples : int, default is 500, the number of samples.
        - centers: int, number of clusters if style is "blobs".
        - n_features: int, number of features if style if "blobs".
        - cluster_std: float, the standard deviation of the clusters
                       if style if "blobs".
        - noise: float, standard deviation of Gaussian noise added to the data
                 in "circles" and "moonss".
        - factor: float, scale factor between inner and outer "circles".
        - random_state: int or None, default is None,
                        the seed for reproducing dataset.

        Outputs:
        --------

        - X : float array in shape [n_samples, 2], features array.
        - y : float list in shape [n_samples, ], labels list.

    '''

    print("Generating dataset ...")

    if style == "blobs":
        # Multiple-cluster samples with n features
        X, y = make_blobs(n_samples=n_samples,
                          centers=centers,
                          n_features=n_features,
                          cluster_std=cluster_std,
                          random_state=random_state)
    elif style == "circles":
        # A larger circle outside, a smaller circle inside,
        # each sample has two features
        X, y = make_circles(n_samples=n_samples, noise=noise,
                            factor=factor, random_state=random_state)
    elif style == "moons":
        # Two interleaving half circles in 2D
        X, y = make_moons(n_samples=n_samples, noise=noise,
                          random_state=random_state)

    return X, y


def plot_clusters(clt):
    '''PLOT_CLUSTERS

        Plot clustering results after fitting DBSCAN.

        Input:
        ------

        - cls: a DBSCAN instance.

    '''

    if clt.X.shape[1] != 2:
        print("Please set n_feaures to 2.")
        return

    clt_max = max(clt.clusters) + 1

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(clt.X[:, 0], clt.X[:, 1], ".")
    plt.title("Original", fontsize=14)
    plt.xlabel("Feature 1", fontsize=12)
    plt.ylabel("Feature 2", fontsize=12)
    plt.subplot(122)
    for i in range(clt.X.shape[0]):
        plt.plot(clt.X[i, 0], clt.X[i, 1], ".",
                 color=cm.hsv((clt.clusters[i] + 1) / clt_max))
    plt.title("Clustering Results", fontsize=14)
    plt.xlabel("Feature 1", fontsize=12)
    plt.ylabel("Feature 2", fontsize=12)

    plt.tight_layout()
    plt.show()
