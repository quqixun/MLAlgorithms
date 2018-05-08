# Conventional Machine Learning Algorithms
# The Script to Test Class "kNN" on Iris.
# Author: Qixun Qu
# Create on: 2018/05/08
# Modify on: 2018/05/08

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
from kmeans import KMeans


# Load dataset
iris_path = os.path.join(DATA_DIR, "Iris", "iris.csv")
iris = pd.read_csv(iris_path, header=None)
X = iris.iloc[:, :-1].values


# Initialize
km = KMeans(k=3, init="kmeans++", tol=1e-4,
            max_iters=100, random_state=9528)

# Cluster samples by sepal length and sepal width
km.fit(X[:, :2])
km.plot_clusters(X=X[:, :2], clusters=km.clusters,
                 axis_labels=["Sepal Length", "Sepal Width"])

# Cluster samples by petal length and petal width
km.fit(X[:, 2:])
km.plot_clusters(X=X[:, 2:], clusters=km.clusters,
                 axis_labels=["Petal Length", "Petal Width"])
