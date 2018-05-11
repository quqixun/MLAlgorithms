# K-Means

This code implements **K-Means** method to cluster data.  
The code is tested under Python 3.6.5 in Windows 7.  
It should work in Linux and macOS if required libraries are installed.

## Algorithm

### Initialize

Two methods are implemented to generate initial *k* cluster centers.

- Randomly selected *k* data points as initial cluster centers.
- Apply **kmeans++** to obtain *k* initial cluster centers.  
Here are steps of *kmeans++* (the copy from [Wikipedia page](https://en.wikipedia.org/wiki/K-means%2B%2B)):  
  1. Choose one center uniformly at random from among the data points.
  2. For each data point x, compute *D(x)*, the distance between x and  
  the nearest center that has already been chosen.
  3. Choose one new data point at random as a new center, using a weighted  
  probability distribution where a point x is chosen with probability proportional to *D(x)^2*.
  4. Repeat Steps ii and iii until *k* centers have been chosen.

### K-Means
1. Choose *k* initial cluster centers.  
2. For each data point, compute Euclidean distance to every center.  
  The cluster of this point is the index where the minimum distance is.  
3. Compute new cluster centers.  
4. If cluster centers converged (the difference between new cluster centers  
  and the cluster centers generated in previous iteration is smaller than a threshold),  
  stop iteration; otherwise, repeat step **2** and step **3**.

## Dataset

### Simulation Dataset

Use scikit-learn sample generator to create random examples in 2D or 3D.  
See [Sample generators](http://scikit-learn.org/stable/datasets/index.html#sample-generators) for more information.  
In this case, 2000 samples (in 4 clusters) are generated, and each  
sample has 2 or 3 features.

```python
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=2000,
                  centers=4,
                  n_features=2,  # or 3
                  cluster_std=2,
                  random_state=9527)
```

### Iris Dataset

The data set contains 3 classes of 50 instances each, where each class  
refers to a type of iris plant. Each data point has 4 arributes, which are  
sepal length, sepal width, petal length and petal width. See [UCI Data Repo](https://archive.ics.uci.edu/ml/datasets/iris)  
for more information, dataset can be downloaded [here (select iris.data)](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/).

## Performance

Change working directory to this folder.  
And make sure that you have activated environment "algo".
```
cd KMeans\src
```

### Simulation Dataset

Run the test script on **simulation dataset**.  
```
python test.py
```

Plot initialized state and the final clustering result  
over training data and testing data respectively.  

For training data,  
<img src="https://github.com/quqixun/MLAlgorithms/blob/master/KMeans/images/simu_train.png" alt="train" width="700">  
For testing data,  
<img src="https://github.com/quqixun/MLAlgorithms/blob/master/KMeans/images/simu_test.png" alt="test" width="700">

### Iris Dataset

Run test on **Iris dataset**.
```
python iris.py
```

Plot initialized state and the final clustering result  
using sepal feaures (length and width) and petal features  
(length and width) respectively.

Cluster data using sepal features,
<img src="https://github.com/quqixun/MLAlgorithms/blob/master/KMeans/images/iris_sepal.png" alt="sepal" width="700">  
Cluster data using petal features,
<img src="https://github.com/quqixun/MLAlgorithms/blob/master/KMeans/images/iris_petal.png" alt="petal" width="700">  
