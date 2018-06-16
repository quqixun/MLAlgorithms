# k-Nearest Neighbors

This code implements **k-Nearest Neighbors** classifier by searching data in **kD-tree**.  
The code is tested under Python 3.6.5 in Windows 7. It should work in Linux and macOS  
if required libraries are installed.

## Algorithm

Detailed description of **k-nearest neighbors** can be found in [Wikipedia page](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm).  
The paper [Multidimensional Binary Search Trees Used for Associative Searching](http://delivery.acm.org/10.1145/370000/361007/p509-bentley.pdf?ip=129.16.80.135&id=361007&acc=ACTIVE%20SERVICE&key=74F7687761D7AE37%2E3C5D6C4574200C81%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1529185085_1e8668008ebb34ee05d78792813d3ccc) proposed kD-tree.

## Dataset

### Simulation Dataset

Use scikit-learn sample generator to create random examples in 2D.  
See [Sample generators](http://scikit-learn.org/stable/datasets/index.html#sample-generators) for more information.  
In this case, 10000 samples (in 5 clusters) are generated, and each  
sample has 5 features.

```python
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=10000,
                  centers=5,
                  n_features=5,
                  cluster_std=3,
                  random_state=9527)
```

### Breast Cancer Wisconsin (Diagnostic) Dataset (WDBC)

WDBC is an open dataset. It has 569 samples, and each sample has 32 attributes.  
Except ID and label, the other 30 features extracted from images of cell nuclei of breast mass.  
See [UCI Data Repo](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) for more information, dataset can be downloaded [here (select wdbc.data)](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/).

## Performance

Change working directory to this folder.  
And make sure that you have activated environment "algo".
```
cd kNN/src
```

### Simulation Dataset
Run the test script on **simulation dataset**.
```
python test.py
```
Plot the curve of accuracies over different **k**. The accuracy can reach **0.9725**.  
<img src="https://github.com/quqixun/MLAlgorithms/blob/master/kNN/images/test.png" alt="Test" width="500">

### WDBC Dataset
Run test on **WDBC dataset**.
```
python wdbc.py
```

Plot the curve of accuracies over different **k**.  
<img src="https://github.com/quqixun/MLAlgorithms/blob/master/kNN/images/wdbc.png" alt="WDBC" width="500">

The accuracy of test set can reach **0.9737**.
