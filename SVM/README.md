# Support Vector Machine

This code implements Support Vector Machine classifier based on Sequential Minimal Optimization.  
The code is tested under Python 3.6.4 and Python 2.7.12 in Ubuntu 16.04.  
It should work in Windows and macOS if required libraries are installed.  

## Algorithm

See original paper of [Sequential Minimal Optimization](https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/).  
All equations mentioned in the source code can be found in this paper.

## Dataset

### Simulation Dataset

Use scikit-learn sample generator to create random examples in 2D.  
See [Sample generators](http://scikit-learn.org/stable/datasets/index.html#sample-generators) for more information.

### Breast Cancer Wisconsin (Diagnostic) Dataset (WDBC)

WDBC is an open dataset. It has 569 samples, and each sample has 32 attributes.  
Except ID and label, the other 30 features extracted from images of cell nuclei of breast mass.  
See [UCI Data Repo](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) for more information, dataset can be downloaded [here (select wdbc.data)](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/).

## Performance

Run the test script in command line.
```
python test.py
```
Plot decision boundaries on both train set and test set.  
The algorithm works well.

| Dataset    | Train Set | Test Set |
|:----------:|:---------:|:---------:|
| **Blob**   | <img src="https://github.com/quqixun/MLAlgorithms/blob/master/SVM/images/blob_train.png" alt="Learning Curves" width="350"> | <img src="https://github.com/quqixun/MLAlgorithms/blob/master/SVM/images/blob_test.png" alt="Learning Curves" width="350"> |
| **Circle** | <img src="https://github.com/quqixun/MLAlgorithms/blob/master/SVM/images/circle_train.png" alt="Learning Curves" width="350"> | <img src="https://github.com/quqixun/MLAlgorithms/blob/master/SVM/images/circle_test.png" alt="Learning Curves" width="350"> |
| **Moon**   | <img src="https://github.com/quqixun/MLAlgorithms/blob/master/SVM/images/moon_train.png" alt="Learning Curves" width="350"> | <img src="https://github.com/quqixun/MLAlgorithms/blob/master/SVM/images/moon_test.png" alt="Learning Curves" width="350"> |

Run the test on WDBC dataset.
```
python wdbc.py
```
The accuracy of test set can reach **0.9649**.
