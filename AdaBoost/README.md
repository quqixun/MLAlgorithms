# AdaBoost

This code implements AdaBoost method based on Decision Tree classifier.  
The code is tested under Python 3.6.4 and Python 2.7.12 in Ubuntu 16.04.  
It should work in Windows and macOS if required libraries are installed.

## Algorithm

The original AdaBoost algorithm can be found [here](http://www.face-rec.org/algorithms/Boosting-Ensemble/decision-theoretic_generalization.pdf).  
The code implements an [improved version](https://link.springer.com/content/pdf/10.1023%2FA%3A1007614523901.pdf) described as below.

<img src="https://github.com/quqixun/MLAlgorithms/blob/master/AdaBoost/images/algorithms.png" alt="Algorithms" width="600">

## Dataset

### Simulation Dataset

According to the document of [scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_hastie_10_2.html), data used for binary classification  
is generated as [Hastie et al. 2009, Example 10.2 (page 337)](https://web.stanford.edu/~hastie/Papers/ESLII.pdf).  
In this case, 12000 samples are generated.  
Each sample has ten features which are standard independent Gaussian  
and the label y is defined as:

```python
y[i] = 1 if np.sum(X[i] ** 2) > 9.34 else -1
```
### Breast Cancer Wisconsin (Diagnostic) Dataset (WDBC)

WDBC is an open dataset. It has 569 samples, and each sample has 32 attributes.  
Except ID and label, the other 30 features extracted from images of cell nuclei of breast mass.  
See [UCI Data Repo](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) for more information, dataset can be downloaded [here (select wdbc.data)](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/).

## Performance
Change working directory to this folder.  
And make sure that you have activated environment "algo".
```
cd AdaBoost/src
```
Run the test script in command line.
```
python test.py
```
Plot learning curves of error rates of training set and test set.  
The algorithm works well.

<img src="https://github.com/quqixun/MLAlgorithms/blob/master/AdaBoost/images/learning_curves.png" alt="Learning Curves" width="500">

Run test on WDBC dataset.
```
python wdbc.py
```
<img src="https://github.com/quqixun/MLAlgorithms/blob/master/AdaBoost/images/wdbc_learning_curves.png" alt="WDBC Learning Curves" width="500">
The accuracy of test set can reach **0.9825**.
