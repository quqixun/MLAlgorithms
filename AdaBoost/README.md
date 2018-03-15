# AdaBoost

This code implements AdaBoost method based on Decision Tree classifier.  
The code is tested under Python 3.6.4 and Python 2.7.12 in Ubuntu 16.04.  
It should work in Windows and macOS if required libraries are installed.

## Algorithm

The original AdaBoost algorithm can be found ![here](http://www.face-rec.org/algorithms/Boosting-Ensemble/decision-theoretic_generalization.pdf).  
The code implements an ![improved version](https://link.springer.com/content/pdf/10.1023%2FA%3A1007614523901.pdf) described as below.

## Dataset

According to the document of ![scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_hastie_10_2.html), data used for binary classification  
is generated as ![Hastie et al. 2009, Example 10.2](https://web.stanford.edu/~hastie/Papers/ESLII.pdf).  
In this case, 12000 samples are generated.  
Each sample has ten features which are standard independent Gaussian  
and the label y is defined as:

```python
y[i] = 1 if np.sum(X[i] ** 2) > 9.34 else -1
```

## Performance

Plot learning curves of error rates of training set and test set.

The algorithm works well.

![Learning Curves](https://github.com/quqixun/MLAlgorithms/blob/master/AdaBoost/Images/learning_curves.png)
