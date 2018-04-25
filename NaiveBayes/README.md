# Naive Bayes Classifier

This code implements Naive Bayes classifier for discrete data as well as continuous data.  
The code is tested under Python 3.6.5 in Windows 7.  
It should work in Linux and macOS if required libraries are installed.

# Algorithms

## Multinomial Naive Bayes

## Gaussian Naive Bayes

# Dataset

### Simulation Dataset

According to the document of [scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_hastie_10_2.html), data used for binary classification  
is generated as [Hastie et al. 2009, Example 10.2 (page 337)](https://web.stanford.edu/~hastie/Papers/ESLII.pdf).  
In this case, 10000 samples are generated.  
Each sample has ten features which are standard independent Gaussian  
and the label y is defined as:

```python
y[i] = 1 if np.sum(X[i] ** 2) > 9.34 else -1
```

### Mushroom Dataset

**Mushroom** is an open dataset, which have 8124 samples, and each sample has 22 features.  
Each sample is classified based on whether it is toxic. 4208 samples are ediable, the other  
3916 samples are poisonous. See [UCI Data Repo](http://archive.ics.uci.edu/ml/datasets/Mushroom) for more information,  
dataset can be downloaded [here (select agaricus-lepiota.data)](http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/).

# Performance

Change working directory to this folder.  
And make sure that you have activated environment "algo".
```
cd NaiveBayes/src
```

Run the test script in command line.
```
python test.py
```

Classification accuracy of test set can reach **0.9765**.  

Run test on **Mushroom** dataset.
```
python mushroom.py
```

The accuracy of test set can reach **0.9938**.