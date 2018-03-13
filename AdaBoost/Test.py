# The script to test class "AdaBoostTree".


import numpy as np
from AdaBoostTree import *
from sklearn.datasets import make_hastie_10_2
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


#
# Generate Dataset for training and testing.
#

X, Y = make_hastie_10_2()
data = np.concatenate([X, np.reshape(Y, (-1, 1))], axis=1)
train, test = train_test_split(data, test_size=0.2)

X_train, Y_train = train[:, :-1], train[:, -1]
X_test, Y_test = test[:, :-1], test[:, -1]


#
# Generate the classifier.
#

clf = DecisionTreeClassifier(criterion="entropy",
                             max_depth=3,
                             random_state=325)


#
# Test class "AdaBoostTree"
#

# Set the number of iterations
M = 200
abt = AdaBoostTree(M, clf)

# Fit training data
abt.fit(X_train, Y_train, X_test, Y_test,
        verbose=True, vb_num=10)

# Predict test data
abt.predict(X_test)

# Plot learning curves
abt.plot_curve()
