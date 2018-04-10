# Conventional Machine Learning Algorithms
# Test script for Class of "SVM".
# Author: Qixun Qu
# Create on: 2018/03/24
# Modify on: 2018/04/10

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
from SVM import SVC


# Basic setting to generate dataset and
# make experiment reproducibale.

# Seed
random_state = 9527

# Number of all samples
n_samples = 500

# Proportion of test set
test_size = 0.2


# Blob Dataset

# Generate Blob dataset, and split it into train set and test set
X, y = generate_dataset("blob", n_samples, random_state)
X_train, y_train, X_test, y_test = split_dataset(X, y, test_size,
                                                 random_state)
# Nomalize the data
X_train_scaled, X_test_scaled = scale_dataset(X_train, X_test)

# Create a clissifier using linear kernel
svc = SVC(C=1, kernel="linear", coef0=1.0,
          random_state=random_state)

# Train and test the model
svc.fit(X_train_scaled, y_train)
blob_pred = svc.predict(X_test_scaled)
blob_acc = accuracy(blob_pred, y_test)
print("Accuracy of Blob dataset is: ", blob_acc)

# Draw decision boundary on train set and test set
plot_decision_boundary(svc)
plot_decision_boundary(svc, X_test_scaled, y_test)


# Circle

# Generate Circle dataset, and split it into train set and test set
X, y = generate_dataset("circle", n_samples, random_state)
X_train, y_train, X_test, y_test = split_dataset(X, y, test_size,
                                                 random_state)
# Nomalize the data
X_train_scaled, X_test_scaled = scale_dataset(X_train, X_test)

# Create a clissifier using RBF kernel
svc = SVC(C=1, kernel="rbf", gamma=1.0,
          random_state=random_state)

# Train and test the model
svc.fit(X_train_scaled, y_train)
circle_pred = svc.predict(X_test_scaled)
circle_acc = accuracy(circle_pred, y_test)
print("Accuracy of Circle dataset is: ", circle_acc)

# Draw decision boundary on train set and test set
plot_decision_boundary(svc)
plot_decision_boundary(svc, X_test_scaled, y_test)


# Moon

# Generate Moon dataset, and split it into train set and test set
X, y = generate_dataset("moon", n_samples, random_state)
X_train, y_train, X_test, y_test = split_dataset(X, y, test_size,
                                                 random_state)
# Nomalize the data
X_train_scaled, X_test_scaled = scale_dataset(X_train, X_test)

# Create a clissifier using polynomial kernel
svc = SVC(C=1, kernel="poly", coef0=1.0, degree=3,
          random_state=random_state)

# Train and test the model
svc.fit(X_train_scaled, y_train)
moon_pred = svc.predict(X_test_scaled)
moon_acc = accuracy(moon_pred, y_test)
print("Accuracy of Moon dataset is: ", moon_acc)

# Draw decision boundary on train set and test set
plot_decision_boundary(svc)
plot_decision_boundary(svc, X_test_scaled, y_test)
