# Conventional Machine Learning Algorithms
# Test script for Class of "SVM".
# Author: Qixun Qu
# Create on: 2018/03/24
# Modify on: 2018/04/03

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


from Utils import *
from SVM import SVC


random_state = 9527
n_samples = 500
test_size = 0.2


# Blob

X, y = generate_dataset("blob", n_samples, random_state)
X_train, y_train, X_test, y_test = split_dataset(X, y, test_size,
                                                 random_state)
X_train_scaled, X_test_scaled = scale_dataset(X_train, X_test)

svc = SVC(C=1,
          kernel="linear",
          coef0=1.0)

svc.fit(X_train_scaled, y_train)
blob_pred = svc.predict(X_test_scaled)
blob_acc = accuracy(blob_pred, y_test)

print("Accuracy of Blob dataset is: ", blob_acc)
plot_decision_boundary(svc)
# plot_decision_boundary(svc, X_test_scaled, y_test)


# Circle

X, y = generate_dataset("circle", n_samples, random_state)
X_train, y_train, X_test, y_test = split_dataset(X, y, test_size,
                                                 random_state)
X_train_scaled, X_test_scaled = scale_dataset(X_train, X_test)

svc = SVC(C=1,
          kernel="rbf",
          sigma=1.0)

svc.fit(X_train_scaled, y_train)
circle_pred = svc.predict(X_test_scaled)
circle_acc = accuracy(circle_pred, y_test)

print("Accuracy of Circle dataset is: ", circle_acc)
plot_decision_boundary(svc)
# plot_decision_boundary(svc, X_test_scaled, y_test)


# Moon

X, y = generate_dataset("moon", n_samples, random_state)
X_train, y_train, X_test, y_test = split_dataset(X, y, test_size,
                                                 random_state)
X_train_scaled, X_test_scaled = scale_dataset(X_train, X_test)

svc = SVC(C=1,
          kernel="poly",
          coef0=1.0,
          degree=3)

svc.fit(X_train_scaled, y_train)
moon_pred = svc.predict(X_test_scaled)
moon_acc = accuracy(moon_pred, y_test)

print("Accuracy of Moon dataset is: ", moon_acc)
plot_decision_boundary(svc)
# plot_decision_boundary(svc, X_test_scaled, y_test)
