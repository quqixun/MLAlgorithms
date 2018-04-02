# Conventional Machine Learning Algorithms
# Test script for Class of "SVM".
# Author: Qixun Qu
# Create on: 2018/03/24
# Modify on: 2018/04/02

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


# Step 1

seed = 9526
X, y = generate_dataset("blob", 2000, seed)
X_train, y_train, X_test, y_test = split_dataset(X, y, 0.2, seed)
X_train_scaled, X_test_scaled = scale_dataset(X_train, X_test)

svc = SVC(C=1,
          kernel="linear",
          coef0=1.0,
          tol=1e-2,
          epsilon=1e-2)

svc.fit(X_train_scaled, y_train)
blob_pred = svc.predict(X_test_scaled)
blob_acc = accuracy(blob_pred, y_test)

print("Accuracy of Blob dataset is: ", blob_acc)
