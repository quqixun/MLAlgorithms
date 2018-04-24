# Conventional Machine Learning Algorithms
# Test Script for Class of "NaiveBayes".
# Author: Qixun Qu
# Create on: 2018/04/24
# Modify on: 2018/04/24

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


from __future__ import division
from __future__ import print_function


import os
import pandas as pd
from NaiveBayes import *


PARENT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
DATA_DIR = os.path.join(PARENT_DIR, "Data")

mushroom_path = os.path.join(DATA_DIR, "Mushroom", "mushroom.csv")
mushroom = pd.read_csv(mushroom_path, header=None)

X = mushroom.iloc[:, 1:].values
y = mushroom.iloc[:, 0].values
# print(type(X))
# print(X)


nb = NaiveBayes(lb=1)
nb.fit(X, y)
