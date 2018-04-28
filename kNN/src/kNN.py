# Conventional Machine Learning Algorithms
# Class of "kNN".
# Author: Qixun Qu
# Create on: 2018/04/26
# Modify on: 2018/04/26

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


import numpy as np


class kNN(object):

    def __init__(self, k=10):
        '''__INIT__
        '''

        self.k = k
        self.kd_tree = None

        return

    def _build_kd_tree(self):
        '''_BUILD_KD_TREE
        '''

        return

    def _search_kd_tree(self):
        '''_SEARCH_KD_TREE
        '''

        return

    def fit(self, X, y):
        '''FIT
        '''

        return

    def predict(self, X):
        '''PREDICT
        '''

        return
