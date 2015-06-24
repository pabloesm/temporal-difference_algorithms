# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 12:09:34 2015

@author: pabloem
"""


import numpy as np


class FA_task1():
    """
    Function approximator employed in the paper True Online TD ($\lambda$)
    by Seijen & Sutton (2014). Task 1 of the Section 4.3 Empirical Results
    """
    def __init__(self):
        self.theta = np.zeros((10, 1))

    def value(self, s):
        """ Compute the value of state s, i.e., V(s) """
        return np.dot(self.theta.T, self.features(s))

    def features(self, state):
        """Maps states to a feature vector: s -> $\phi(s)$"""
        featuresMap = self.map_()
        return featuresMap[state]

    def map_(self):
        """ Hand-coding the features """
        p = 1 / np.sqrt(2)
        q = 1 / np.sqrt(3)

        m = {}
        m[0] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], ndmin=2).T
        m[1] = np.array([p, p, 0, 0, 0, 0, 0, 0, 0, 0], ndmin=2).T
        m[2] = np.array([q, q, q, 0, 0, 0, 0, 0, 0, 0], ndmin=2).T
        m[3] = np.array([0, q, q, q, 0, 0, 0, 0, 0, 0], ndmin=2).T
        m[4] = np.array([0, 0, q, q, q, 0, 0, 0, 0, 0], ndmin=2).T
        m[5] = np.array([0, 0, 0, q, q, q, 0, 0, 0, 0], ndmin=2).T
        m[6] = np.array([0, 0, 0, 0, q, q, q, 0, 0, 0], ndmin=2).T
        m[7] = np.array([0, 0, 0, 0, 0, q, q, q, 0, 0], ndmin=2).T
        m[8] = np.array([0, 0, 0, 0, 0, 0, q, q, q, 0], ndmin=2).T
        m[9] = np.array([0, 0, 0, 0, 0, 0, 0, q, q, q], ndmin=2).T
        m[10] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ndmin=2).T
        return m

    def valueFunction(self):
        v = np.zeros((1, 11))
        for s in range(11):
            v[0][s] = self.value(s)
        return v


class FA_task2(FA_task1):
    """
    Function approximator employed in the paper True Online TD ($\lambda$)
    by Seijen & Sutton (2014). Task 2 of the Section 4.3 Empirical Results
    """
    def map_(self):
        """ Coding the features """
        n_features = len(self.theta)  # number of features
        n_states = 11  # number of states
        m = np.zeros((n_states, n_features))

        for i in range(n_states - 1):
            p = 1 / np.sqrt(i+1)
            m[i][0:i+1] = p
        return self.matrix2dict(m)

    def matrix2dict(self, mat):
        n_keys, array_length = mat.shape
        dic = {}
        for i in range(n_keys):
            dic[i] = np.expand_dims(mat[i, :], axis=0).T
        return dic


if __name__ == "__main__":
    fa = FA_task1()
    fa.theta = np.ones((10, 1))
    print(fa.value(1))
    print(fa.value(5))

    fa = FA_task2()
    fa.theta = np.ones((10, 1))
    # print(fa.map_())
    print(fa.value(1))
    print(fa.value(5))
