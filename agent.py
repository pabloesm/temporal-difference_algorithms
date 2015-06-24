# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 09:38:14 2015

@author: pabloem
"""

import numpy as np

import functionApproximator


class Td(object):
    """TD(0) algorithm"""
    def __init__(self, settings, funcApprox):
        self.gamma = settings["gamma"]
        self.alpha = settings["alpha"]
        self.fa = funcApprox

    def update(self, transition):
        s, s_prime, r = transition

        v_s_prime = np.dot(self.fa.theta.T, self.fa.features(s_prime))
        v_s = np.dot(self.fa.theta.T, self.fa.features(s))
        delta = r + self.gamma * v_s_prime - v_s
        self.fa.theta = self.fa.theta + (self.alpha *
                                         delta * self.fa.features(s))


class Tdlambda(Td):
    """TD($\lambda$) algorithm with accumulating traces"""
    def __init__(self, settings, funcApprox):
        super(Tdlambda, self).__init__(settings, funcApprox)
        self.lambda_ = settings["lambda_"]
        # eligibility vector. Initialize in each episode
        self.e = np.zeros((len(self.fa.theta), 1))

    def update(self, transition):
        s, s_prime, r = transition

        v_s_prime = np.dot(self.fa.theta.T, self.fa.features(s_prime))
        v_s = np.dot(self.fa.theta.T, self.fa.features(s))
        delta = r + self.gamma * v_s_prime - v_s
        self.e = self.gamma * self.lambda_ * self.e + (self.alpha *
                                                       self.fa.features(s))
        self.fa.theta = self.fa.theta + delta * self.e


class Tdlambda_replacing(Tdlambda):
    """TD($\lambda$) algorithm with replacing traces"""
    def update(self, transition, *args, **kwargs):
        s, s_prime, r = transition

        v_s_prime = np.dot(self.fa.theta.T, self.fa.features(s_prime))
        v_s = np.dot(self.fa.theta.T, self.fa.features(s))
        delta = r + self.gamma * v_s_prime - v_s

        self.traces_update(s)
        self.fa.theta = self.fa.theta + delta * self.e

    def traces_update(self, s):
        phi = self.fa.features(s)
        for i, item in enumerate(self.e):
            if phi[i] == 0:
                self.e[i] = self.gamma * self.lambda_ * self.e[i]
            else:
                self.e[i] = self.alpha * phi[i]


class True_online_Tdlambda(Tdlambda):
    """True online TD($\lambda$) algorithm"""
    def __init__(self, settings, funcApprox):
        super(True_online_Tdlambda, self).__init__(settings, funcApprox)
        # Initialization of V(s) using $\theta_{t-1}$ assuming s_0 = 0
        self.v_s = np.dot(self.fa.theta.T, self.fa.features(0))

    def update(self, transition):
        s, s_prime, r = transition

        v_s_prime = np.dot(self.fa.theta.T, self.fa.features(s_prime))
        delta = r + self.gamma * v_s_prime - self.v_s

        self.traces_update(s)

        term = self.alpha * (self.v_s -
                             np.dot(self.fa.theta.T,
                                    self.fa.features(s))) * self.fa.features(s)

        self.fa.theta = self.fa.theta + delta * self.e + term

        self.v_s = v_s_prime

    def traces_update(self, s):
        phi = self.fa.features(s)
        g = self.gamma
        l = self.lambda_
        a = self.alpha
        e = self.e

        self.e = g * l * e + a * phi - a * g * l * np.dot(e.T, phi) * phi


if __name__ == "__main__":
    settings = {}
    settings["lambda_"] = 0.5
    settings["gamma"] = 0.99
    settings["alpha"] = 0.125

    fa = functionApproximator.FA_task1()
    td = Tdlambda(settings, fa)
