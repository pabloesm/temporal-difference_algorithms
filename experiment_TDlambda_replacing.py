# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:05:47 2015

@author: pabloem
"""

import numpy as np
import random
random.seed(1985)

from agent import Tdlambda_replacing
from functionApproximator import FA_task1
from environment import Randomwalk
from exputils import performance, plot_curves

# Agent
settings = {}
settings["gamma"] = 0.99
settings["alpha"] = 1
settings["lambda_"] = 0

fa = FA_task1()
tdAgent = Tdlambda_replacing(settings, fa)

# Environment
n_states = 11
transition_prob = 0.9
walker = Randomwalk(n_states, transition_prob)

# Experiment
confg = {}
confg["episodes"] = 10
confg["independent_runs"] = 25


alpha_vect = np.arange(0, 1.6, 0.1)
lambda_vect = np.arange(0, 1.1, 0.1)

rmse = np.zeros((len(lambda_vect), len(alpha_vect)))

# Loop over the lambda and alpha parameters
for i, lambda_ in enumerate(lambda_vect):
    for j, alpha in enumerate(alpha_vect):
        setattr(tdAgent, "lambda_", lambda_)
        setattr(tdAgent, "alpha", alpha)
        rmse[i][j] = performance(tdAgent, walker, confg)

print rmse
plot_curves(alpha_vect, rmse, save=1, name="TDlambda_replacing_task1")
