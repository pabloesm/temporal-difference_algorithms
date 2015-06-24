# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:05:47 2015

@author: pabloem
"""

import numpy as np
import random
random.seed(1985)

from agent import Td
from functionApproximator import FA_task1
from environment import Randomwalk
from exputils import param_sweep, plot_curves

# Agent
settings = {}
settings["gamma"] = 0.99
settings["alpha"] = 1

fa = FA_task1()
tdAgent = Td(settings, fa)

# Environment
n_states = 11
transition_prob = 0.9
walker = Randomwalk(n_states, transition_prob)

# Experiment
confg = {}
confg["episodes"] = 10
confg["independent_runs"] = 20


param = "alpha"
alpha_vect = np.arange(0, 1.6, 0.1)
rmse = param_sweep(tdAgent, walker, confg, param, alpha_vect)
print rmse
plot_curves(alpha_vect, rmse)
