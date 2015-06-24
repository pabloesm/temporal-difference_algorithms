# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:05:47 2015

@author: pabloem
"""

from agent import Td
from functionApproximator import FA_task1
from environment import Randomwalk

import random
random.seed(1985)


settings = {}  # Agent settings
settings["gamma"] = 0.99
settings["alpha"] = 0.125

fa = FA_task1()
tdAgent = Td(settings, fa)

walker = Randomwalk(11, 0.9)
for episode in range(50):
    walker.state = 0
    walker.terminal = False
    walker.reward = 0
    terminal = walker.terminal

    while not terminal:
        s = walker.state
        walker.action()

        s_prime = walker.state
        r = walker.reward
        terminal = walker.terminal
        transition = (s, s_prime, r)

        tdAgent.update(transition)
        # print "Episode %s, step %s" % (episode, s)

print tdAgent.fa.valueFunction()
