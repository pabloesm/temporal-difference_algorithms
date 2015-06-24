# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 09:55:13 2015
@author: pabloem
"""

import numpy as np
import random


class Randomwalk:
    """
    Random walk environment from the paper True Online TD ($\lambda$) by
    Seijen & Sutton (2014)
    """
    def __init__(self, numStates=11, propabilityToTerminal=0.9):
        self.numStates = numStates
        self.propabilityToTerminal = propabilityToTerminal
        self.state = 0
        self.terminal = False
        self.reward = 0

    def action(self):
        if self.terminal:
            raise NameError('The current state is terminal!')

        if random.random() < self.propabilityToTerminal:
            self.state += 1
        elif self.state > 0:
            self.state -= 1
        else:
            pass

        if self.state >= self.numStates - 1:
            self.terminal = True
            self.reward = 1

    def value_function(self):
        """
        True value function analytically computed from code in
        http://webdocs.cs.ualberta.ca/~vanseije/trueonline.html
        """
        return np.array([0.89253031,
                         0.90254748,
                         0.9137901,
                         0.92529506,
                         0.9369583,
                         0.94877002,
                         0.9607308,
                         0.97284239,
                         0.98510667,
                         0.99752556,
                         0.0])


if __name__ == "__main__":
    random.seed(1985)
    walker = Randomwalk(5, 0.5)
    for i in range(50):
        walker.action()
        print "s: %s, r: %s, terminal: %s" % (walker.state,
                                              walker.reward, walker.terminal)
        if walker.terminal:
            break
