# -*- coding: utf-8 -*-
"""
Utility functions to perform the experiments

Created on Fri Jun 19 12:10:51 2015
@author: pabloem
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# It allows to edit the text of the figures in Adobe Illustrator
mpl.rcParams['pdf.fonttype'] = 42


def performance(agent, environment, confg):
    """Compute the performance of a method as the root-mean-squared (RMS) error
    of the value estimates of all non-terminal states at the end of an episode
    with respect to their true values (which are analytically determined),
    averaged over the first 10 episodes and 100 independent runs.
    """
    RMSE_vect = np.array([])

    for run in range(confg["independent_runs"]):
        # Initializations per run
        agent.fa.theta = agent.fa.theta * 0
        for episode in range(confg["episodes"]):
            # Initializations per episode
            environment.state = 0
            environment.terminal = False
            environment.reward = 0
            terminal = environment.terminal
            if hasattr(agent, 'e'):
                agent.e = np.zeros((len(agent.fa.theta), 1))
            if hasattr(agent, 'v_s'):
                # Initialization of V(s) using $\theta_{t-1}$ assuming s_0 = 0
                agent.v_s = np.dot(agent.fa.theta.T, agent.fa.features(0))

            while not terminal:
                s = environment.state
                environment.action()

                s_prime = environment.state
                r = environment.reward
                terminal = environment.terminal
                transition = (s, s_prime, r)

                agent.update(transition)
                # print "Run %s, episode %s, step %s" % (run, episode, s)

            E = agent.fa.valueFunction() - environment.value_function()
            E = E[0][:-1]  # Remove the value of the terminal state
            SE = E**2
            MSE = SE.mean(0)
            RMSE = np.sqrt(MSE)
            RMSE_vect = np.append(RMSE_vect, RMSE)

    return RMSE_vect.mean(0)


def param_sweep(agent, environment, confg, param, values):
    """Computes the performance according to the function performance() for a
    valid parameter param present in agent setting and a valid set of values"""
    result = np.zeros((len(values)))

    for i, item in enumerate(result.tolist()):
        setattr(agent, param, values[i])
        result[i] = performance(agent, environment, confg)

    return result


def plot_curves(x, y, save=0, name="figure"):
    dims = y.shape
    if len(dims) == 1:
        y = np.expand_dims(y, axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for i, yi in enumerate(y):
        ax.plot(x, yi)

    ax.set_xlim(0, 1.5)
    ax.set_xlabel("step-size")

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("RMS error")

    if save:
        plt.savefig(name)


if __name__ == "__main__":
    x = np.arange(0, 20, 0.2)
    y = np.array([np.sin(x), np.cos(x)])
    plot_curves(x, y)

    y2 = np.array(np.sin(x))
    plot_curves(x, y2, save=1, name="temp")
