# exec(open('distributions\\poisson.py').read())
# ----------------------------------------------------------------------
#
# Reference Material Montgomery & Runger: Applied statistics and Probability
# for Engineers 7ed
#
# ----------------------------------------------------------------------
import math
import os

import numpy as np
import plots
import scipy.stats as stats

if __name__ == "__main__":
    space = "    "
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ----------------------------------------------------------------------
    # Plotting Parameters
    # ----------------------------------------------------------------------
    edgecolor = (
        np.array([0.121568627, 0.466666667, 0.705882353]) / 1.6
    )  # color of the edges of the bar graph rectangles

    # ----------------------------------------------------------------------
    #
    # Poisson Distribution
    # Consider the following situation: Flaws occur at random along the length
    # of a thin copper wire. It is given that the average number of flaws per
    # mm = lambda. What is the probability of encounter x = 10 flaws
    # in T = 5 mm of wire? The Poisson Distribution answers these kinds of
    # questions.
    #
    # In general, given an average number of occurrences of event E per unit
    # parameter(lambda), the Poisson Distribution gives the probability of
    # event E occurring x times in T units.
    #
    # ----------------------------------------------------------------------
    print("----------------------------------------")
    print("  Poisson Distribution")
    print("----------------------------------------")
    # ----------------------------------------
    # probability mass function
    # ----------------------------------------
    lamb = 2.3  # number of times event E occurs per unit on average
    x = 10  # number of times event E occurs
    T = 5  # unit examined
    prob = stats.poisson.pmf(x, lamb * T)
    prob2 = math.exp(-lamb * T) * (lamb * T) ** x / math.factorial(x)
    assert round(prob - prob2, 8) == 0
    assert 0.113 - round(prob, 3) == 0
    print("{0}An event E occurs {1} times per unit on average".format(space, lamb))
    print("{0}The number of units examined T = {1}".format(space, T))
    print(
        "{0}The probability of E occurring x = {1} times is {2:.8}".format(
            space, x, prob
        )
    )
    probs = list()
    xs = range(0, x + 5)
    for xl in xs:
        prob = stats.poisson.pmf(xl, lamb * T)
        probs.append(prob)
        print(
            "{0}The probability that E occurs {1} times".format(space, xl)
            + " in T = {0} units is {1:.8}".format(T, prob)
        )

    # ----------------------------------------
    # plotting
    # ----------------------------------------
    xs = np.array(xs)
    probs = np.array(probs).round(4)
    plots.barplot(
        xs,
        probs,
        title="Poisson Distribution; lamb = {0}, T = {1}".format(lamb, T),
        align="edge",
        edgecolor=edgecolor,
        save=True,
        savepath=f"{script_dir}/poisson.png",
    )
    print("")

    # ----------------------------------------
    # cumulative probabilities
    # ----------------------------------------
    x = 1
    T = 2
    probcum = stats.poisson.cdf(x, lamb * T)
    probcum = 1 - probcum + stats.poisson.pmf(x, lamb * T)
    prob = 0
    for xl in range(0, x):
        prob = prob + stats.poisson.pmf(xl, lamb * T)
    prob = 1 - prob
    assert round(prob - probcum, 8) == 0
    assert 0.9899 - round(probcum, 4) == 0
    print(
        "{0}The probability that E occurs x >= {1} times P(x >= {2}) = {3:.8}".format(
            space, x, x, probcum
        )
    )

    # ----------------------------------------
    # mean and variance
    # ----------------------------------------
    mu = stats.poisson.mean(lamb * T)
    sigmasq = stats.poisson.var(lamb * T)
    assert round(mu - (lamb * T), 8) == 0
    assert round(sigmasq - (lamb * T), 8) == 0
    print("{0}The mean of this poisson distribution is {1:.8}".format(space, mu))
    print(
        "{0}The variance of this poisson distribution is {1:.8}".format(space, sigmasq)
    )
    print("")
