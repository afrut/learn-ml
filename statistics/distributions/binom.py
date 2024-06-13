# exec(open('distributions\\binom.py').read())
# ----------------------------------------------------------------------
#
# Reference Material Montgomery & Runger: Applied statistics and Probability
# for Engineers 7ed
#
# ----------------------------------------------------------------------
import os

import counting_techniques as ct
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plots
import scipy.stats as stats

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    space = "    "

    # ----------------------------------------------------------------------
    # Plotting Parameters
    # ----------------------------------------------------------------------
    edgecolor = (
        np.array([0.121568627, 0.466666667, 0.705882353]) / 1.6
    )  # color of the edges of the bar graph rectangles
    show = False  # whether or not to show plots
    close = True  # whether or not to close plots

    # ----------------------------------------------------------------------
    #
    # Binomial Distribution
    # Consider the following situation: The probability of encountering a
    # poisonous molecule in a sample is 0.1. What is the probability of 5
    # samples containing a poisonous molecule in the next 20 samples? Take an
    # event that has a probability of occurrence p. The binomial distribution
    # gives the probability of this event occurring x times in the next n
    # trials.
    #
    # ----------------------------------------------------------------------
    print("----------------------------------------")
    print("  Binomial Distribution")
    print("----------------------------------------")
    # ----------------------------------------
    # probability mass function
    # ----------------------------------------
    p = 0.1  # the probability of an event E occurring
    n = 18  # in this many trials
    x = 2  # the number of times E occurs
    print("{0}An event E occurs with a probability of {1}.".format(space, p))
    prob = stats.binom.pmf(x, n, p)
    prob2 = ct.comb(n, x) * p**x * (1 - p) ** (n - x)
    assert round(prob - prob2, 8) == 0
    assert 0.284 - round(prob, 3) == 0
    print(
        "{0}The probability that E occurs {1} times".format(space, x)
        + " in the next {0} events is {1:.8}.".format(n, prob)
    )
    probs = list()
    xs = range(0, n + 1)
    for x in xs:
        prob = stats.binom.pmf(x, n, p)
        probs.append(round(prob, 8))
        print(
            "{0}The probability that E occurs {1} times".format(space, x)
            + " in the next {0} events is {1:.8}.".format(n, prob)
        )

    # ----------------------------------------
    # plotting
    # ----------------------------------------
    xs = np.array(xs)
    probs = np.array(probs).round(4)
    plots.barplot(
        xs,
        probs,
        title="Binomial Distribution; p = {0:.8}, n = {1}".format(p, n),
        align="edge",
        edgecolor=edgecolor,
        save=True,
        savepath=f"{script_dir}/binom.png",
    )
    print("")

    # ----------------------------------------
    # cumulative probabilities
    # ----------------------------------------
    x = 4
    prob = 0
    probcum = stats.binom.cdf(x - 1, n, p)
    probcum = 1 - probcum
    for x in range(0, x):
        prob = prob + stats.binom.pmf(x, n, p)
    prob = 1 - prob
    assert round(abs(probcum - prob), 8) == 0
    assert abs(0.098 - round(probcum, 3)) == 0
    print(
        "{0}The probability that E occurs >= {1} times in {2} events {3:.8} is".format(
            space, x + 1, n, prob
        )
    )

    x1 = 3
    x2 = 7
    prob = 0
    for x in range(x1, x2):
        prob = prob + stats.binom.pmf(x, n, p)
    probcum = (
        stats.binom.cdf(x2, n, p)
        - stats.binom.cdf(x1, n, p)
        - stats.binom.pmf(x2, n, p)
        + stats.binom.pmf(x1, n, p)
    )
    assert round(abs(prob - probcum), 8) == 0
    print(
        "{0}The probability that E occurs {1} <= x < {2} times in {3} events is {4:.8}".format(
            space, x1, x2, n, prob
        )
    )

    # ----------------------------------------
    # mean and variance
    # ----------------------------------------
    mu = stats.binom.mean(n, p)
    sigmasq = stats.binom.var(n, p)
    assert round(mu - (n * p), 8) == 0
    assert round(sigmasq - (n * p * (1 - p)), 8) == 0
    print("{0}The mean of this binomial distribution is {1:.8}".format(space, mu))
    print(
        "{0}The variance of this binomial distribution is {1:.8}".format(space, sigmasq)
    )
