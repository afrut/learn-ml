# exec(open('distributions\\geom.py').read())
# ----------------------------------------------------------------------
#
# Reference Material Montgomery & Runger: Applied statistics and Probability
# for Engineers 7ed
#
# ----------------------------------------------------------------------
import os

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
    # Geometric Distribution
    # Consider the following situation: The probability of encountering a
    # poisonous molecule in a sample is 0.1. What is the probability of
    # encountering the first sample with a poisonous molecule on the 10th
    # sample? How about the 15th? The geometric distribution answers these
    # questions. It gives the probability that the first occurrence of event E
    # (with probability p) occurs in exactly x events.
    #
    # ----------------------------------------------------------------------
    print("----------------------------------------")
    print("  Geometric Distribution")
    print("----------------------------------------")
    # ----------------------------------------
    # probability mass function
    # ----------------------------------------
    p = 0.1  # probability of an event E occuring
    x = 5  # the number of trials performed before E occurs
    prob = stats.geom.pmf(x, p)
    prob2 = (1 - p) ** (x - 1) * p
    assert round(prob - prob2, 8) == 0
    assert 0.066 == round(prob, 3)
    print("{0}An event E occurs with a probability of {1}.".format(space, p))
    print(
        "{0}The probability that E first occurs in {1} times is {2:.8}".format(
            space, x, prob
        )
    )
    probs = list()
    xs = range(x + 1)
    for x in xs:
        prob = stats.geom.pmf(x, p)
        probs.append(round(prob, 8))
        print(
            "{0}The probability that E first occurs in {1} times is {2:.8}".format(
                space, x, prob
            )
        )

    # ----------------------------------------
    # plotting
    # ----------------------------------------
    xs = np.array(xs)
    probs = np.array(probs).round(4)
    plots.barplot(
        xs,
        probs,
        title="Geometric Distribution; p = {0:.8}".format(p),
        align="edge",
        edgecolor=edgecolor,
        save=True,
        savepath=f"{script_dir}/geom.png",
    )
    print("")

    # ----------------------------------------
    # cumulative probabilities
    # ----------------------------------------
    x = 4
    prob = 0
    for x in range(0, x):
        prob = prob + stats.geom.pmf(x, p)
    prob = 1 - prob
    assert 0.729 == round(prob, 3)
    print(
        "{0}The probability that E first occurs in >= {1} events is {2:.8}".format(
            space, x + 1, prob
        )
    )

    x1 = 3
    x2 = 7
    prob = 0
    for x in range(x1, x2):
        prob = prob + stats.geom.pmf(x, p)
    probcum = (
        stats.geom.cdf(x2, p)
        - stats.geom.cdf(x1, p)
        - stats.geom.pmf(x2, p)
        + stats.geom.pmf(x1, p)
    )
    assert round(abs(probcum - prob), 8) == 0
    print(
        "{0}The probability that first E occurs in {1} <= x < {2} events is {3:.8}".format(
            space, x1, x2, prob
        )
    )

    # ----------------------------------------
    # mean and variance
    # ----------------------------------------
    mu = stats.geom.mean(p)
    sigmasq = stats.geom.var(p)
    assert round(mu - (1 / p), 8) == 0
    assert round(sigmasq - ((1 - p) / p**2), 8) == 0
    print("{0}The mean of this geometric distribution is {1:.8}".format(space, mu))
    print(
        "{0}The variance of this geometric distribution is {1:.8}".format(
            space, sigmasq
        )
    )
    print("")
