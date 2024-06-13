# exec(open('distributions\\expon.py').read())
# ----------------------------------------------------------------------
#
# Reference Material Montgomery & Runger: Applied statistics and Probability
# for Engineers 7ed
#
# ----------------------------------------------------------------------
import math
import os

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
    # Exponential Distribution
    # Consider the following situation: Flaws occur at random along the length
    # of a thin copper wire. It is given that the average number of flaws per
    # mm = lambda. What is the probability of encounter x = 10 flaws
    # in T = 5 mm of wire? The Poisson Distribution answers these kinds of
    # questions. This is a Poisson process.
    #
    # Now, starting from a flaw, what is the probability that the next flaw
    # occurs in 1 mm? The exponential distribution answers these kinds of
    # questions.
    #
    # ----------------------------------------------------------------------
    print("----------------------------------------")
    print("  Exponential Distribution")
    print("----------------------------------------")
    # ----------------------------------------
    # probability density function
    # ----------------------------------------
    lamb = (float)(25)
    x = 0.1
    print("{0}An event E occurs {1} times per unit on average".format(space, lamb))
    pdfval = stats.expon.pdf(x, 0, 1 / lamb)
    pdfval2 = lamb * math.exp(-lamb * x)
    assert round(pdfval - pdfval2, 8) == 0
    print(
        "{0}The value of the probability density function at x = {1} is {2:.8}.".format(
            space, x, pdfval
        )
    )
    pdfvals = list()
    xstart = (float)(0.02)
    xend = (float)(0.7)
    h = (float)(0.02)
    xs = np.arange(xstart, xend, h)
    pdfvals = stats.expon.pdf(xs, 0, 1 / lamb).round(8)

    # ----------------------------------------
    # plotting
    # ----------------------------------------
    xs = np.array(xs).round(4)
    pdfvals = np.array(pdfvals).round(2)
    fig, ax1 = plots.barplot(
        xs,
        pdfvals,
        title="Exponential Distribution; lambda = {0:.8}".format(lamb),
        align="edge",
        edgecolor=edgecolor,
        width=h,
    )
    ax2 = ax1.twinx()
    fig, ax2 = plots.scatter(
        xs,
        pdfvals,
        fig=fig,
        ax=ax1,
        ylim=ax1.get_ylim(),
        markersize=0,
        linewidth=2,
    )
    ax2.set_title("")
    with open(f"{script_dir}/expon.png", "wb") as fl:
        fig.savefig(fl)

    # ----------------------------------------
    # sample calculations 1
    # ----------------------------------------
    prob1 = stats.expon.cdf(x, 0, 1 / lamb)
    prob1calc = 1 - math.exp(-x * lamb)
    prob2 = 1 - prob1
    assert 0.082 - round(prob2, 3) == 0
    assert round(prob1 - prob1calc, 8) == 0
    print(
        "{0}The probability that event E will occur in <= {1:.2} units is {2:.8}".format(
            space, x, prob1
        )
    )

    x1 = 2 / 60
    x2 = 3 / 60
    prob1 = stats.expon.cdf(x1, 0, 1 / lamb)
    prob1calc = -math.exp(-lamb * x1)

    prob2 = stats.expon.cdf(x2, 0, 1 / lamb)
    prob = prob2 - prob1
    assert 0.148 - round(prob, 3) == 0
    print(
        "{0}The probability that event E will occur in {1:.2} <= x <= {2:.2} units is {3:.8}".format(
            space, x1, x2, prob
        )
    )

    p = 0.9
    x = stats.expon.ppf(1 - p, 0, 1 / lamb)
    assert 0.00421 - round(x, 5) == 0
    print(
        "{0}The the interval that event E will NOT occur with a probability of {1:.4} is {2:.4}".format(
            space, p, x
        )
    )

    mu = stats.expon.mean(0, 1 / lamb)
    mucalc = 1 / lamb
    sigma = stats.expon.var(0, 1 / lamb)
    sigmacalc = 1 / lamb**2
    assert round(mu - mucalc, 8) == 0
    assert round(sigma - sigmacalc, 8) == 0
    assert 0.04 - round(mu, 2) == 0
    assert 0.0016 - round(sigma, 4) == 0
    print("{0}The mean of the distribution is {1:.4}".format(space, mu))
    print("{0}The sigma of the distribution is {1:.4}".format(space, sigma))
    print("")

    # ----------------------------------------
    # sample calculations 2
    # ----------------------------------------
    mu = 1.4
    lamb = 1 / mu
    x = 0.5
    prob = stats.expon.cdf(x, 0, 1 / lamb)
    print(
        "{0}The probability of event E occuring in <= {1:.2} units is {2:.3}".format(
            space, x, prob
        )
    )
