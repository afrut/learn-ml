# exec(open('distributions\\binomNormApprox.py').read())
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
    close = True

    # ----------------------------------------------------------------------
    #
    # Normal Approximation of Binomial Distribution
    #
    # ----------------------------------------------------------------------
    # ----------------------------------------
    # probability mass function
    # ----------------------------------------
    p = 0.5  # the probability of an event E occurring
    n = 10  # in this many trials
    x1 = 0
    x2 = 10
    probs = list()
    xs1 = np.array(range(x1, x2 + 1))
    probs = stats.binom.pmf(xs1, n, p)
    xs2 = np.arange(x1, x2, 1e-1)
    proba = stats.norm.pdf((xs2 - (n * p)) / (math.sqrt((n * p) * (1 - p))))

    # ----------------------------------------
    # plotting
    # ----------------------------------------
    probs = np.array(probs).round(4)
    fig, ax1 = plots.barplot(
        xs1,
        probs,
        title="Normal Approximation of Binomial Distribution; p = {0:.8}, n = {1}".format(
            p, n
        ),
        align="edge",
        edgecolor=edgecolor,
        save=True,
        savepath=f"{script_dir}/binomNormApprox.png",
    )
    ax2 = ax1.twinx()
    fig, ax2 = plots.scatter(
        xs2 + 0.5, proba, fig=fig, ax=ax2, markersize=0, linewidth=2
    )
    ax2.set_title("")
    print("")

    # ----------------------------------------
    # sample calculations
    # ----------------------------------------
    p = 1e-5  # probability of E occurring
    n = 16e6  # in this many trials
    x = 150  # contains this many occurrences of E
    h = 1  # step size
    xs1 = np.arange(0, x + h, 1)
    probs = stats.binom.pmf(xs1, n, p)
    probcum = probs.sum()
    print(
        "Probability that fewer than {0} occurrences of E occur in {1} trials is P(X <= {2}) = {3:.8}".format(
            x, n, x, probcum
        )
    )
    assert (n * p) > 5
    assert n * (1 - p) > 5

    # approximate this probability with the standardized normal distribution
    zval = ((x + 0.5) - (n * p)) / math.sqrt((n * p) * (1 - p))
    probcum = stats.norm.cdf(zval)
    print(
        "Normal approximation of P(X <= {0}) = P(X <= {1}) ~= P(Z <= {2:.2}) = {3:.8}".format(
            x, x + 0.5, zval, probcum
        )
    )
    print("")

    p = 1e-1  # probability of E occurring
    n = 50  # in this many trials
    x = 2  # contains this many occurrences of E
    h = 1  # step size
    xs1 = np.arange(0, x + h, 1)
    probs = stats.binom.pmf(xs1, n, p)
    probcum = probs.sum()
    print(
        "Probability that fewer than {0} occurrences of E occur in {1} trials is P(X <= {2}) = {3:.8}".format(
            x, n, x, probcum
        )
    )

    # approximate this probability with the standardized normal distribution
    zval = ((x + 0.5) - (n * p)) / math.sqrt((n * p) * (1 - p))
    probcum = stats.norm.cdf(zval)
    print(
        "Normal approximation of P(X <= {0}) = P(X <= {1}) ~= P(Z <= {2:.2}) = {3:.8}".format(
            x, x + 0.5, zval, probcum
        )
    )
    print("")

    x = 5
    zval1 = ((x - 0.5) - (n * p)) / math.sqrt((n * p) * (1 - p))
    zval2 = ((x + 0.5) - (n * p)) / math.sqrt((n * p) * (1 - p))
    prob1 = stats.binom.pmf(x, n, p).sum()
    prob2 = stats.norm.cdf(zval2) - stats.norm.cdf(zval1)
    print(
        "P(X = {0}) ~= P({1} <= X <= {2}) ~= P({3:.2} <= Z <= {4:.2}) = {5:.8} ~= {6:.8}".format(
            x, x - 0.5, x + 0.5, zval1, zval2, prob1, prob2
        )
    )

    if show:
        plt.show()
    if close:
        plt.close()
