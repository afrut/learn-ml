# exec(open('distributions\\stdnorm.py').read())
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
    show = False  # whether or not to show plots
    close = True  # whether or not to close plots

    # ----------------------------------------------------------------------
    #
    # Standard Normal Distribution
    # The Standard Normal Distribution is simply the Normal Distribution
    # centered around 0 and has a standard deviation of 1. A random variable
    # that is distribution according to this distribution is denoted Z and its
    # values are called z-values. Standardization of a normal variable X
    # results in a z-value. z-value = (x - mu) / sigma.
    #
    # ----------------------------------------------------------------------
    print("----------------------------------------")
    print("  Standard Normal Distribution")
    print("----------------------------------------")
    # ----------------------------------------
    # probability density function
    # ----------------------------------------
    mu = 0
    sigma = 1
    x = 1.5
    print("{0}A random variable Z is standard-normally distributed".format(space))
    pdfval = stats.norm.pdf(x, mu, sigma)
    pdfval2 = (
        1
        / (math.sqrt(2 * math.pi) * sigma)
        * math.exp(-((x - mu) ** 2) * (1 / (2 * sigma**2)))
    )
    assert round(pdfval - pdfval2, 8) == 0
    print(
        "{0}The value of the probability density function at x = {1} is {2:.8}.".format(
            space, x, pdfval
        )
    )
    pdfvals = list()
    xs = np.arange(mu - 5, mu + 5 + 1, 1)
    for xl in xs:
        pdfval = stats.norm.pdf(xl, mu, sigma)
        pdfvals.append(pdfval)
        print(
            "{0}The value of the probability density function at x = {1} is {2:.8}.".format(
                space, xl, pdfval
            )
        )

    # ----------------------------------------
    # plotting
    # ----------------------------------------
    xs = np.array(xs)
    pdfvals = np.array(pdfvals).round(4)
    plots.barplot(
        xs,
        pdfvals,
        title="Standard Normal Distribution; mu = {0}, sigma = {1}".format(mu, sigma),
        align="edge",
        edgecolor=edgecolor,
        save=True,
        savepath=f"{script_dir}/stdnorm.png",
    )
    print("")

    # ----------------------------------------
    # cumulative probabilities
    # ----------------------------------------
    x = 0
    h = 1e-6
    probcum = stats.norm.cdf(x, mu, sigma)
    xs = np.arange((int)(-6 * sigma), x, h)
    probs = stats.norm.pdf(xs, mu, sigma)
    probcum2 = probs * h
    probcum2 = probcum2.sum()
    assert round(abs(probcum - probcum2), 4) == 0
    print(
        "{0}The probability that z <= {1} = P(Z <= {2}) = {3:.8}".format(
            space, x, x, probcum
        )
    )
    print("")

    # ----------------------------------------
    # some sample calculations
    # ----------------------------------------
    z = 1.26
    prob = stats.norm.cdf(z)
    print("{0}P(Z <= {1}) = {2:.8}".format(space, z, prob))
    assert 0.89617 - round(prob, 5) == 0

    prob = 1 - prob
    print("{0}P(Z > {1}) = {2:.8}".format(space, z, prob))
    assert 0.10383 - round(prob, 5) == 0

    z = -0.86
    prob = stats.norm.cdf(z)
    print("{0}P(Z < {1}) = {2:.8}".format(space, z, prob))
    assert 0.19489 - round(prob, 5) == 0

    z1 = -1.25
    z2 = 0.37
    prob1 = stats.norm.cdf(z1)
    prob2 = stats.norm.cdf(z2)
    prob = prob2 - prob1
    print("{0}P({1} < Z < {2}) = {3:.8}".format(space, z1, z2, prob))
    assert 0.53866 - round(prob, 5) == 0
