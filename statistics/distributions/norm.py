# exec(open('distributions\\norm.py').read())
# ----------------------------------------------------------------------
#
# Reference Material Montgomery & Runger: Applied statistics and Probability
# for Engineers 7ed
#
# ----------------------------------------------------------------------
import math
import os

import counting_techniques as ct
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
    # Normal Distribution
    # Take a sample of 10 similar objects, measure them, and average their
    # measurements and call this a1. Take another sample of 10, measure them,
    # average and call this a2. Take yet another sample of 10, measure,
    # average, and call this a3. Do this many, many times, create a histogram
    # of the sample averages and the distribution of these sample avergaes will
    # be approximately Normal. This is the Central Limit Theorem.
    #
    # ----------------------------------------------------------------------
    print("----------------------------------------")
    print("  Normal/Gaussian Distribution")
    print("----------------------------------------")
    # ----------------------------------------
    # probability density function
    # ----------------------------------------
    mu = (float)(10)
    sigma = (float)(4)
    x = 5
    print(
        "{0}A random variable X is normally distributed".format(space)
        + " with mu = {0} and sigma = {1}.".format(mu, sigma)
    )
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
    xs = np.arange(mu - 5, mu + 5 + 1, (mu / 10))
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
        title="Normal Distribution; mu = {0:.8}, sigma = {1}".format(mu, sigma),
        align="edge",
        edgecolor=edgecolor,
        save=True,
        savepath=f"{script_dir}/norm.png",
    )
    print("")

    # ----------------------------------------
    # cumulative probabilities
    # ----------------------------------------
    x = 7
    h = 1e-6
    probcum = stats.norm.cdf(x, mu, sigma)
    xs = np.arange((int)(-6 * sigma), x, h)
    probs = stats.norm.pdf(xs, mu, sigma)
    probcum2 = probs * h
    probcum2 = probcum2.sum()
    assert round(abs(probcum - probcum2), 4) == 0
    print(
        "{0}The probability that x <= {1} P(x <= {2}) = {3:.8}".format(
            space, x, x, probcum
        )
    )
