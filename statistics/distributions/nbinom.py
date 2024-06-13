# exec(open('distributions\\nbinom.py').read())
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
    # Negative Binomial Distribution
    # This is similar to the Geometric Distribution with a slight twist.
    #
    # Consider the following situation: The probability of encountering a
    # poisonous molecule in a sample is 0.1. What is the probability of
    # encountering the exactly 3 samples with a poisonous molecule by the 10th
    # sample? How about exactly 5 samples by the 15th? The Negative Binomial
    # Distribution answers these questions. It gives the probability that the
    # rth occurrence of event E (with probability p) occurs by exactly x events.
    #
    # ----------------------------------------------------------------------
    print("----------------------------------------")
    print("  Negative Binomial Distribution")
    print("----------------------------------------")
    # ----------------------------------------
    # probability mass function
    # ----------------------------------------
    p = 0.1  # probability of event E occurring
    r = 4  # number of occurrences of event E
    x = 10  # number of events
    pplot = p
    rplot = r
    print("{0}An event E occurs with a probability of {1}.".format(space, p))
    # pmf(num trials - num event E, num event E, probability)
    prob = stats.nbinom.pmf(x - r, r, p)
    print(
        "{0}The probability that E occurs exactly {1} times by the {2}th event is {3:.8}".format(
            space, r, x, prob
        )
    )
    assert round(abs(prob - 0.004464), 6) == 0
    xs = range(4, x + 1)
    probs = list()
    for x in xs:
        prob = stats.nbinom.pmf(x - r, r, p)
        probs.append(round(prob, 8))
        print(
            "{0}The probability that E occurs exactly {1} times by the {2}th event is {3:.8}".format(
                space, r, x, prob
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
        title="Negative Binomial Distribution; p = {0:.8}, r = {1}".format(
            pplot, rplot
        ),
        align="edge",
        edgecolor=edgecolor,
        save=True,
        savepath=f"{script_dir}/nbinom.png",
    )

    # ----------------------------------------
    # cumulative probabilities
    # ----------------------------------------
    x = 5
    r = 3
    p = 0.2
    print("")
    print("{0}An event E occurs with a probability of {1}.".format(space, p))
    prob = 0
    xs = range(r, x + 1)
    for x in xs:
        prob = prob + stats.nbinom.pmf(x - r, r, p)
    probcum = stats.nbinom.cdf(x - r, r, p)
    assert round(abs(prob - probcum), 8) == 0
    assert 0.058 == round(probcum, 3)
    print(
        "{0}The probability that E occurs exactly {1} times by the <= {2}th event is {3:.8}".format(
            space, r, x, prob
        )
    )

    # ----------------------------------------
    # mean and variance
    # ----------------------------------------
    mu = stats.nbinom.mean(r, p, r)
    sigmasq = stats.nbinom.var(r, p)
    assert round(mu - (r / p), 8) == 0
    assert round(sigmasq - (r * (1 - p) / p**2), 8) == 0
    print(
        "{0}The mean of this negative binomial distribution is {1:.8}".format(space, mu)
    )
    print(
        "{0}The variance of this negative binomial distribution is {1:.8}".format(
            space, sigmasq
        )
    )
    print("")
