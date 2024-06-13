# exec(open('distributions\\hypergeom.py').read())
# ----------------------------------------------------------------------
#
# Reference Material Montgomery & Runger: Applied statistics and Probability
# for Engineers 7ed
#
# ----------------------------------------------------------------------
import os

import counting_techniques as ct
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

    # ----------------------------------------------------------------------
    #
    # Hypergeometric Distribution
    # Consider the following situation: 850 parts produced contains 50
    # defective parts. Two parts are selected at random without replacement.
    # What is the probability that both parts are defective? What is the
    # probability that both parts are not defective? What is the probability
    # that only 1 of the parts is defective?
    #
    # Suppose that the event of interest is a part being defective. The
    # Hypergeometric Distribution gives the probability of the sample
    # containing x = 2 defective parts. In general, consider a pool of N = 850
    # objects, K = 50 of which are of interest. A sample of size n = 2 is drawn
    # from this pool. The Hypergeometric Distribution gives the probability
    # that x = 1 of these objects is of interest.
    #
    # ----------------------------------------------------------------------
    print("----------------------------------------")
    print("  Hypergeometric Distribution")
    print("----------------------------------------")
    # ----------------------------------------
    # probability mass function
    # ----------------------------------------
    N = 850  # number of total objects
    K = 50  # number of objects of interest
    n = 2  # number of objects drawn from pool as a sample
    x = 1  # number of objects of interest in the sample
    prob = stats.hypergeom.pmf(x, N, K, n)
    prob2 = ct.comb(K, x) * ct.comb(N - K, n - x) / ct.comb(N, n)
    assert round(prob - prob2, 8) == 0
    assert 0.111 - round(prob, 3) == 0
    print("{0}A pool contains N = {1} objects.".format(space, N))
    print("{0}K = {1} of these objects are of interest.".format(space, K))
    print("{0}A sample of size n = {1} is drawn from the pool.".format(space, n))
    print(
        "{0}The probability that the pool contains x = {1} objects of interest is {2:.8}.".format(
            space, x, prob
        )
    )
    probs = list()
    xs = range(0, n + 1)
    for x in xs:
        prob = stats.hypergeom.pmf(x, N, K, n)
        probs.append(round(prob, 8))
        print(
            "{0}The probability that the sample contains x = {1} objects of interest is {2:.8}.".format(
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
        title="Hypergeometric Distribution; N = {0}, K = {1}, n = {2}".format(N, K, n),
        align="edge",
        edgecolor=edgecolor,
        save=True,
        savepath=f"{script_dir}//hypergeom.png",
    )
    print("")

    # ----------------------------------------
    # cumulative probabilities
    # ----------------------------------------
    N = 300
    K = 100
    n = 4
    x = 2
    probcum = stats.hypergeom.cdf(x, N, K, n)
    probcum = 1 - probcum + stats.hypergeom.pmf(x, N, K, n)
    prob = 0
    xs = range(0, x)
    for xl in xs:
        prob = prob + stats.hypergeom.pmf(xl, N, K, n)
    prob = 1 - prob
    assert round(probcum - prob, 8) == 0
    assert 0.407 - round(probcum, 3) == 0
    print(
        "{0}The probability that the sample of size n = {1} contains".format(space, n)
        + " x >= {0} objects of interest is {1:.8}".format(x, probcum)
    )

    # ----------------------------------------
    # mean and variance
    # ----------------------------------------
    mu = stats.hypergeom.mean(N, K, n)
    sigmasq = stats.hypergeom.var(N, K, n)
    p = K / N
    assert round(mu - (n * p), 8) == 0
    assert round(sigmasq - (n * p * (1 - p) * ((N - n) / (N - 1))), 8) == 0
    print("{0}The mean of this hypergeometric distribution is {1:.8}".format(space, mu))
    print(
        "{0}The variance of this hypergeometric distribution is {1:.8}".format(
            space, sigmasq
        )
    )
    print("")
