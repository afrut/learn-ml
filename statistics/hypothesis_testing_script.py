import math
import os

import confidence_intervals as ci
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plots
import scipy.stats as stats
from hypothesis_testing import *

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.abspath(os.path.join(script_dir, "../visualization/outputs"))

    # ----------------------------------------------------------------------
    # Create a population of 1 million heights in cm with known mean and
    # variance. This data set will be used for simulations.
    # ----------------------------------------------------------------------
    mu = 175
    sigma = 5
    population = np.random.randn(1000000) * sigma + mu
    mu = np.mean(population)  # population mean
    sigma = np.std(population, ddof=0)  # population standard deviation
    print("Population Characteristics:")
    print("Population mean = " + str(round(mu, 2)))
    print("Population standard deviation = " + str(round(sigma, 2)))
    print(
        "Population range = ["
        + str(round(min(population), 2))
        + ","
        + str(round(max(population), 2))
        + "]"
    )
    print("")
    dfPop = pd.DataFrame(population, columns=["height"])

    # ----------------------------------------------------------------------
    # Draw a random sample with which to perform hypothesis testing.
    # ----------------------------------------------------------------------
    n = 20
    samp = np.random.choice(population, size=n)
    sampmean = samp.mean()
    sampstd = samp.std(ddof=1)
    print("Sample Characteristics:")
    print("Sample mean = " + str(round(sampmean, 2)))
    print("Sample standard deviation = " + str(round(sampstd, 2)))
    print(
        "Sample range = ["
        + str(round(min(samp), 2))
        + ","
        + str(round(max(samp), 2))
        + "]"
    )
    print("")

    print("----------------------------------------------------------------------")
    print("  Two-sided Hypothesis Testing with Known Population Standard Deviation")
    print("----------------------------------------------------------------------")
    alpha = 0.05
    mu0 = 175
    xlo, xhi = ci.twoTail(alpha, n=n, sampmean=mu0, sigma=sigma / math.sqrt(n))
    print("Sample mean = {0:.5}".format(sampmean))
    print(
        "Confidence Interval = {0:.5} <= x <= {1:.5}, alpha = {2:.2}".format(
            xlo, xhi, alpha
        )
    )
    if sampmean < xlo or sampmean > xhi:
        print("Reject H0: mu0 = {0} in favor of Ha: mu0 != {0}".format(mu0))
    else:
        print("Fail to reject H0: mu0 = {0}".format(mu0))
    print("")

    mu0 = 180
    xlo, xhi = ci.twoTail(alpha, n=n, sampmean=mu0, sigma=sigma / math.sqrt(n))
    print("Sample mean = {0:.5}".format(sampmean))
    print(
        "Confidence Interval = {0:.5} <= x <= {1:.5}, alpha = {2:.2}".format(
            xlo, xhi, alpha
        )
    )
    if sampmean < xlo or sampmean > xhi:
        print("Reject H0: mu0 = {0} in favor of Ha: mu0 != {0}".format(mu0))
    else:
        print("Fail to reject H0: mu0 = {0}".format(mu0))
    print("")

    print("----------------------------------------------------------------------")
    print("  Two-sided Hypothesis Testing with Unknown Population Standard Deviation")
    print("----------------------------------------------------------------------")
    alpha = 0.05
    mu0 = 175
    xlo, xhi = ci.twoTail(alpha, n=n, sampmean=mu0, sampstd=sampstd / math.sqrt(n))
    print("Sample mean = {0:.5}".format(sampmean))
    print(
        "Confidence Interval = {0:.5} <= x <= {1:.5}, alpha = {2:.2}".format(
            xlo, xhi, alpha
        )
    )
    if sampmean < xlo or sampmean > xhi:
        print("Reject H0: mu0 = {0} in favor of Ha: mu0 != {0}".format(mu0))
    else:
        print("Fail to reject H0: mu0 = {0}".format(mu0))
    print("")

    mu0 = 180
    xlo, xhi = ci.twoTail(alpha, n=n, sampmean=mu0, sampstd=sampstd / math.sqrt(n))
    print("Sample mean = {0:.5}".format(sampmean))
    print(
        "Confidence Interval = {0:.5} <= x <= {1:.5}, alpha = {2:.2}".format(
            xlo, xhi, alpha
        )
    )
    if sampmean < xlo or sampmean > xhi:
        print("Reject H0: mu0 = {0} in favor of Ha: mu0 != {0}".format(mu0))
    else:
        print("Fail to reject H0: mu0 = {0}".format(mu0))
    print("")

    print("----------------------------------------------------------------------")
    print(
        "  One-sided Lower-Bound Hypothesis Testing with Known Population Standard Deviation"
    )
    print("----------------------------------------------------------------------")
    alpha = 0.05
    mu0 = 175
    xlo = ci.oneTailLo(alpha, n=n, sampmean=mu0, sigma=sigma / math.sqrt(n))
    print("Sample mean = {0:.5}".format(sampmean))
    print("Confidence Interval = x >= {0:.5} , alpha = {1:.2}".format(xlo, alpha))
    if sampmean < xlo:
        print("Reject H0: mu0 = {0} in favor of Ha: mu0 < {0}".format(mu0))
    else:
        print("Fail to reject H0: mu0 = {0}".format(mu0))
    print("")

    mu0 = 180
    xlo = ci.oneTailLo(alpha, n=n, sampmean=mu0, sigma=sigma / math.sqrt(n))
    print("Sample mean = {0:.5}".format(sampmean))
    print("Confidence Interval = x >= {0:.5}, alpha = {1:.2}".format(xlo, alpha))
    if sampmean < xlo:
        print("Reject H0: mu0 = {0} in favor of Ha: mu0 < {0}".format(mu0))
    else:
        print("Fail to reject H0: mu0 = {0}".format(mu0))
    print("")

    print("----------------------------------------------------------------------")
    print(
        "  One-sided Lower-Bound Hypothesis Testing with Unknown Population Standard Deviation"
    )
    print("----------------------------------------------------------------------")
    alpha = 0.05
    mu0 = 175
    xlo = ci.oneTailLo(alpha, n=n, sampmean=mu0, sampstd=sampstd / math.sqrt(n))
    print("Sample mean = {0:.5}".format(sampmean))
    print("Confidence Interval = x >= {0:.5} , alpha = {1:.2}".format(xlo, alpha))
    if sampmean < xlo:
        print("Reject H0: mu0 = {0} in favor of Ha: mu0 < {0}".format(mu0))
    else:
        print("Fail to reject H0: mu0 = {0}".format(mu0))
    print("")

    mu0 = 180
    xlo = ci.oneTailLo(alpha, n=n, sampmean=mu0, sampstd=sampstd / math.sqrt(n))
    print("Sample mean = {0:.5}".format(sampmean))
    print("Confidence Interval = x >= {0:.5}, alpha = {1:.2}".format(xlo, alpha))
    if sampmean < xlo:
        print("Reject H0: mu0 = {0} in favor of Ha: mu0 < {0}".format(mu0))
    else:
        print("Fail to reject H0: mu0 = {0}".format(mu0))
    print("")

    print("----------------------------------------------------------------------")
    print(
        "  One-sided Upper-Bound Hypothesis Testing with Known Population Standard Deviation"
    )
    print("----------------------------------------------------------------------")
    alpha = 0.05
    mu0 = 175
    xhi = ci.oneTailHi(alpha, n=n, sampmean=mu0, sigma=sigma / math.sqrt(n))
    print("Sample mean = {0:.5}".format(sampmean))
    print("Confidence Interval = x =< {0:.5} , alpha = {1:.2}".format(xhi, alpha))
    if sampmean > xhi:
        print("Reject H0: mu0 = {0} in favor of Ha: mu0 > {0}".format(mu0))
    else:
        print("Fail to reject H0: mu0 = {0}".format(mu0))
    print("")

    mu0 = 165
    xhi = ci.oneTailHi(alpha, n=n, sampmean=mu0, sigma=sigma / math.sqrt(n))
    print("Sample mean = {0:.5}".format(sampmean))
    print("Confidence Interval = x =< {0:.5}, alpha = {1:.2}".format(xhi, alpha))
    if sampmean > xhi:
        print("Reject H0: mu0 = {0} in favor of Ha: mu0 > {0}".format(mu0))
    else:
        print("Fail to reject H0: mu0 = {0}".format(mu0))
    print("")

    print("----------------------------------------------------------------------")
    print(
        "  One-sided Upper-Bound Hypothesis Testing with Unknown Population Standard Deviation"
    )
    print("----------------------------------------------------------------------")
    alpha = 0.05
    mu0 = 175
    xhi = ci.oneTailHi(alpha, n=n, sampmean=mu0, sampstd=sampstd / math.sqrt(n))
    print("Sample mean = {0:.5}".format(sampmean))
    print("Confidence Interval = x =< {0:.5} , alpha = {1:.2}".format(xhi, alpha))
    if sampmean > xhi:
        print("Reject H0: mu0 = {0} in favor of Ha: mu0 > {0}".format(mu0))
    else:
        print("Fail to reject H0: mu0 = {0}".format(mu0))
    print("")

    mu0 = 165
    xhi = ci.oneTailHi(alpha, n=n, sampmean=mu0, sampstd=sampstd / math.sqrt(n))
    print("Sample mean = {0:.5}".format(sampmean))
    print("Confidence Interval = x =< {0:.5}, alpha = {1:.2}".format(xhi, alpha))
    if sampmean > xhi:
        print("Reject H0: mu0 = {0} in favor of Ha: mu0 > {0}".format(mu0))
    else:
        print("Fail to reject H0: mu0 = {0}".format(mu0))
    print("")

    print("----------------------------------------------------------------------")
    print("  P-value Hypothesis Tests")
    print("----------------------------------------------------------------------")
    sampmean = samp.mean()
    sampstd = samp.std(ddof=1)
    mu0 = 177
    pvalue = twoTailPvalue(n, mu0, sampmean, sigma=sigma)
    if pvalue < alpha / 2:
        msg = "reject"
    else:
        msg = "fail to reject"
    print("For a two-sided test with known sigma, the pvalue = {0:.6}".format(pvalue))
    print("    {0} the null hypothesis".format(msg))

    pvalue = twoTailPvalue(n, mu0, sampmean, sampstd=sampstd)
    if pvalue < alpha / 2:
        msg = "reject"
    else:
        msg = "fail to reject"
    print("For a two-sided test with unknown sigma, the pvalue = {0:.6}".format(pvalue))
    print("    {0} the null hypothesis".format(msg))

    mu0 = 177
    pvalue = oneTailPvalueLo(n, mu0, sampmean, sigma=sigma)
    if pvalue < alpha:
        msg = "reject"
    else:
        msg = "fail to reject"
    print(
        "For a one-sided lower-bound test with known sigma, the pvalue = {0:.6}".format(
            pvalue
        )
    )
    print("    {0} the null hypothesis".format(msg))

    pvalue = oneTailPvalueLo(n, mu0, sampmean, sampstd=sampstd)
    if pvalue < alpha:
        msg = "reject"
    else:
        msg = "fail to reject"
    print(
        "For a one-sided lower-bound test with unknown sigma, the pvalue = {0:.6}".format(
            pvalue
        )
    )
    print("    {0} the null hypothesis".format(msg))

    mu0 = 173
    pvalue = oneTailPvalueHi(n, mu0, sampmean, sigma=sigma)
    if pvalue < alpha:
        msg = "reject"
    else:
        msg = "fail to reject"
    print(
        "For a one-sided upper-bound test with known sigma, the pvalue = {0:.6}".format(
            pvalue
        )
    )
    print("    {0} the null hypothesis".format(msg))

    pvalue = oneTailPvalueHi(n, mu0, sampmean, sampstd=sampstd)
    if pvalue < alpha:
        msg = "reject"
    else:
        msg = "fail to reject"
    print(
        "For a one-sided upper-bound test with unknown sigma, the pvalue = {0:.6}".format(
            pvalue
        )
    )
    print("    {0} the null hypothesis".format(msg))

    print("")

    # ----------------------------------------------------------------------
    # Visualize Type 2 Error. NOTE: It is the area under the distribution
    # centered at mua that falls in the acceptance region of the distribution
    # centered at mu0.
    # ----------------------------------------------------------------------
    mu0 = 175
    mua = 179
    xmin = min(mu0, mua) - (5 * sigma / math.sqrt(n))
    xmax = max(mu0, mua) + (5 * sigma / math.sqrt(n))
    x = np.linspace(xmin, xmax, 500)
    y0 = stats.norm.pdf(x, loc=mu0, scale=sigma / math.sqrt(n))
    ya = stats.norm.pdf(x, loc=mua, scale=sigma / math.sqrt(n))
    ymin = min(y0.min(), ya.min())
    ymax = max(y0.max(), ya.max())

    # plot both distributions
    fig, ax = plots.scatter(
        x,
        y0,
        ylim=(0, max(y0.max(), ya.max())),
        xlabel="height",
        ylabel="f(x)",
        markersize=0,
        linewidth=2,
        color=plots.BLUE,
    )
    plots.scatter(
        x,
        ya,
        fig=fig,
        ax=ax,
        ylim=(0, max(y0.max(), ya.max())),
        xlabel="height",
        ylabel="f(x)",
        markersize=0,
        linewidth=2,
        color=plots.RED,
    )

    # find the acceptance region and fill it
    xlo, xhi = ci.twoTail(alpha, n=n, sampmean=mu0, sigma=sigma / math.sqrt(n))
    idx = np.multiply(x > xlo, x < xhi)
    xaccept = x[idx]
    ax.fill_between(xaccept, y0[idx], color=plots.BLUE)

    # find the type 2 error region
    idx = x < xhi
    xt2 = x[idx]
    ax.fill_between(xt2, ya[idx], color=plots.RED, zorder=3)

    # plot both means
    ax.plot(
        np.array([mu0, mu0]),
        np.array([0, ymax]),
        markersize=0,
        linewidth=2,
        color=plots.LIGHT_BLUE,
        linestyle="dashed",
    )
    ax.plot(
        np.array([mua, mua]),
        np.array([0, ymax]),
        markersize=0,
        linewidth=2,
        color=plots.LIGHT_RED,
        linestyle="dashed",
    )

    legend = [
        mpl.lines.Line2D(
            [0], [0], color=plots.BLUE, linewidth=2, label="Null Distribution"
        ),
        mpl.lines.Line2D(
            [0], [0], color=plots.RED, linewidth=2, label="Alternative Distribution"
        ),
        mpl.lines.Line2D(
            [0],
            [0],
            color=plots.LIGHT_BLUE,
            linewidth=2,
            linestyle="dashed",
            label="mu0 = {0}".format(mu0),
        ),
        mpl.lines.Line2D(
            [0],
            [0],
            color=plots.LIGHT_RED,
            linewidth=2,
            linestyle="dashed",
            label="mua = {0}".format(mua),
        ),
        mpl.patches.Patch(
            facecolor=plots.BLUE,
            label="{0}% Probability Interval".format(int((1 - alpha) * 100)),
        ),
        mpl.patches.Patch(facecolor=plots.RED, label="Probability of Type 2 Error"),
    ]
    ax.legend(handles=legend)

    ax.set_title("Area Representing Probability of Type 2 Error")
    fig.tight_layout()
    plt.savefig(f"{save_path}/type_2_error.png", format="png")

    print("----------------------------------------------------------------------")
    print("  Type 2 Error Calculations")
    print("----------------------------------------------------------------------")
    mu0 = 175
    mua = 179
    beta = twoTailT2(alpha, mua, mu0, n, sigma)
    print(
        "For a two-sided hypothesis test with alpha = {0}, mu0 = {1},".format(
            alpha, mu0
        )
        + " mua = {0}, n = {1}, and sigma = {2:.5},".format(mua, n, sigma)
    )
    print("    the probability of Type 2 Error beta is {0:.4}".format(beta))

    mu0 = 175
    mua = 171
    beta = twoTailT2(alpha, mua, mu0, n, sigma)
    print(
        "For a two-sided hypothesis test with alpha = {0}, mu0 = {1},".format(
            alpha, mu0
        )
        + " mua = {0}, n = {1}, and sigma = {2:.5},".format(mua, n, sigma)
    )
    print("    the probability of Type 2 Error beta is {0:.4}".format(beta))

    mu0 = 175
    mua = 171
    beta = oneTailT2Lo(alpha, mua, mu0, n, sigma)
    print(
        "For a one-sided lower-bound hypothesis test with alpha = {0}, mu0 = {1},".format(
            alpha, mu0
        )
        + " mua = {0}, n = {1}, and sigma = {2:.5},".format(mua, n, sigma)
    )
    print("    the probability of Type 2 Error beta is {0:.4}".format(beta))

    mu0 = 175
    mua = 179
    beta = oneTailT2Hi(alpha, mua, mu0, n, sigma)
    print(
        "For a one-sided upper bound hypothesis test with alpha = {0}, mu0 = {1},".format(
            alpha, mu0
        )
        + " mua = {0}, n = {1}, and sigma = {2:.5},".format(mua, n, sigma)
    )
    print("    the probability of Type 2 Error beta is {0:.4}".format(beta))
    print("")

    print("----------------------------------------------------------------------")
    print("  Type 2 Error Sample Size Calculations")
    print("----------------------------------------------------------------------")
    mu0 = 175
    mua = 179
    delta = mua - mu0
    beta = 0.01
    n = twoTailT2N(alpha, mua, mu0, beta, sigma)
    assert abs(twoTailT2(alpha, mua, mu0, n, sigma) - beta) < 1e-6
    print(
        "For a two-sided hypothesis test with alpha = {0}, mu0 = {1},".format(
            alpha, mu0
        )
        + " mua = {0}, beta = {1}, and sigma = {2:.5},".format(mua, beta, sigma)
    )
    print("    the sample size required is {0:.6}".format(n))

    mu0 = 175
    mua = 171
    delta = mua - mu0
    beta = 0.01
    n = twoTailT2N(alpha, mua, mu0, beta, sigma)
    assert abs(twoTailT2(alpha, mua, mu0, n, sigma) - beta) < 1e-6
    print(
        "For a two-sided hypothesis test with alpha = {0}, mu0 = {1},".format(
            alpha, mu0
        )
        + " mua = {0}, beta = {1}, and sigma = {2:.5},".format(mua, beta, sigma)
    )
    print("    the sample size required is {0:.6}".format(n))

    mu0 = 175
    mua = 171
    delta = mua - mu0
    beta = 0.01
    n = oneTailT2NLo(alpha, mua, mu0, beta, sigma)
    assert abs(oneTailT2Lo(alpha, mua, mu0, n, sigma) - beta) < 1e-6
    print(
        "For a one-sided hypothesis lower-bound test with alpha = {0}, mu0 = {1},".format(
            alpha, mu0
        )
        + " mua = {0}, beta = {1}, and sigma = {2:.5},".format(mua, beta, sigma)
    )
    print("    the sample size required is {0:.6}".format(n))

    mu0 = 175
    mua = 179
    delta = mua - mu0
    beta = 0.01
    n = oneTailT2NHi(alpha, mua, mu0, beta, sigma)
    assert abs(oneTailT2Hi(alpha, mua, mu0, n, sigma) - beta) < 1e-6
    print(
        "For a one-sided hypothesis upper-bound test with alpha = {0}, mu0 = {1},".format(
            alpha, mu0
        )
        + " mua = {0}, beta = {1}, and sigma = {2:.5},".format(mua, beta, sigma)
    )
    print("    the sample size required is {0:.6}".format(n))
    print("")

    print("----------------------------------------------------------------------")
    print("  Hypothesis Tests for a Population Proportion")
    print("----------------------------------------------------------------------")
    # find the 20th percentile
    x20 = np.percentile(population, 20)

    # true proportion of population belonging to class of interest
    p = 0.2

    # define a sample size and draw a sample
    n = 40
    samp = np.random.choice(population, size=n)

    # find the proportion of items that belong to the class of interest in the sample
    psamp = len(samp[samp <= x20]) / n

    # approximate normal sampling distribution
    sampmean = psamp
    sampstd = math.sqrt(p * (1 - p) / n)
    print("Sample proportion as mean = {0:.4}".format(sampmean))
    print("Sample standard deviation = {0:.4}".format(sampstd))
    print("")

    # ----------------------------------------------------------------------
    # two-sided hypothesis test
    # ----------------------------------------------------------------------
    mu0 = 0.2
    xlo, xhi = ci.twoTail(alpha, n=n, sampmean=mu0, sigma=sampstd)
    print("Two-sided confidence interval = {0:.4} <= x <= {1:.4}".format(xlo, xhi))
    if psamp < xlo or psamp > xhi:
        print("Reject H0: mu0 = {0} in favor of Ha: mu0 != {0}".format(mu0))
    else:
        print("Fail to reject H0: mu0 = {0}".format(mu0))
    print("")

    mu0 = 0.4
    xlo, xhi = ci.twoTail(alpha, n=n, sampmean=mu0, sigma=sampstd)
    print("Two-sided confidence interval = {0:.4} <= x <= {1:.4}".format(xlo, xhi))
    if psamp < xlo or psamp > xhi:
        print("Reject H0: mu0 = {0} in favor of Ha: mu0 != {0}".format(mu0))
    else:
        print("Fail to reject H0: mu0 = {0}".format(mu0))
    print("")

    # ----------------------------------------------------------------------
    # one-sided lower-bound hypothesis test
    # ----------------------------------------------------------------------
    mu0 = 0.2
    xlo = ci.oneTailLo(alpha, n=n, sampmean=mu0, sigma=sampstd)
    print("One-sided lower-bound confidence interval = x >= {0:.4}".format(xlo))
    if psamp < xlo:
        print("Reject H0: mu0 = {0} in favor of Ha: mu0 < {0}".format(mu0))
    else:
        print("Fail to reject H0: mu0 = {0}".format(mu0))
    print("")

    mu0 = 0.4
    xlo = ci.oneTailLo(alpha, n=n, sampmean=mu0, sigma=sampstd)
    print("One-sided lower-bound confidence interval = x >= {0:.4}".format(xlo))
    if psamp < xlo:
        print("Reject H0: mu0 = {0} in favor of Ha: mu0 < {0}".format(mu0))
    else:
        print("Fail to reject H0: mu0 = {0}".format(mu0))
    print("")

    # ----------------------------------------------------------------------
    # one-sided upper-bound hypothesis test
    # ----------------------------------------------------------------------
    mu0 = 0.2
    xhi = ci.oneTailHi(alpha, n=n, sampmean=mu0, sigma=sampstd)
    print("One-sided upper-bound confidence interval = x <= {0:.4}".format(xhi))
    if psamp > xhi:
        print("Reject H0: mu0 = {0} in favor of Ha: mu0 > {0}".format(mu0))
    else:
        print("Fail to reject H0: mu0 = {0}".format(mu0))
    print("")

    mu0 = 0.05
    xhi = ci.oneTailHi(alpha, n=n, sampmean=mu0, sigma=sampstd)
    print("One-sided upper-bound confidence interval = x <= {0:.4}".format(xhi))
    if psamp > xhi:
        print("Reject H0: mu0 = {0} in favor of Ha: mu0 > {0}".format(mu0))
    else:
        print("Fail to reject H0: mu0 = {0}".format(mu0))
    print("")
