import math
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plots
import scipy.stats as stats
from confidence_intervals import *

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
    print("----------------------------------------------------------------------")
    print("  Population Characteristics:")
    print("----------------------------------------------------------------------")
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
    # Calculate the sampling distribution's properties given that we know
    # the standard deviation of the population
    # ----------------------------------------------------------------------
    n = 20
    sampDistMu = mu
    sampDistSigma = sigma / math.sqrt(n)

    # ----------------------------------------------------------------------
    # Visualize one-sided vs two-sided confidence intervals.
    # ----------------------------------------------------------------------
    figsize = (14.4, 9)

    # define the minimum and maximum x values for the sampling distribution
    # to be 3 times the sampling distribution's standard deviation
    xmin = mu - 4 * (sigma / math.sqrt(n))
    xmax = mu + 4 * (sigma / math.sqrt(n))

    # create an array of x-values over which to calculate the pdf values of the
    # sampling distribution
    x = np.linspace(xmin, xmax, 500)

    # define the significance level
    alpha = 0.05

    # calculate values of the probability function
    y = stats.norm.pdf(x, loc=mu, scale=sigma / math.sqrt(n))

    # calculate the high and low values of x corresponding to a two-tailed
    # confidence interval
    xlotest = stats.norm.ppf(alpha / 2, loc=mu, scale=sampDistSigma)
    xhitest = stats.norm.ppf(1 - (alpha / 2), loc=mu, scale=sampDistSigma)
    xlo, xhi = twoTail(alpha, n=n, sampmean=mu, sigma=sigma / math.sqrt(n))
    assert abs(xlo - xlotest) < 1e-8
    assert abs(xhi - xhitest) < 1e-8

    # initialize plotting parameters
    nplot = 1
    nrow = 2
    ncol = 2
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(nrow, ncol, nplot)

    # plot the population distribution
    plots.scatter(
        x,
        y,
        fig=fig,
        ax=ax,
        title="Two-sided Confidence Interval",
        xlabel="height",
        ylabel="f(x)",
        linewidth=2,
        markersize=0,
    )

    # fill the areas corresponding to the significance level of a
    # 2-sided confidence interval
    xfill = x[x <= xlo]
    yfill = y[x <= xlo]
    ax.fill_between(xfill, yfill, color=plots.BLUE)
    xfill = x[x >= xhi]
    yfill = y[x >= xhi]
    ax.fill_between(xfill, yfill, color=plots.BLUE)
    nplot = nplot + 1

    # fill areas corresponding to the significance level of an
    # lower bound confidence interval
    ax = fig.add_subplot(nrow, ncol, nplot)
    plots.scatter(
        x,
        y,
        fig=fig,
        ax=ax,
        title="Lower Bound Confidence Interval",
        markersize=0,
        linewidth=2,
        xlabel="height",
        ylabel="f(x)",
        color=plots.BLUE,
    )
    xlotest = stats.norm.ppf(alpha, loc=mu, scale=sampDistSigma)
    xlo = oneTailLo(alpha, n=n, sampmean=mu, sigma=sigma / math.sqrt(n))
    assert abs(xlo - xlotest) < 1e-8
    xfill = x[x <= xlo]
    yfill = y[x <= xlo]
    ax.fill_between(xfill, yfill, color=plots.BLUE)
    nplot = nplot + 1

    # fill areas corresponding to the significance level of an
    # upper bound confidence interval
    ax = fig.add_subplot(nrow, ncol, nplot)
    plots.scatter(
        x,
        y,
        fig=fig,
        ax=ax,
        title="Upper Bound Confidence Interval",
        markersize=0,
        linewidth=2,
        xlabel="height",
        ylabel="f(x)",
        color=plots.BLUE,
    )
    xhitest = stats.norm.ppf(1 - alpha, loc=mu, scale=sampDistSigma)
    xhi = oneTailHi(alpha, n=n, sampmean=mu, sigma=sigma / math.sqrt(n))
    xfill = x[x >= xhi]
    yfill = y[x >= xhi]
    ax.fill_between(xfill, yfill, color=plots.BLUE)
    nplot = nplot + 1
    fig.suptitle("Sampling Distrubitions with alpha = {0:.2}".format(alpha))
    fig.tight_layout()
    plt.savefig(f"{save_path}/sampling_dist.png", format="png")

    # ----------------------------------------------------------------------
    # Student's t Distribution vs standard normal distribution
    # ----------------------------------------------------------------------
    # Calculate the probability density fucntion values for the sampling
    # distribution with known standard deviation
    yz = stats.norm.pdf(x, loc=sampDistMu, scale=sampDistSigma)

    # Draw a sample and caluculate the sample standard deviation.
    sample = np.random.choice(population, size=n)
    s = sample.std(ddof=1)

    # Calculate the probability density function values for the t distribution.
    # df = n - 1 specifies to use n - 1 degrees of freedom
    yt = stats.t.pdf(x, df=n - 1, loc=mu, scale=s / math.sqrt(n))

    # Visualize Student's t Distribution and compare to the population distribution
    # NOTE: Different samples yield different t distributions since t distributions
    # are dependent on the sample standard deviation s. The sample standard
    # deviation is a random variable that can change from sample to sample.
    ylim = (0, max(yz.max(), yt.max()))
    fig, ax = plots.scatter(
        x,
        yz,
        figsize=figsize,
        ylim=ylim,
        xlabel="height",
        ylabel="f(x)",
        linewidth=2,
        markersize=0,
        color=plots.BLUE,
    )
    plots.scatter(
        x,
        yt,
        fig=fig,
        ax=ax,
        ylim=ylim,
        title="",
        linewidth=2,
        markersize=0,
        color=plots.RED,
    )
    ax.set_title("t Distribution vs Standard Normal Distribution")
    fig.tight_layout()
    plt.savefig(f"{save_path}/t_dist_vs_std_normal_dist.png", format="png")

    # ----------------------------------------------------------------------
    # Plot the corrensponding normal distributions of the population and sampling
    # distributions. NOTE: The sampling distribution is much "tighter" implying
    # a smaller variance. Note that both are centered around the same value.
    # ----------------------------------------------------------------------
    x = dfPop.values
    xmin = x.min()
    xmax = x.max()
    x = np.linspace(xmin, xmax, 500)

    # calculate normal probability density function of sampling distribution
    # by using sigma/sqrt(n) as the sampling distribution variance
    ysamp = pdfnorm(x, sampDistMu, sampDistSigma)
    ylim = (0, ysamp.max())

    # plot sampling distribution
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    plots.scatter(
        x,
        ysamp,
        fig=fig,
        ax=ax,
        ylim=ylim,
        title="",
        markersize=0,
        linewidth=2,
        color=plots.RED,
    )

    # calculate normal probability density function of population with population variance
    ypop = pdfnorm(x, mu, sigma)

    # get the largest y value for y axis limits
    ylim = ax.get_ylim()
    if ypop.max() > ylim[1]:
        ylim = (ylim[0], ylim[1])

    # plot the population distribution
    plots.scatter(
        x,
        ypop,
        fig=fig,
        ax=ax,
        ylim=ylim,
        title="",
        markersize=0,
        linewidth=2,
        color=plots.BLUE,
    )

    # ----------------------------------------------------------------------
    # Plot areas that represent a 95% confidence interval. The population variance
    # sigma is assumed to be known. NOTE: The shaded region is represents a
    # range of x values that represent a 1 - alpha probability with respect
    # to the sampling distribution.
    # ----------------------------------------------------------------------
    # calculate the upper and lower x values that represent the
    # upper and lower limits of the confidence interval
    xlo, xhi = twoTail(alpha, n=n, sampmean=sampDistMu, sigma=sigma / math.sqrt(n))

    # create an array of x values that define the fill region
    xfill = np.linspace(xlo, xhi, 500)
    yfill = pdfnorm(xfill, sampDistMu, sampDistSigma)
    ax.fill_between(xfill, yfill, color=plots.RED)

    # plot the mean of the sampling distribution
    ax.plot(
        np.array([sampDistMu, sampDistMu]),
        np.array((ylim)),
        linewidth=1.25,
        color=plots.LIGHT_RED,
        linestyle="dashed",
    )

    # ----------------------------------------------------------------------
    # Build multiple confidence intervals by repeated sampling. Visualize.
    # NOTE: Not all confidence intervals contain the true mean mu.
    # ----------------------------------------------------------------------
    nDraw = 100  # number of samples to draw

    # Create an array of y positions where the confidence intervals are to be drawn.
    # Exclude that first and last points so as not to plot on the edges of the plotting area.
    ypos = np.linspace(ylim[0], ylim[1], nDraw + 2)
    ypos = ypos[1:-1]

    # Visualization parameters
    whiskerWidth = 1e-3 / 2

    cntNoMean = 0
    for y in ypos:
        markersize = 3
        linewidth = 0.75
        color = plots.GREEN

        # draw a sample
        sample = np.random.choice(population, size=n)
        sampMean = sample.mean()

        # calculate the confidence interval
        cilo, cihi = twoTail(alpha, n=n, sampmean=sampMean, sigma=sigma / math.sqrt(n))

        # count confidence intervals that do not contain
        # the sampling distribution mean
        if sampDistMu < cilo or sampDistMu > cihi:
            cntNoMean = cntNoMean + 1
            markersize = 7
            linewidth = 2
            color = plots.ORANGE

        # visualize
        ax.plot(
            sampMean, y, marker="o", markersize=markersize, linewidth=0, color=color
        )
        ax.plot([cilo, cihi], [y, y], linewidth=linewidth, color=color)
        ax.plot(
            [cilo, cilo],
            [y - whiskerWidth, y + whiskerWidth],
            linewidth=linewidth,
            color=color,
        )
        ax.plot(
            [cihi, cihi],
            [y - whiskerWidth, y + whiskerWidth],
            linewidth=linewidth,
            color=color,
        )

    print("----------------------------------------------------------------------")
    print("  Results of Multiple Confidence Intervals")
    print("----------------------------------------------------------------------")
    print(
        "Out of {0} samples, {1} standard normal confidence intervals do not contain the sampling distribution mean.".format(
            nDraw, cntNoMean
        )
    )

    # format plot
    legend = [
        mpl.lines.Line2D(
            [0], [0], color=plots.RED, linewidth=2, label="Sampling Distribution"
        ),
        mpl.lines.Line2D(
            [0], [0], color=plots.BLUE, linewidth=2, label="Population Distribution"
        ),
        mpl.lines.Line2D(
            [0],
            [0],
            color=plots.LIGHT_RED,
            linewidth=1.25,
            linestyle="dashed",
            label="Sampling Distribution Mean",
        ),
        mpl.patches.Patch(
            facecolor=plots.RED,
            label="{0}% Probability Interval".format(int((1 - alpha) * 100)),
        ),
        mpl.lines.Line2D(
            [0],
            [0],
            color=plots.GREEN,
            linewidth=1,
            label="{0}% Confidence Interval".format(int((1 - alpha) * 100)),
        ),
        mpl.lines.Line2D(
            [0],
            [0],
            color=plots.ORANGE,
            linewidth=2,
            label="Confidence Intervals that do not bound the sampling distribution mean",
        ),
    ]
    ax.legend(handles=legend)
    ax.set_xlabel("height")
    ax.set_ylabel("f(x)")
    ax.set_title(
        "{0} Confidence Intervals using a Standard Normal Distribution".format(nDraw)
    )
    fig.tight_layout()
    plt.savefig(f"{save_path}/ci_std_norm.png", format="png")

    # ----------------------------------------------------------------------
    # Student's t Distribution vs the population distribution
    # ----------------------------------------------------------------------
    # Calculate probability density function values for the standard normal distribution
    yz = stats.norm.pdf(x, loc=mu, scale=sigma)

    # Draw a sample and caluculate the sample standard deviation.
    sample = np.random.choice(population, size=n)
    s = sample.std(ddof=1)

    # Calculate the probability density function values for the t distribution.
    # df = n - 1 specifies to use n - 1 degrees of freedom
    yt = stats.t.pdf(x, df=n - 1, loc=mu, scale=s / math.sqrt(n))

    # Visualize Student's t Distribution and compare to the population distribution
    ylim = (0, max(yz.max(), yt.max()))
    fig, ax = plots.scatter(
        x,
        yz,
        figsize=figsize,
        ylim=ylim,
        xlabel="height",
        ylabel="f(x)",
        linewidth=2,
        markersize=0,
        color=plots.BLUE,
    )
    plots.scatter(
        x,
        yt,
        fig=fig,
        ax=ax,
        ylim=ylim,
        title="",
        linewidth=2,
        markersize=0,
        color=plots.RED,
    )

    # ----------------------------------------------------------------------
    # Build multiple confidence intervals using the t distribution.
    # ----------------------------------------------------------------------
    # calculate the upper and lower t values that represent the
    # upper and lower limits of the confidence interval using the t distribution
    tlo = stats.t.ppf(alpha / 2, df=n - 1)
    thi = stats.t.ppf(1 - (alpha / 2), df=n - 1)

    # upper and lower limits of the confidence interval
    xlo, xhi = twoTail(alpha, n=n, sampmean=sampDistMu, sampstd=s / math.sqrt(n))

    # create an array of x values that define the fill region
    xfill = np.linspace(xlo, xhi, 500)
    yfill = stats.t.pdf(xfill, n - 1, loc=sampDistMu, scale=s / math.sqrt(n))
    ax.fill_between(xfill, yfill, color=plots.RED)

    # plot the mean of the sampling distribution
    ax.plot(
        np.array([sampDistMu, sampDistMu]),
        np.array((ylim)),
        linewidth=1.25,
        color=plots.LIGHT_RED,
        linestyle="dashed",
    )

    # Create an array of y positions where the confidence intervals are to be drawn.
    # Exclude that first and last points so as not to plot on the edges of the plotting area.
    ypos = np.linspace(ylim[0], ylim[1], nDraw + 2)
    ypos = ypos[1:-1]

    # Visualization parameters
    whiskerWidth = 1e-3 / 2

    cntNoMean = 0
    for y in ypos:
        markersize = 3
        linewidth = 0.75
        color = plots.GREEN

        # draw a sample
        sample = np.random.choice(population, size=n)
        sampMean = sample.mean()
        s = sample.std(ddof=1)

        # calculate the confidence interval
        cilo, cihi = twoTail(alpha, n=n, sampmean=sampMean, sampstd=s / math.sqrt(n))

        # count confidence intervals that do not contain
        # the sampling distribution mean
        if sampDistMu < cilo or sampDistMu > cihi:
            cntNoMean = cntNoMean + 1
            markersize = 7
            linewidth = 2
            color = plots.ORANGE

        # visualize
        ax.plot(
            sampMean, y, marker="o", markersize=markersize, linewidth=0, color=color
        )
        ax.plot([cilo, cihi], [y, y], linewidth=linewidth, color=color)
        ax.plot(
            [cilo, cilo],
            [y - whiskerWidth, y + whiskerWidth],
            linewidth=linewidth,
            color=color,
        )
        ax.plot(
            [cihi, cihi],
            [y - whiskerWidth, y + whiskerWidth],
            linewidth=linewidth,
            color=color,
        )

    print(
        "Out of {0} samples, {1} t distribution confidence intervals do not contain the sampling distribution mean.".format(
            nDraw, cntNoMean
        )
    )
    print("")

    legend = [
        mpl.lines.Line2D(
            [0],
            [0],
            color=plots.BLUE,
            linewidth=2,
            label="Standard Normal Population Distribution",
        ),
        mpl.lines.Line2D(
            [0], [0], color=plots.RED, linewidth=2, label="t Distribution"
        ),
        mpl.lines.Line2D(
            [0],
            [0],
            color=plots.LIGHT_RED,
            linewidth=1.25,
            linestyle="dashed",
            label="Sampling Distribution Mean",
        ),
        mpl.patches.Patch(
            facecolor=plots.RED,
            label="{0}% Probability Interval".format(int((1 - alpha) * 100)),
        ),
        mpl.lines.Line2D(
            [0],
            [0],
            color=plots.GREEN,
            linewidth=1,
            label="{0}% Confidence Interval".format(int((1 - alpha) * 100)),
        ),
        mpl.lines.Line2D(
            [0],
            [0],
            color=plots.ORANGE,
            linewidth=2,
            label="Confidence Intervals that do not bound the sampling distribution mean",
        ),
    ]
    ax.legend(handles=legend)
    ax.set_title("{0} Confidence Intervals using a t Distribution".format(nDraw))
    fig.tight_layout()
    plt.savefig(f"{save_path}/ci_t_dist.png", format="png")

    print("----------------------------------------------------------------------")
    print("  Confidence Intervals on a Population Proportion")
    print("----------------------------------------------------------------------")
    # find the 20th percentile
    x20 = np.percentile(population, 20)

    # true proportion of population belonging to class of interest
    p = 0.2

    # define a sample size and draw a sample
    n = 30
    sample = np.random.choice(population, size=n)

    # find the proportion of items that belong to the class of interest in the sample
    psamp = len(sample[sample <= x20]) / n

    # approximate normal sampling distribution
    sampmean = psamp
    sampstd = math.sqrt(psamp * (1 - psamp) / n)
    print("Sample proportion as mean = {0:.4}".format(sampmean))
    print("Sample standard deviation = {0:.4}".format(sampstd))

    # two-sided confidence interval
    xlo, xhi = twoTail(alpha, n=n, sampmean=sampmean, sigma=sampstd)
    print("Two-sided confidence interval = {0:.4} <= x <= {1:.4}".format(xlo, xhi))

    # one-sided lower-bound confidence interval
    xlo = oneTailLo(alpha, n=n, sampmean=sampmean, sigma=sampstd)
    print("One-sided lower-bound confidence interval = x >= {0:.4}".format(xlo))

    # one-sided upper-bound confidence interval
    xhi = oneTailHi(alpha, n=n, sampmean=sampmean, sigma=sampstd)
    print("One-sided upper-bound confidence interval = x <= {0:.4}".format(xhi))
    print("")

    print("----------------------------------------------------------------------")
    print("  Sample Size Calculations")
    print("----------------------------------------------------------------------")
    # find sample size so that sample mean and population mean do not differ
    # by more than E
    E = 0.8
    n = math.ceil((stats.norm.ppf(1 - (alpha / 2)) * sigma / E) ** 2)

    # sample multiple times and count how many times the  difference between
    # sample mean and population mean exceed E
    nOut = 0
    for cnt in range(nDraw):
        sample = np.random.choice(population, size=n)
        if abs(sample.mean() - mu) > E:
            nOut = nOut + 1
    print("Using a sample size n = {0},".format(n))
    print(
        "    the number of sample means that differ from population mean by more than {0}".format(
            E
        )
        + " is {0}".format(nOut)
    )

    # find sample size so that minimum deviation from sample mean and
    # population mean is no less than E
    E = -0.8
    n = math.ceil((stats.norm.ppf(alpha) * sigma / E) ** 2)

    # sample multiple times and count how many times the  difference between
    # sample mean and population mean are less than E
    nOut = 0
    for cnt in range(nDraw):
        sample = np.random.choice(population, size=n)
        if sample.mean() - mu < E:
            nOut = nOut + 1
    print("Using a sample size n = {0},".format(n))
    print(
        "    the number of sample means that differ from population mean by less than {0}".format(
            E
        )
        + " is {0}".format(nOut)
    )

    # find sample size so that maximum deviation from sample mean and
    # population mean is no less than E
    E = 0.8
    n = math.ceil((stats.norm.ppf(1 - alpha) * sigma / E) ** 2)

    # sample multiple times and count how many times the  difference between
    # sample mean and population mean are less than E
    nOut = 0
    for cnt in range(nDraw):
        sample = np.random.choice(population, size=n)
        if sample.mean() - mu > E:
            nOut = nOut + 1
    print("Using a sample size n = {0},".format(n))
    print(
        "    the number of sample means that differ from population mean by more than {0}".format(
            E
        )
        + " is {0}".format(nOut)
    )
    print("")
