import math
import os

import numpy as np
import pandas as pd
import plots
from matplotlib import pyplot as plt

# TODO: use standard plotting functions
# TODO: add sampling distribution of the difference between two means


def pdfnorm(x, mu, sigma):
    return (
        1 / (math.sqrt(2 * math.pi) * sigma) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    )


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.abspath(os.path.join(script_dir, "../visualization/outputs"))

    # ----------------------------------------------------------------------
    # Create a population of 1 million heights in cm with known mean and
    # variance. This data set will be used for simulations.
    # ----------------------------------------------------------------------
    mu = 175
    sigma = 15
    population = np.random.randn(1000000) * sigma + mu
    mu = np.mean(population)  # population mean
    sigma = np.std(population, ddof=0)  # population standard deviation
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
    # Draw a sample from the population with sample size of n. The sample
    # mean is our best point estimate of the population mean mu.
    # ----------------------------------------------------------------------
    n = 20
    sample = np.random.choice(population, size=n)
    dfSamp = pd.DataFrame(sample, columns=["height"])
    sampMean = round(np.mean(sample), 2)  # sample mean
    sampStd = round(np.std(sample, ddof=0), 2)  # sample standard deviation
    print("Sample mean = " + str(round(sampMean, 2)))
    print("Sample standard deviation = " + str(round(sampStd, 2)))
    print("")

    # ----------------------------------------------------------------------
    # Sample means can vary between samples. If we take 5 samples and
    # calculate the sample means, they will all be slightly different. The
    # natural question is, how do sample means vary with regards to the
    # population mean? It turns out that sample means can be thought of as
    # belonging to a sampling distribution whose mean is the population mean and
    # whose standard deviation = sigma / (sqrt(n)). The standard deviation
    # of the sampling distribution is called the standard error of the
    # sample mean. It gives us an idea of how sample means vary around
    # the true population mean. The sample mean can be thought of
    # as a value taken from the sampling distribution.
    #
    # NOTE: The population standard deviation is usually not known
    # when calculating the standard error of the sample mean. This is known
    # here because this is a a simulation. However, when the sampling size, n,
    # is greater than or equal to 30, the sampling distribution is
    # approximately normal and the standard deviation of the sample can be used
    # in place of the population standard deviation.
    # ----------------------------------------------------------------------
    # create the sampling distribution
    sampMeans = list()
    for cnt in range(0, 100000):
        sample = np.random.choice(population, size=n)
        sampMeans.append(np.mean(sample))
    dfSampDist = pd.DataFrame(sampMeans, columns=["height"])
    sampMeans = np.array(sampMeans)
    sampDistStd = round(np.std(sampMeans), 2)
    sampDistStdCalc = round(sigma / (np.sqrt(n)), 2)
    sampStdEst = round(np.std(sample, ddof=1) / math.sqrt(n), 4)
    print("Sampling distribution mean = " + str(round(np.mean(sampMeans), 2)))
    print("Sampling distribution standard deviation = " + str(sampDistStd))
    print(
        "Sampling distribution standard deviation calculated from "
        + "population standard deviation = "
        + str(sampDistStdCalc)
    )
    print(
        "Estimate of standard error of sample mean from sample data= " + str(sampStdEst)
    )
    print("")

    # NOTE: The standard deviation of the sampling distribution sampDistStdCalc
    # calculated based on the population standard deviation is quite close to
    # the actual standard deviation sampDistStd.

    # ----------------------------------------------------------------------
    # Visualization of Distrubtion Shapes
    # ----------------------------------------------------------------------

    numBins = 10  # number of bins in histograms
    figWidth = 14.4  # width of figure in inches
    nrow = 2  # number of subplot rows
    ncol = 3  # number of subplot columns
    nplot = 1  # current plot number

    # create single figure with subplots for all plots
    fig = plt.figure()
    fig.set_size_inches(figWidth, figWidth / 1.6)

    # histogram of population
    ax = fig.add_subplot(nrow, ncol, nplot)
    plots.histogram(
        dfPop,
        fig=fig,
        ax=ax,
        numBins=numBins,
        title="Population Distribution",
        xlabel=["height"],
        ylabel=["count"],
    )

    # create array of x values for calculating pdf values
    xmin = dfPop.loc[:, "height"].min()
    xmax = dfPop.loc[:, "height"].max()
    x = np.linspace(xmin, xmax, 500)

    # plot normal probability density function with population mean and variance
    pdf = pdfnorm(x, mu, sigma)
    ax = ax.twinx()
    plots.scatter(
        x,
        pdf,
        fig=fig,
        ax=ax,
        ylim=(pdf.min(), pdf.max()),
        title="",
        markersize=0,
        linewidth=2,
        color=plots.RED,
    )
    nplot = nplot + 1

    # NOTE that the shape population distribution is normal.

    # histogram of sample
    ax = fig.add_subplot(nrow, ncol, nplot)
    plots.histogram(
        dfSamp,
        fig=fig,
        ax=ax,
        numBins=numBins,
        title="Single Sample",
        xlabel=["height"],
        ylabel=["count"],
    )
    nplot = nplot + 1

    # histogram of sampling distribution
    ax = fig.add_subplot(nrow, ncol, nplot)
    plots.histogram(
        dfSampDist,
        fig=fig,
        ax=ax,
        numBins=numBins,
        title="Sampling Distribution",
        xlabel=["height"],
        ylabel=["count"],
    )

    # plot normal probability density function with  mean and variance
    # of sampling distribution
    xmin = dfSampDist.loc[:, "height"].min()
    xmax = dfSampDist.loc[:, "height"].max()
    x = np.linspace(xmin, xmax, 500)
    pdf = pdfnorm(x, sampMeans.mean(), sampDistStd)
    ax = ax.twinx()
    plots.scatter(
        x,
        pdf,
        fig=fig,
        ax=ax,
        ylim=(pdf.min(), pdf.max()),
        title="",
        markersize=0,
        linewidth=2,
        color=plots.RED,
    )
    nplot = nplot + 1

    # NOTE: The shape of the sampling distribution is normal. It is
    # also centered around the population mean and it has a smaller variance.
    # In other words, the spread is much less and the sampling distribution
    # appears "tighter" than the population distribution.

    # normal probability plot of population distribution
    ax = fig.add_subplot(nrow, ncol, nplot)
    plots.probplot(
        dfPop, fig=fig, ax=ax, title="Population Standard Normal Probability Plot"
    )
    nplot = nplot + 1

    # normal probability plot of the sample
    ax = fig.add_subplot(nrow, ncol, nplot)
    plots.probplot(
        dfSamp, fig=fig, ax=ax, title="Single Sample Standard Normal Probability Plot"
    )
    nplot = nplot + 1

    # normal probability plot of sample means
    ax = fig.add_subplot(nrow, ncol, nplot)
    plots.probplot(
        dfPop,
        fig=fig,
        ax=ax,
        title="Sampling Distribution Standard Normal Probability Plot",
    )
    nplot = nplot + 1

    # NOTE: The distribution of the individual sample may not be normal based
    # on the normal probability plots. What is important is that the
    # sampling distribution is normal.

    # ----------------------------------------------------------------------
    # Comparison of Population and Sampling Distributions
    # ----------------------------------------------------------------------
    xmin = dfPop.loc[:, "height"].min()
    xmax = dfPop.loc[:, "height"].max()
    x = np.linspace(xmin, xmax, 500)
    ypop = pdfnorm(x, mu, sigma)
    fig, ax = plots.scatter(
        x,
        ypop,
        ylim=(ypop.min(), ypop.max()),
        title="",
        markersize=0,
        linewidth=2,
        color=plots.BLUE,
    )
    ysamp = pdfnorm(x, sampMeans.mean(), sampDistStdCalc)
    fig, ax = plots.scatter(
        x,
        ysamp,
        fig=fig,
        ax=ax,
        ylim=(ysamp.min(), ysamp.max()),
        title="Population vs Sampling Distribution",
        markersize=0,
        linewidth=2,
        color=plots.RED,
    )
    ax.legend(["Population Distribution", "Sampling Distribution"])

    # show all plots
    fig.tight_layout()
    plt.savefig(f"{save_path}/population_vs_sampling_dist.png", format="png")
