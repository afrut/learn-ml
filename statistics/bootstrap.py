# exec(open('bootstrap.py').read())
import subprocess as sp

import numpy as np
import pandas as pd

if __name__ == "__main__":
    # create a population of 1 million heights in cm
    mu = 175
    sigma = 15
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
    dfPopulation = pd.DataFrame(population, columns=["height"])

    # get a sample from the population with sample size of n
    n = 50
    sample = np.random.choice(population, size=n)
    dfSample = pd.DataFrame(sample, columns=["height"])
    xbar = round(np.mean(sample), 2)  # sample mean
    sx = round(np.std(sample, ddof=0), 2)  # sample standard deviation
    print("Single Sample:")
    print("Sample mean = " + str(round(xbar, 2)))
    print("Sample standard deviation = " + str(round(sx, 2)))
    print("")

    # create the sampling distribution
    lsXBar = list()
    for cnt in range(0, 100000):
        samp = np.random.choice(population, size=n)
        lsXBar.append(np.mean(samp))
    dfSamplingDist = pd.DataFrame(lsXBar, columns=["height"])
    xbars = np.array(lsXBar)
    exbar = round(xbars.mean(), 2)
    sexbar = round(np.std(xbars), 2)
    print("Sampling Distribution:")
    print("E(xbar) = {0:.8}".format(exbar))
    print("SE(xbar) = {0:.8}".format(sexbar))
    print("")

    # bootstrap by treating sample as population
    nB = 200
    nb = 10
    assert nb < n
    lsXBarB = list()
    for cnt in range(nB):
        samp = np.random.choice(sample, size=nb)
        lsXBarB.append(samp.mean())
    dfSamplingDistB = pd.DataFrame(lsXBarB, columns=["height"])
    xbarsB = np.array(lsXBarB)
    exbarB = round(xbarsB.mean(), 2)
    sexbarB = round(xbarsB.std(), 2)
    print("Bootstrap:")
    print("E(xbarB) = {0:.8}".format(exbarB))
    print("SE(xbarB) = {0:.8}".format(sexbarB))
