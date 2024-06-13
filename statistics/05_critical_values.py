# TODO: finsih this script
import subprocess as sp

import numpy as np
import pandas as pd
import scipy.stats as stats

sp.call("cls", shell=True)  # clear screen

# ----------------------------------------------------------------------
#
# Critical values are used in the context of hypothesis testing. They
# are values of samples statistics that correspond to a particular
# significance level.
#
# Critical values can be interpreted as sample statistic values
# that are a certain distance away from an assumed null value. For
# example, 95%, two-tailed critical values are 1.96 (z-score with alpha
# = 0.025) standard errors away from the null value in both the + and
# - directions
#
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
#
# Main script
#
# ----------------------------------------------------------------------
# create a population of 100,000 numbers
mu = 160
sigma = 10
population = np.random.randn(10000) * sigma + mu
mu = np.mean(population)
sigma = np.std(population)
print("Population mean = " + str(round(mu, 2)))
print("Population standard deviation = " + str(round(sigma, 2)))
print("")

# draw a sample from the population with sample size of n
n = 50
sample = np.random.choice(population, size=n)
xbar = round(np.mean(sample), 2)
sx = round(np.std(sample), 2)
sexbar = round(sigma / (np.sqrt(n)), 2)
print("Sample mean = " + str(round(xbar, 2)))
print("Sample standard deviation = " + str(round(sx, 2)))
print("Standard error of sample mean = " + str(sexbar))
print("")

# ----------------------------------------------------------------------
#
# p-value calculations
#
# ----------------------------------------------------------------------
# calculate the z-score using the usually unknown population standard deviation
z = (xbar - mu) / (sigma / np.sqrt(n))

# alternatively, calculate the z-score using the sample standard deviation
z = (xbar - mu) / (sx / np.sqrt(n))

# find the one-tailed p-value
pvalue1 = 1 - stats.norm.cdf(z)

# find the two-tailed p-value
pvalue2 = (1 - stats.norm.cdf(z)) * 2

print("The one-tailed p-value = " + str(pvalue1))
print("The two-tailed p-value = " + str(pvalue2))
print("")
