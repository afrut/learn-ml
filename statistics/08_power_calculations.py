# TODO: add margin of error in confidence intervals
# TODO: add critical values to everything
import subprocess as sp

import numpy as np
import pandas as pd
import scipy.stats as stats

sp.call("cls", shell=True)  # clear screen

# ----------------------------------------------------------------------
#
# Main script
#
# ----------------------------------------------------------------------
# create a population of 10,000 numbers
mu = [np.random.randint(170, 175), np.random.randint(170, 175)]
sigma = [np.random.randint(5, 10), np.random.randint(10, 15)]
population = [
    np.random.randn(10000) * sigma[0] + mu[0],
    np.random.randn(10000) * sigma[1] + mu[1],
]
mu = [round(np.mean(population[0])), round(np.mean(population[1]))]
sigma = [round(np.std(population[0]), 4), round(np.std(population[1]), 4)]
print("Population 0 mean = " + str(round(mu[0], 2)))
print("Population 0 standard deviation = " + str(round(sigma[0], 2)))
print("Population 1 mean = " + str(round(mu[1], 2)))
print("Population 1 standard deviation = " + str(round(sigma[1], 2)))
print("")

# draw a sample from the population with sample size of n
n = [np.random.randint(20, 29), np.random.randint(50, 70)]
sample = list()
sample.append(np.random.choice(population[0], size=n[0]))
sample.append(np.random.choice(population[1], size=n[1]))
xbar = [round(np.mean(sample[0]), 2), round(np.mean(sample[1]), 2)]
sx = [round(np.std(sample[0]), 2), round(np.std(sample[1]), 2)]
sexbar = [round(sigma[0] / (np.sqrt(n[0])), 2), round(sigma[1] / (np.sqrt(n[1])), 2)]
sexbar = [
    round(stats.sem(sample[0], ddof=0), 4),
    round(stats.sem(sample[1], ddof=0), 4),
]  # use scipy function to
# calculate sexbar
print("Sample 0 mean = " + str(round(xbar[0], 2)))
print("Sample 0 standard deviation = " + str(round(sx[0], 2)))
print("Sample 0 has size " + str(n[0]))
print("Standard error of sample 0 mean = " + str(sexbar[0]))
print("")
print("Sample 1 mean = " + str(round(xbar[1], 2)))
print("Sample 1 standard deviation = " + str(round(sx[1], 2)))
print("Sample 1 has size " + str(n[1]))
print("Standard error of sample 1 mean = " + str(sexbar[1]))
print("")

# ----------------------------------------------------------------------
#
# Power calculation for a 2-sided difference of two means
#
# ----------------------------------------------------------------------

# specifications
nullvalue = 0
alpha = 0.05

print("Assertion: Population 0 mean - Population 1 mean = 0")
print("H0: mu[0] - mu[1] = " + str(nullvalue))
print("HA: mu[0] - mu[1] != " + str(nullvalue))
print("xbar[0] = " + str(xbar[0]))
print("xbar[1] = " + str(xbar[1]))
print("xbar[0] - xbar[1] = " + str(round(xbar[0] - xbar[1], 4)))

# calculate the standard error of the difference between sample means
# without using the pooled standard deviation
sexbarDiff = np.sqrt(((sx[0] ** 2) / n[0]) + ((sx[1] ** 2 / n[1])))

# calculate the margin of error
moe = round(stats.norm.ppf(1 - alpha / 2) * sexbarDiff, 4)

# calculate the critical values of the corresponding significance level
cval = list()
cval.append(round(nullvalue + stats.norm.ppf(alpha / 2) * sexbarDiff, 4))
cval.append(round(nullvalue + stats.norm.ppf(1 - alpha / 2) * sexbarDiff, 4))
print(moe)
print(cval)
