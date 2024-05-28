import subprocess as sp

import numpy as np
import pandas as pd
import scipy.stats as stats

sp.call("cls", shell=True)  # clear screen

# ----------------------------------------------------------------------
#
# Problems often involve making statements (hypotheses) about the
# population based on sample data. Assertions may be made that the
# population mean is greater than, less than, or equal to a certain
# value, c. These assertions depend on sample means since the population
# mean is usually unknown.
#
# ----------------------------------------------------------------------

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
# Hypothesis test for inequality
#
# ----------------------------------------------------------------------
nullvalue = mu[0]
print("Assertion: Population 1 mean is not equal to " + str(nullvalue))
print("H0: mu[1] = " + str(nullvalue))
print("HA: mu[1] != " + str(nullvalue))
print("xbar[1] = " + str(xbar[1]) + " vs " + "H0 = " + str(nullvalue))

# calculate a 95% confidence interval
alpha = 0.05
df = n[1] - 1
t = round(stats.t.ppf((1 - alpha / 2), df), 4)
cilo = round(xbar[1] - (t * sexbar[1]), 4)
cihi = round(xbar[1] + (t * sexbar[1]), 4)
print(
    "two-tailed 95% confidence interval = "
    + str(xbar[1])
    + " +/- "
    + str(t)
    + " * "
    + str(sexbar[1])
)
print("two-tailed 95% confidence interval = (" + str(cilo) + ", " + str(cihi) + ")")

# calculate a p-value
t = abs((xbar[1] - nullvalue) / sexbar[1])
pvalue = round((1 - stats.t.cdf(t, df)) * 2, 4)
print("pvalue = " + str(pvalue))

# state results of hypothesis tests
if (nullvalue >= cilo and nullvalue <= cihi) or (pvalue >= alpha):
    print("Fail to reject null hypothesis H0")
else:
    print("Reject null hypothesis H0")
print("")

# ----------------------------------------------------------------------
#
# One-sided hypothesis test for increase
#
# ----------------------------------------------------------------------
nullvalue = mu[0]
print("Assertion: Population 1 mean is greater than " + str(nullvalue))
print("H0: mu[1] = " + str(nullvalue))
print("HA: mu[1] > " + str(nullvalue))
print("xbar[1] = " + str(xbar[1]) + " vs " + "H0 = " + str(nullvalue))

# calculate a 95% confidence interval
alpha = 0.05
df = n[1] - 1
t = abs(round(stats.t.ppf(alpha, df), 4))
cilo = round(xbar[1] - (t * sexbar[1]), 4)
cihi = float("inf")
print("one-tailed 95% confidence interval = (" + str(cilo) + ", " + str(cihi) + ")")

# calculate a p-value
t = abs((xbar[1] - nullvalue) / sexbar[1])
pvalue = round(1 - stats.t.cdf(t, df), 4)
print("pvalue = " + str(pvalue))

# state the results of the hypothesis tests
if (nullvalue >= cilo and nullvalue <= cihi) or (pvalue > alpha or xbar[1] < nullvalue):
    print("Fail to reject null hypothesis H0")
else:
    print("Reject null hypothesis H0")
print("")

# ----------------------------------------------------------------------
#
# One-sided hypothesis test for decrease
#
# ----------------------------------------------------------------------
nullvalue = mu[0]
print("Assertion: Population 1 mean is less than " + str(nullvalue))
print("H0: mu[1] = " + str(nullvalue))
print("HA: mu[1] > " + str(nullvalue))
print("xbar[1] = " + str(xbar[1]) + " vs " + "H0 = " + str(nullvalue))

# calculate a 95% confidence interval
alpha = 0.05
df = n[1] - 1
t = abs(round(stats.t.ppf(alpha, df), 4))
cilo = float("-inf")
cihi = round(xbar[1] + (t * sexbar[1]), 4)
print("one-tailed 95% confidence interval = (" + str(cilo) + ", " + str(cihi) + ")")

# calculate a p-value
t = abs((xbar[1] - nullvalue) / sexbar[1])
pvalue = round(1 - stats.t.cdf(t, df), 4)
print("pvalue = " + str(pvalue))

# state the results of the hypothesis tests
if (nullvalue >= cilo and nullvalue <= cihi) or (pvalue > alpha or xbar[1] > nullvalue):
    print("Fail to reject null hypothesis H0")
else:
    print("Reject null hypothesis H0")
print("")

# ----------------------------------------------------------------------
#
# Difference of two means
#
# ----------------------------------------------------------------------
nullvalue = 0
print("Assertion: Population 0 mean - Population 1 mean = 0")
print("H0: mu[0] - mu[1] = " + str(nullvalue))
print("HA: mu[0] - mu[1] != " + str(nullvalue))
print("xbar[0] = " + str(xbar[0]))
print("xbar[1] = " + str(xbar[1]))

# Use pooled standard deviations, sp, in place of sx[0] and sx[1] in the
# above analyses when background research indicates that the population
# standard eviations are nearly equal.
sp = np.sqrt(
    ((sx[0] ** 2 * (n[0] - 1)) + (sx[1] ** 2 * (n[1] - 1))) / (n[0] + n[1] - 2)
)

# calculate the standard error of the difference between two means
if abs(1 - (sigma[0] / sigma[1])) <= 0.01:
    # use pooled standard deviations
    sexbarDiff = np.sqrt(((sp**2) / n[0]) + ((sp**2 / n[1])))
    print(
        "Population standard deviations are close enough. Use"
        + " pooled standard deviation, sp = "
        + str(round(sp, 4))
    )
else:
    sexbarDiff = np.sqrt(((sx[0] ** 2) / n[0]) + ((sx[1] ** 2 / n[1])))

# calculate a 95% confidence interval
df = min([num - 1 for num in n])
t = abs(round(stats.t.ppf(1 - (alpha / 2), df)))
cilo = round(xbar[0] - xbar[1] - (t * sexbarDiff), 4)
cihi = round(xbar[0] - xbar[1] + (t * sexbarDiff), 4)
print("two-tailed 95% confidence interval = (" + str(cilo) + ", " + str(cihi) + ")")

# calculate a p-value
t = abs((xbar[0] - xbar[1] - nullvalue) / sexbarDiff)
pvalue = round((1 - stats.t.cdf(t, df)) * 2, 4)
print("pvalue = " + str(pvalue))

# state results of the hypothesis tests
if (nullvalue >= cilo and nullvalue <= cihi) or (pvalue >= alpha):
    print("Fail to reject null hypothesis H0")
else:
    print("Reject null hypothesis H0")
print("")

# TODO: explore power calculations for a difference of means
