import math

import numpy as np
import scipy.stats as stats


# ----------------------------------------------------------------------
# Function to calculate the probability density function values of the normal
# distribution.
# ----------------------------------------------------------------------
def pdfnorm(x, mu, sigma):
    return (
        1 / (math.sqrt(2 * math.pi) * sigma) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    )


# ----------------------------------------------------------------------
# Two-tail confidence intervals
# ----------------------------------------------------------------------
def twoTail(alpha, n=None, sampmean=None, sigma=None, sampstd=None, samp=None):
    ret = None

    if n == None and samp is not None:
        n = len(samp)

    # check if a sample mean has been passed in
    if sampmean is None and samp is not None:
        sampmean = samp.mean()

    # check if the population standard deviation is unknown
    if sigma == None:

        # t distribution always needs the sample size for degrees of freedom
        assert n is not None

        # use t distribution
        tlo = stats.t.ppf(alpha / 2, df=n - 1)
        thi = stats.t.ppf(1 - (alpha / 2), df=n - 1)

        # check if sample standard deviation is provided or can be calculated
        if sampstd == None and samp is not None:
            sampstd = samp.std(ddof=1)

        # check if x values or t values are to be returned
        if sampmean is not None and sampstd is not None:
            xlo = sampmean + (tlo * sampstd)
            xhi = sampmean + (thi * sampstd)
            ret = (xlo, xhi)
        else:
            ret = (thi, tlo)

    else:

        # use standard normal distribution
        zlo = stats.norm.ppf(alpha / 2)
        zhi = stats.norm.ppf(1 - (alpha / 2))

        # check if x values or z values are to be returned
        if sampmean is not None:
            xlo = sampmean + (zlo * sigma)
            xhi = sampmean + (zhi * sigma)
            ret = (xlo, xhi)
        else:
            ret = (zlo, zhi)

    return ret


# ----------------------------------------------------------------------
# One-tail lower-bound confidence intervals
# ----------------------------------------------------------------------
def oneTailLo(alpha, n=None, sampmean=None, sigma=None, sampstd=None, samp=None):
    ret = None

    if n == None and samp is not None:
        n = len(samp)

    # check if a sample mean has been passed in
    if sampmean is None and samp is not None:
        sampmean = samp.mean()

    # check if the population standard deviation is unknown
    if sigma == None:

        # t distribution always needs the sample size for degrees of freedom
        assert n is not None

        # use t distribution
        tlo = stats.t.ppf(alpha, df=n - 1)

        # check if sample standard deviation is provided or can be calculated
        if sampstd == None and samp is not None:
            sampstd = samp.std(ddof=1)

        # check if x values or t values are to be returned
        if sampmean is not None and sampstd is not None:
            xlo = sampmean + (tlo * sampstd)
            ret = xlo
        else:
            ret = tlo

    else:

        # use standard normal distribution
        zlo = stats.norm.ppf(alpha)

        # check if x values or z values are to be returned
        if sampmean is not None:
            xlo = sampmean + (zlo * sigma)
            ret = xlo
        else:
            ret = zlo

    return ret


# ----------------------------------------------------------------------
# One-tail upper-bound confidence intervals
# ----------------------------------------------------------------------
def oneTailHi(alpha, n=None, sampmean=None, sigma=None, sampstd=None, samp=None):
    ret = None

    if n == None and samp is not None:
        n = len(samp)

    # check if a sample mean has been passed in
    if sampmean is None and samp is not None:
        sampmean = samp.mean()

    # check if the population standard deviation is unknown
    if sigma == None:

        # t distribution always needs the sample size for degrees of freedom
        assert n is not None

        # use t distribution
        thi = stats.t.ppf(1 - alpha, df=n - 1)

        # check if sample standard deviation is provided or can be calculated
        if sampstd == None and samp is not None:
            sampstd = samp.std(ddof=1)

        # check if x values or t values are to be returned
        if sampmean is not None and sampstd is not None:
            xhi = sampmean + (thi * sampstd)
            ret = xhi
        else:
            ret = thi

    else:

        # use standard normal distribution
        zhi = stats.norm.ppf(1 - alpha)

        # check if x values or z values are to be returned
        if sampmean is not None:
            xhi = sampmean + (zhi * sigma)
            ret = xhi
        else:
            ret = zhi

    return ret


if __name__ == "__main__":
    print("hello world")
