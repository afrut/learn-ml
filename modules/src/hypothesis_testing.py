import math

import scipy.stats as stats


# ------------------------------------------------------------
# Type 2 Error for a two-sided hypothesis test
# ------------------------------------------------------------
def twoTailT2(alpha, mua, mu0, n, sigma):
    delta = mua - mu0
    return stats.norm.cdf(
        stats.norm.ppf(1 - (alpha / 2)) - ((delta * math.sqrt(n)) / sigma)
    ) - stats.norm.cdf(stats.norm.ppf(alpha / 2) - ((delta * math.sqrt(n)) / sigma))


# ------------------------------------------------------------
# Type 2 Error for a one-sided lower-bound hypothesis test
# ------------------------------------------------------------
def oneTailT2Lo(alpha, mua, mu0, n, sigma):
    delta = mua - mu0
    return 1 - stats.norm.cdf(stats.norm.ppf(alpha) - ((delta * math.sqrt(n)) / sigma))


# ------------------------------------------------------------
# Type 2 Error for a one-sided upper-bound hypothesis test
# ------------------------------------------------------------
def oneTailT2Hi(alpha, mua, mu0, n, sigma):
    delta = mua - mu0
    return stats.norm.cdf(stats.norm.ppf(1 - alpha) - ((delta * math.sqrt(n)) / sigma))


# ------------------------------------------------------------
# Sample size to control Type 2 Error for a two-sided hypothesis test
# ------------------------------------------------------------
def twoTailT2N(alpha, mua, mu0, beta, sigma):
    delta = mua - mu0
    if delta > 0:
        return (
            sigma * (stats.norm.ppf(1 - (alpha / 2)) - stats.norm.ppf(beta)) / delta
        ) ** 2
    else:
        return (
            sigma * (stats.norm.ppf(alpha / 2) - stats.norm.ppf(1 - beta)) / delta
        ) ** 2


# ------------------------------------------------------------
# Sample size to control Type 2 Error for a one-sided lower-bound hypothesis test
# ------------------------------------------------------------
def oneTailT2NLo(alpha, mua, mu0, beta, sigma):
    delta = mua - mu0
    return (sigma * (stats.norm.ppf(alpha) - stats.norm.ppf(1 - beta)) / delta) ** 2


# ------------------------------------------------------------
# Sample size to control Type 2 Error for a one-sided upper-bound hypothesis test
# ------------------------------------------------------------
def oneTailT2NHi(alpha, mua, mu0, beta, sigma):
    delta = mua - mu0
    return (sigma * (stats.norm.ppf(1 - alpha) - stats.norm.ppf(beta)) / delta) ** 2


# ------------------------------------------------------------
# p-value for a two-tail hypothesis test
# ------------------------------------------------------------
def twoTailPvalue(n, mu0, sampmean, sigma=None, sampstd=None):
    ret = None
    if sigma is not None:
        if sampmean > mu0:
            ret = (
                1 - stats.norm.cdf(sampmean, loc=mu0, scale=sigma / math.sqrt(n))
            ) * 2
        else:
            ret = stats.norm.cdf(sampmean, loc=mu0, scale=sigma / math.sqrt(n)) * 2
    elif sampstd is not None:
        if sampmean > mu0:
            ret = (
                1
                - stats.t.cdf(sampmean, df=n - 1, loc=mu0, scale=sampstd / math.sqrt(n))
            ) * 2
        else:
            ret = (
                stats.t.cdf(sampmean, df=n - 1, loc=mu0, scale=sampstd / math.sqrt(n))
                * 2
            )
    return ret


# ------------------------------------------------------------
# p-value for a one-tail lower-bound hypothesis test
# ------------------------------------------------------------
def oneTailPvalueLo(n, mu0, sampmean, sigma=None, sampstd=None):
    ret = None
    if sigma is not None:
        ret = stats.norm.cdf(sampmean, loc=mu0, scale=sigma / math.sqrt(n))
    elif sampstd is not None:
        ret = stats.t.cdf(sampmean, df=n - 1, loc=mu0, scale=sampstd / math.sqrt(n))
    return ret


# ------------------------------------------------------------
# p-value for a one-tail upper-bound hypothesis test
# ------------------------------------------------------------
def oneTailPvalueHi(n, mu0, sampmean, sigma=None, sampstd=None):
    ret = None
    if sigma is not None:
        ret = 1 - stats.norm.cdf(sampmean, loc=mu0, scale=sigma / math.sqrt(n))
    elif sampstd is not None:
        ret = 1 - stats.t.cdf(sampmean, df=n - 1, loc=mu0, scale=sampstd / math.sqrt(n))
    return ret
