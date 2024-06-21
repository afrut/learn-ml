import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plots
import scipy.stats as stats

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.abspath(os.path.join(script_dir, "../visualization/outputs"))

    # ----------------------------------------------------------------------
    # specify a data set
    # x = np.array([1,2,4,6,8,10,12,14,16,18,20])
    # y = np.array([0.6,0.7,1.5,1.9,2.1,2.6,2.9,3.7,3.5,3.7,3.8])
    x = np.array([0, 4, 11, 6, 14])
    y = np.array([9, 7, 6, 4, 1])

    n = len(x)
    sxi = x.sum()
    sxi2 = (x**2).sum()
    syi = y.sum()
    syi2 = (y**2).sum()
    sxiyi = (x * y).sum()

    # or specify some sample statistics
    # n = 250
    # sxi = 6322.28
    # sxi2 = 162674.18
    # syi = 4757.90
    # syi2 = 107679.27
    # sxiyi = 125471.10

    # specify xo around which to calculate the mean response and prediction interval
    xo = np.array([8])

    # specify a significance level
    alpha = 0.05
    # ----------------------------------------------------------------------

    # calculate preliminary values
    ybar = syi / n
    xbar = sxi / n
    sxx = sxi2 - (sxi**2 / n)
    sxy = sxiyi - ((sxi * syi) / n)
    xbar = sxi / n
    ybar = syi / n
    print("n = {0}".format(n))
    print("sxx = {0}".format(round(sxx, 6)))
    print("sxy = {0}".format(round(sxy, 6)))
    print("xbar = {0}".format(round(xbar, 6)))
    print("ybar = {0}".format(round(ybar, 6)))

    # calculate regression coefficients
    b1 = sxy / sxx
    b0 = ybar - (b1 * xbar)
    print("b0 = {0}".format(round(b0, 6)))
    print("b1 = {0}".format(round(b1, 6)))
    print("")

    # calculate sum of squares and mean squares
    sst = syi2 - (n * ybar**2)
    sse = sst - (b1 * sxy)
    ssr = b1 * sxy
    msr = ssr / 1
    mse = sse / (n - 2)
    print("sst = {0}".format(round(sst, 6)))
    print("sse = {0}".format(round(sse, 6)))
    print("ssr = {0}".format(round(ssr, 6)))
    print("msr = {0}".format(round(msr, 6)))
    print("mse = {0}".format(round(mse, 6)))
    print("")

    # estimate standard deviation of errors
    sigma = math.sqrt(sse / (n - 2))
    print("sigma2 = {0}".format(round(sigma**2, 6)))
    print("")

    # test significance by t-test of b0
    varb0 = sigma**2 * ((1 / n) + (xbar**2 / sxx))
    seb0 = math.sqrt(varb0)
    tb0 = (b0 - 0) / seb0
    tb0lo = stats.t.ppf(alpha / 2, df=n - 2)
    tb0hi = stats.t.ppf(1 - (alpha / 2), df=n - 2)
    b0lo = b0 + (tb0lo * seb0)
    b0hi = b0 + (tb0hi * seb0)
    print("varb0 = {0}".format(round(varb0, 6)))
    print("seb0 = {0}".format(round(seb0, 6)))
    print("tb0 = {0}".format(round(tb0, 6)))
    print("tb0lo = {0}".format(round(tb0lo, 6)))
    print("tb0hi = {0}".format(round(tb0hi, 6)))
    print("b0lo = {0}".format(round(b0lo, 6)))
    print("b0hi = {0}".format(round(b0hi, 6)))
    if tb0 < tb0lo or tb0 > tb0hi:
        print("b0 is significant")
    else:
        print("b0 is not significant")
    print("")

    # test significance of b2
    varb1 = sigma**2 / sxx
    seb1 = math.sqrt(varb1)
    tb1 = (b1 - 0) / seb1
    tb1lo = stats.t.ppf(alpha / 2, df=n - 2)
    tb1hi = stats.t.ppf(1 - (alpha / 2), df=n - 2)
    b1lo = b1 + (tb1lo * seb1)
    b1hi = b1 + (tb1hi * seb1)
    print("varb1 = {0}".format(round(varb1, 6)))
    print("seb1 = {0}".format(round(seb1, 6)))
    print("tb1 = {0}".format(round(tb1, 6)))
    print("tb1lo = {0}".format(round(tb1lo, 6)))
    print("tb1hi = {0}".format(round(tb1hi, 6)))
    print("b1lo = {0}".format(round(b1lo, 6)))
    print("b1hi = {0}".format(round(b1hi, 6)))
    if tb0 < tb0lo or tb0 > tb0hi:
        print("b1 is significant")
    else:
        print("b1 is not significant")
    print("")

    # test significance of regression through anova
    f = (ssr / 1) / (sse / (n - 2))
    fcrit = stats.f.ppf(1 - alpha, 1, n - 2)
    print("f = {0}".format(round(f, 6)))
    print("fcrit = {0}".format(round(fcrit, 6)))
    if f > fcrit:
        print("regression is significant")
    else:
        print("regression is not significant")
    print("")

    # confidence intervals on the mean response
    ypredxo = b0 + (b1 * xo)
    varmu = sigma**2 * ((1 / n) + ((xo - xbar) ** 2 / sxx))
    semu = np.sqrt(varmu)
    mulo = ypredxo + (stats.t.ppf(alpha / 2, df=n - 2) * semu)
    muhi = ypredxo + (stats.t.ppf(1 - (alpha / 2), df=n - 2) * semu)
    print("mulo = {0}".format(mulo))
    print("ypredxo = {0}".format(ypredxo))
    print("muhi = {0}".format(muhi))
    print("")

    # predicition interval
    varpred = sigma**2 * (1 + (1 / n) + ((xo - xbar) ** 2 / sxx))
    sepred = np.sqrt(varpred)
    ypredlo = ypredxo + (stats.t.ppf(alpha / 2, df=n - 2) * sepred)
    ypredhi = ypredxo + (stats.t.ppf(1 - (alpha / 2), df=n - 2) * sepred)
    print("ypredlo = {0}".format(ypredlo))
    print("ypredxo = {0}".format(ypredxo))
    print("ypredhi = {0}".format(ypredhi))
    print("")

    # coefficient of determination R2
    R2 = ssr / sst
    print("R2 = {0}".format(round(R2, 6)))
    print("")

    # correlation coefficient
    rho = sxy / (sxx * sst) ** (1 / 2)
    print("rho = {0}".format(round(rho, 6)))
    print("")

    try:
        # calculate all predictions
        ypred = b0 + (b1 * x)

        # standardize residuals
        res = (y - ypred) / sigma

        # normal probability plot of residuals
        df = pd.DataFrame(res, columns=["e"])
        plots.probplot(df, title="Normal Probability Plot of Residuals")

        # plot residuals against predictions
        fig, ax = plots.scatter(x, res, title="Residuals vs x")
        ax.plot(np.array([x.min(), x.max()]), np.array([0, 0]), linewidth=2)
        fig, ax = plots.scatter(ypred, res, title="Residuals vs y")
        ax.plot(np.array([ypred.min(), ypred.max()]), np.array([0, 0]), linewidth=2)

        # plot predicted vs actual values
        fig, ax = plots.scatter(
            x, ypred, linewidth=3, markersize=0, title="Actual vs Predicted"
        )
        ax.scatter(x, y)
        plt.savefig(f"{save_path}/simple_ols.png", format="png")
    except NameError:
        pass
