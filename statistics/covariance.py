import math
import os
import pickle as pk

import dfutl
import numpy as np
import pandas as pd

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.abspath(
        os.path.join(script_dir, "./distributions/responseTime.csv")
    )
    df = pd.read_csv(file_path, header=0, index_col="resp_time")

    # Given a bivariate joint probability distribution, the covariance
    # and correlation of with respect to both variables can be calculated as
    # follows:

    # Calculate the marginal probability distribution of both variables.
    mpdX = df.sum(axis=0)
    mpdY = df.sum(axis=1)

    # Calculate the expectation of both variables.
    Ex = (mpdX.index.values.astype(float) * mpdX.values).sum()
    Ey = (mpdY.index.values.astype(float) * mpdY.values).sum()

    # sigmaXY = E[(X - muX)(Y - muY)]
    sigmaXY = 0
    for xcol in df.columns:
        xval = float(xcol)
        for yval in df.index:
            sigmaXY = sigmaXY + (xval - Ex) * (yval - Ey) * df.loc[yval, xcol]

    # sigmaXY2 = E(XY) - E(X)E(Y)
    sigmaXY2 = 0
    for xcol in df.columns:
        xval = float(xcol)
        for yval in df.index:
            sigmaXY2 = sigmaXY2 + (xval) * (yval) * df.loc[yval, xcol]
    sigmaXY2 = sigmaXY2 - (Ex * Ey)
    assert abs(sigmaXY - sigmaXY2) < 1e-8

    # Calculate the variance and standard deviation of X and Y.
    sigma2X = ((mpdX.index.values.astype(float) - Ex) ** 2 * mpdX.values).sum()
    sigma2Y = ((mpdY.index.values.astype(float) - Ey) ** 2 * mpdY.values).sum()
    sigmaX = math.sqrt(sigma2X)
    sigmaY = math.sqrt(sigma2Y)
    corrXY = sigmaXY / (sigmaX * sigmaY)

    print("Given a joint probability distribution:")
    print(df)
    print("")
    print("cov(X, Y) = {0:.8}".format(sigmaXY))
    print("corr(X, Y) = {0:.8}".format(corrXY))
    print("\n\n\n\n")

    file_path = os.path.abspath(os.path.join(script_dir, "../data/iris.pkl"))
    with open(file_path, "rb") as fl:
        df = pk.load(fl)
        cols = dfutl.numericColumns(df)
        df = df.loc[:, cols]

    # Given a data set of observations, the covariance and correlation
    # can be calculated in the following manner.

    # Calculate covariances.
    cov = df.cov()
    cov2 = np.cov(np.transpose(df.values))
    assert np.all(abs(cov.values - cov2) < 1e-8)

    # Calculate correlations.
    corr = df.corr()
    corr2 = np.corrcoef(np.transpose(df.values))
    assert np.all(abs(corr.values - corr2) < 1e-8)

    print("Given a set of observations of multiple variables:")
    print(df)
    print("")
    print("The pair-wise covariances are:")
    print(cov)
    print("")
    print("The pair-wise correlations are:")
    print(corr)
    print("")
