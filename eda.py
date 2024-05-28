# exec(open('eda.py').read())
import importlib as il
import os
import pickle as pk
import subprocess as sp

import datacfg
import dfutl
import matplotlib.pyplot as plt
import numpy as np
import plots
import scipy.stats as stats
import seaborn as sns
from matplotlib import cm
from matplotlib.patches import Polygon


# front line exploratory data analysis function
def eda(
    filepath: str,
    features=None,
    targets=None,
    removeOutliers: bool = False,
    datasetname: str = "",
):

    # load the data
    df = pk.load(open(filepath, "rb"))

    # TODO: properly infer if features or targets are a sequence or a single string
    if features is None:
        features = list(set(df.columns) - set(targets))

    # examine the data
    print("----------------------------------------------------------------------")
    print("{0}Shape of dataset:".format("    "))
    print("----------------------------------------------------------------------")
    print("{0}Number of Rows: {1}".format("    ", df.shape[0]))
    print("{0}Number of Columns: {1}".format("    ", df.shape[1]))
    print("", end="\n\n\n")

    print("----------------------------------------------------------------------")
    print("{0}Column names:".format("    "))
    print("----------------------------------------------------------------------")
    for col in df.columns:
        print("{0}{1}".format("    ", col))
    print("", end="\n\n\n")

    print("----------------------------------------------------------------------")
    print("{0}First 10 rows:".format("    "))
    print("----------------------------------------------------------------------")
    print(df.head(10))
    print("", end="\n\n\n")

    print("----------------------------------------------------------------------")
    print("{0}Last 10 rows:".format("    "))
    print("----------------------------------------------------------------------")
    print(df.tail(10))
    print("", end="\n\n\n")

    print("----------------------------------------------------------------------")
    print("{0}Statistical Summary:".format("    "))
    print("----------------------------------------------------------------------")
    print(df.describe())
    print("", end="\n\n\n")

    # ----------------------------------------------------------------------
    # infer data types of the input DataFrame
    # ----------------------------------------------------------------------
    colNumeric = dfutl.numericColumns(df)

    # ----------------------------------------------------------------------
    # mean centering and scaling: standardize or normalize
    # ----------------------------------------------------------------------
    dfNumeric = df.loc[:, colNumeric]
    df.loc[:, colNumeric] = (dfNumeric - dfNumeric.mean()) / dfNumeric.std()
    dfNumeric = df.loc[:, colNumeric]

    # ----------------------------------------------------------------------
    # outlier detection
    # ----------------------------------------------------------------------
    # use z-score filtering
    # samples that are more than 3 standard deviations away from mean are to be discarded
    print("----------------------------------------------------------------------")
    print("{0}Outlier Detection:".format("    "))
    print("----------------------------------------------------------------------")
    numouttotal = 0
    numout = 1
    passNum = 0
    while numout > 0:
        # determine the number of outliers using zscore
        zscores = stats.zscore(dfNumeric)
        idx = np.logical_not(np.logical_or(zscores < -3, zscores > 3))
        idxrows = np.all(idx, axis=1)
        idxrowsout = np.logical_not(idxrows)
        numout = len(idxrows) - len(idxrows[idxrows])

        print("{0}Pass {1} detected {2} outliers".format("    ", passNum, numout))
        if not removeOutliers:
            break

        # remove outliers and contineu
        if numout > 0 and removeOutliers:
            df = df.loc[idxrows, :]
            dfNumeric = df.loc[:, colNumeric]

        numouttotal = numouttotal + numout
        passNum = passNum + 1
    if removeOutliers:
        print("{0}Total number of outliers: {1}".format("    ", numouttotal))
    print("", end="\n\n\n")

    # ----------------------------------------------------------------------
    # visualization
    # ----------------------------------------------------------------------
    plt.close("all")

    save = True
    if len(datasetname) > 0:
        savepath = ".\\png\\{0}\\eda\\".format(datasetname)
        isdir = os.path.isdir(savepath)
        if not isdir:
            os.makedirs(savepath)
    else:
        savepath = ".\\png\\"

    plots.boxplot(dfNumeric, save=save, savepath=savepath)
    plots.histogram(df, tightLayout=True, save=save, savepath=savepath)
    plots.scattermatrix(dfNumeric, save=save, savepath=savepath)
    plots.heatmap(dfNumeric, correlation=0.5, save=save, savepath=savepath)

    # plt.show()
    plt.close("all")

    return df


if __name__ == "__main__":
    # specify the following variables
    cfg = dict()
    cfg = datacfg.datacfg()

    # for all datasets
    for datasetname in cfg.keys():
        filepath = cfg[datasetname]["filepath"]
        features = cfg[datasetname]["features"]
        targets = cfg[datasetname]["targets"]
        removeOutliers = cfg[datasetname]["removeOutliers"]

        df = eda(
            filepath=filepath,
            features=features,
            targets=targets,
            removeOutliers=removeOutliers,
            datasetname=datasetname,
        )
