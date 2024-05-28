#exec(open('templates\\eda.py').read())
# TODO: add visualization for non-numeric data
import subprocess as sp
import pandas as pd
import numpy as np
import pickle as pk
import math
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sp.call('cls', shell = True)

    # load some data
    with open('.\\data\\pima.pkl','rb') as fl:
        df = pk.load(fl)

    # ----------------------------------------
    # Constants
    # ----------------------------------------
    np.set_printoptions(precision = 4, suppress = True)
    seed = 29
    figsize = (16, 10)

    # ----------------------------------------
    # Data Inspection
    # ----------------------------------------
    # shape of the dataset
    print('Number of Rows: {0}'.format(df.shape[0]))
    print('Number of Columns: {0}'.format(df.shape[1]))
    print('')

    print('Column Names and Data Types:')
    datatypes = df.dtypes
    for idx in datatypes.index:
        print('    {0} - {1}'.format(idx, datatypes[idx]))
    print('')

    print('First 20 rows:')
    print(df.head(20))
    print('')

    print('Last 20 rows:')
    print(df.tail(20))
    print('')

    print('Class sizes:')
    print(df.groupby(['class']).size())
    print('')

    print('Statistical Summary:')
    print(df.describe())
    print('')

    print('Correlations between variables:')
    print(df.corr())
    print('')

    # numbers closer to 0 mean the distribution is closer to Gaussian
    print('Skew of variables:')
    print(df.skew())
    print('')

    # ----------------------------------------
    # Data Visualization
    # ----------------------------------------
    # get numeric types
    numerics = df.select_dtypes([np.number]).columns.to_numpy()

    # get non-numeric types
    nonnum = list(set(df.columns) - set(numerics))
    nonnum = np.array(nonnum)

    # determine layout of univariate plots
    numvar = len(numerics)
    numrows = int(math.sqrt(numvar))
    numcols = numrows
    while(numrows * numcols < numvar):
        numcols = numcols + 1
    layout = (numrows, numcols)

    # get only numeric data
    dfnumeric = df.loc[:, numerics]

    # matrix of histograms
    ret = dfnumeric.hist(figsize = figsize)

    # matrix of probability density functions
    ret = dfnumeric.plot(kind = 'density', subplots = True, layout = layout, sharex = False, figsize = figsize)
    
    # box and whisker plot for all numeric quantities
    ret = dfnumeric.plot(kind = 'box', subplots = True, layout = layout, sharex = False, sharey = False, figsize = figsize)
    
    # matrix/heatmap of correlations
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(1, 1, 1)

    # plot the matrix of correlations, returns the colorbar image
    axesimage = ax.matshow(dfnumeric.corr(), vmin = -1, vmax = 1)

    # plot the colorbar image
    fig.colorbar(axesimage)

    # set x and y-axis ticks and ticklabels
    ticks = np.arange(0, len(dfnumeric.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(dfnumeric.columns)
    ax.set_yticklabels(dfnumeric.columns)

    # scatter matrix plot
    pd.plotting.scatter_matrix(dfnumeric, figsize = figsize)

    # plot time series data according to time

    plt.show()
    plt.close('all')