#exec(open('.\\templates\\plot_scattermatrix.py').read())
import subprocess as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

if __name__ == '__main__':
    sp.call('cls', shell = True)

    # close all plots
    plt.close('all')

    with open('.\\data\\iris\\iris.data', 'rt') as fl:
        df = pd.read_csv(fl
            ,names = ['sepal_length','sepal_width','petal_length','petal_width','class']
            ,header = None
            ,index_col = False)

    # get only numeric columns
    numeric = df.select_dtypes([np.number]).columns
    df = df.loc[:, numeric]

    # create a figure
    fig = plt.figure(figsize = (14.4, 9))

    # add a subplot
    ax = fig.add_subplot(1,1,1)

    # create the plot
    axes = pd.plotting.scatter_matrix(df, ax = ax)

    # format x and y axis labels
    for x in range(axes.shape[0]):
        for y in range(axes.shape[1]):
            ax = axes[x,y]
            ax.xaxis.label.set_rotation(30)
            ax.yaxis.label.set_rotation(0)
            ax.yaxis.labelpad = 50      # increase the distance from ylabel to yaxis

    # title for the whole figure
    fig.suptitle("Iris Scatter Matrix Plot")

    # save the plot as a file
    fig.savefig('.\\iris_scattermatrix.png', format = 'png')
    os.remove('.\\iris_scattermatrix.png')

    # show the plot
    plt.show()
    plt.close('all')