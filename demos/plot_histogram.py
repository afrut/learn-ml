#exec(open('.\\templates\\plot_histogram.py').read())
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

    # specify a series for which a histogram is to be made
    sepal = df.loc[:, 'sepal_length']

    # specify bins by bin width
    binWdt = 0.5
    bins = np.arange(sepal.min(), sepal.max() + binWdt, binWdt)

    # specify bins by number of bins
    binNum = 12
    #bins = np.linspace(sepal.min(), sepal.max(), binNum)

    # create a new figure
    fig = plt.figure(figsize = (14.4, 9))

    # add subplot to the figure
    ax = fig.add_subplot(1, 1, 1)

    # create the boxplot with seaborn, return values and bins
    counts, bins, _ = ax.hist(sepal, bins = bins)

    # title of the plot
    ax.set_title('Iris Sepal Length Histogram')

    # set the x-axis tick marks and labels
    ax.set_xticks(bins)
    ax.set_xticklabels(bins)

    # set minimum and maximum y limits
    #ax.set_ylim(lsVals.min(), lsVals.max())

    # x and y-axis labels
    ax.set_xlabel('Sepal Lengths')
    ax.set_ylabel('Counts')

    # add a grid
    ax.grid(linewidth = 0.5)

    # minimize margins and whitespace
    fig.tight_layout()

    # save the plot as a file
    fig.savefig('.\\iris_sepal_hist.png', format = 'png')
    os.remove('.\\iris_sepal_hist.png')

    # show the plot
    plt.show()
    plt.close('all')