#exec(open('.\\templates\\plot_barplot.py').read())
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

    # get non-numeric data
    data = df.loc[:, df.select_dtypes([object]).columns].values.astype('str')

    # count unique values
    valsunq, counts = np.unique(data, return_counts = True)
    xticks = np.array([x for x in range(len(valsunq))])

    # create a new figure
    fig = plt.figure(figsize = (14.4, 9))

    # add subplot to the figure
    ax = fig.add_subplot(1, 1, 1)

    # create the boxplot with seaborn, return values and bins
    barcontainer = ax.bar(x = xticks, height = counts)

    # title of the plot
    ax.set_title('Count of Classes')

    # set the x-axis tick marks and labels
    ax.set_xticks(xticks)
    ax.set_xticklabels(valsunq)

    # format xticklabels
    for ticklabel in ax.get_xticklabels():
        ticklabel.set_horizontalalignment('right')
        ticklabel.set_rotation_mode('anchor')
        ticklabel.set_rotation(30)
        ticklabel.set_fontsize(10)

    # x and y-axis labels
    ax.set_xlabel('Classes')
    ax.set_ylabel('Counts')

    # minimize margins and whitespace
    fig.tight_layout()

    # save the plot as a file
    fig.savefig('.\\iris_class_counts.png', format = 'png')
    os.remove('.\\iris_class_counts.png')

    # show the plot
    plt.show()
    plt.close('all')