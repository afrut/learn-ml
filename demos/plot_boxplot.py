#exec(open('.\\templates\\plot_boxplot.py').read())
import subprocess as sp
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

if __name__ == '__main__':
    sp.call('cls', shell = True)

    # close all figures
    plt.close('all')

    with open('.\\data\\iris\\iris.data', 'rt') as fl:
        df = pd.read_csv(fl
            ,names = ['sepal_length','sepal_width','petal_length','petal_width','class']
            ,header = None
            ,index_col = False)
        df = df.loc[:, df.select_dtypes([np.number]).columns]

    # create a new figure
    fig = plt.figure(figsize = (14.4, 9))

    # add subplot to the figure
    ax = fig.add_subplot(1, 1, 1)

    # create the boxplot with seaborn
    lines = ax.boxplot(df.values)

    # title of the plot
    ax.set_title('Iris Dataset Boxplot')

    # set xticklabels
    ax.set_xticklabels(df.columns)

    # rotate xticklabels
    for ticklabel in ax.get_xticklabels():
        ticklabel.set_horizontalalignment('right')
        ticklabel.set_rotation_mode('anchor')
        ticklabel.set_rotation(30)
        ticklabel.set_fontsize(10)

    # reduce margins
    fig.tight_layout()

    # save the plot as a file
    fig.savefig('.\\iris_boxplot.png', format = 'png')
    os.remove('.\\iris_boxplot.png')

    # show the plot
    plt.show()
    plt.close('all')