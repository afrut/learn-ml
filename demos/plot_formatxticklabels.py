#exec(open('.\\templates\\plot_formatxticklabels.py').read())
import subprocess as sp
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

if __name__ == '__main__':
    sp.call('cls', shell = True)
    plt.close('all')

    with open('.\\data\\iris\\iris.data', 'rt') as fl:
        df = pd.read_csv(fl
            ,names = ['sepal_length','sepal_width','petal_length','petal_width','class']
            ,header = None
            ,index_col = False)

    numeric = df.select_dtypes([np.number]).columns
    df = df.loc[:, numeric]

    fig = plt.figure(figsize = (14.4, 9))
    ax = fig.add_subplot(1, 1, 1)
    lines = ax.boxplot(df.values)
    ax.set_title('Unformatted xticklabels')
    ax.set_xticklabels(numeric)
    fig.tight_layout()
    fig.savefig('.\\Unformatted xticklabels.png', format = 'png')
    os.remove('.\\Unformatted xticklabels.png')

    fig = plt.figure(figsize = (14.4, 9))
    ax = fig.add_subplot(1, 1, 1)
    lines = ax.boxplot(df.values)
    ax.set_title('Formatted xticklabels')
    ax.set_xticklabels(numeric)

    # format every xticklabel
    for ticklabel in ax.get_xticklabels():
        ticklabel.set_horizontalalignment('right')  # center, right, left
        ticklabel.set_rotation_mode('anchor')       # None or anchor
        ticklabel.set_rotation(30)                  # angle of rotation
        ticklabel.set_fontsize(12)                  # float
    fig.tight_layout()
    fig.savefig('.\\Formatted xticklabels.png', format = 'png')
    os.remove('.\\Formatted xticklabels.png')

    fig = plt.figure(figsize = (14.4, 9))
    ax = fig.add_subplot(1, 1, 1)
    lines = ax.boxplot(df.values)
    ax.set_title('Custom xticks, yticks, and labels')
    fig.savefig('.\\Custom xticks, yticks, and labels.png', format = 'png')
    os.remove('.\\Custom xticks, yticks, and labels.png')

    # customize xticks and xticklabels
    xticks = np.arange(0, len(df.columns)) + 1
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    fig.tight_layout()

    plt.show()
    plt.close('all')