#exec(open('.\\templates\\plot_annotate.py').read())
import subprocess as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
import random

if __name__ == '__main__':
    sp.call('cls', shell = True)

    # close all plots
    plt.close('all')

    # define some data
    x = np.arange(0, 2*np.pi, np.pi / 60)
    noise = (np.random.rand(len(x)) * 2) - 1
    y = (1.3 * x) + np.sin(1.5 * x) + noise

    fig = plt.figure(figsize = (14.4, 9))
    ax = fig.add_subplot(1, 1, 1)

    # create scatter plot
    lines = ax.plot(x, y
        ,linestyle = 'solid'
        ,linewidth = 3
        ,color = (130/255, 150/255, 200/255, 1)
        ,marker = None)

    # get the point to annotate
    idx = np.where(x > 3)[0][0]

    ax.annotate('something happened here'                       # text of annotation
        ,xycoords='data'                                        # specify coordinates in data's coordinates
        ,xy = (3, 2)                                            # coordinates of tip of arrow
        ,textcoords='data'                                      # coordinates of text location in data's coordinates
        ,xytext = (3.5, 1)                                      # coorindates of text annotation
        ,arrowprops = dict(facecolor = 'black', shrink = 0.05)  # shrink affects length of arrow; closer to 1 is shorter
        ,horizontalalignment = 'left'                           # horizontal alighment of text annotation
        ,verticalalignment = 'bottom')                          # vertical alignment of text annotation

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('A simple line plot')

    # minimize margins and whitespace
    fig.tight_layout()

    # save the plot as a file
    fig.savefig('.\\plot_annotate.png', format = 'png')
    os.remove('.\\plot_annotate.png')

    # show the plot
    plt.show()
    plt.close('all')