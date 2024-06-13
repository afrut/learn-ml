#exec(open('.\\templates\\plot_line.py').read())
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

    # create a figure
    fig = plt.figure(figsize = (14.4, 9))

    # add a subplot
    ax = fig.add_subplot(1, 1, 1)

    # create scatter plot
    lines = ax.plot(x, y
        ,linestyle = 'solid'
        ,linewidth = 3
        ,color = (130/255, 150/255, 200/255, 1)
        ,marker = None)

    # set title
    ax.set_title('A simple line plot')

    # x and y axis labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # minimize margins and whitespace
    fig.tight_layout()

    # save the plot as a file
    fig.savefig('.\\simple_scatter.png', format = 'png')
    os.remove('.\\simple_scatter.png')

    # define more data
    noise = (np.random.rand(len(x)) * 2) - 1
    x2 = x + 1
    y2 = (2 * x) + np.sin(2 * x) + noise

    fig = plt.figure(figsize = (14.4, 9))
    ax = fig.add_subplot(1, 1, 1)
    lines = ax.plot(np.stack([x, x + 3, 2 * x], axis = 1)
        , np.stack([y, y2, (3 * y) - 7], axis = 1)
        ,marker = None
        ,linewidth = 3)
    ax.set_title('multiple lines in 1 subplot')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.tight_layout()

    # save the plot as a file
    fig.savefig('.\\multiple_lines.png', format = 'png')
    os.remove('.\\multiple_lines.png')

    # show the plot
    plt.show()
    plt.close('all')