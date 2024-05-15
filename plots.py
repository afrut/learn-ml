#exec(open('plots.py').read())
import subprocess as sp
import matplotlib as mpl
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle as pk
import numpy as np

# general plotting flow
# - create figure: fig = plt.figure(figsize = figsize)
# - add subplots: ax = fig.add_subplot(nrows, ncols, index)
# - plot on axes: ax.plotting_function ie. boxplot
# - set axis title: ax.set_title(some_string)
# - set xlims: ax.set_xlim(xmin, xmax)
# - set_ylims: ax.set_ylim(ymin, ymax)
# - set xticks: ax.set_xticks
# - set xticklabels: ax.set_xticklabels
# - format xticklabels: formatxticklabels(ax)
# - set yticks: ax.set_yticks
# - set yticklabels: ax.set_yticklabels
# - format y ticklabels: formatxticklabels(ax)
# - set x-axis label: ax.set_xlabel(xlabel)
# - set y-axis label: ax.set_ylabel(ylabel)
# - add a grid: ax.grid(linewidth = 0.5)
# - tighten margins: fig.tight_layout()
# - add an overall figure title: ax.suptitle(some_title)
# - save: ax.savefig(some_path, format = 'png')
# - show: plt.show()
# - close: plt.close('all)

# constants
figsize = (14.4, 9)

# helper function to format xticklabels
def formatxticklabels(ax: mpl.axes.Axes
    ,horizontalalignment: str = 'right'
    ,rotationmode: str = 'anchor'
    ,xticklabelrotation: int = 30
    ,xticklabelfontsize: int = 10):
    for ticklabel in ax.get_xticklabels():
        ticklabel.set_horizontalalignment(horizontalalignment)
        ticklabel.set_rotation_mode(rotationmode)
        ticklabel.set_rotation(xticklabelrotation)
        ticklabel.set_fontsize(xticklabelfontsize)

# function to create a histogram plot
def hist(data
    ,binNum = 10
    ,binWdt = None
    ,figsize = figsize
    ,title = ''
    ,xlabel = ''
    ,ylabel = 'Frequency'
    ,savepath = None
    ,show = True
    ,close = True) -> mpl.figure:

    # create bins
    if binWdt is not None:
        bins = np.arange(data.min(), data.max() + binWdt, binWdt)
    else:
        bins = np.linspace(data.min(), data.max(), binNum)

    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(1, 1, 1)
    counts, bins, _ = ax.hist(data, bins = bins)
    ax.set_title(title)
    ax.set_xticks(bins)
    ax.set_xticklabels(bins)
    formatxticklabels(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(linewidth = 0.5)
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, format = 'png')
    if show:
        plt.show()
    if close:
        plt.close()

    return fig

# function to creat a bar plot
def bar(data
    ,xticks = None
    ,xticklabels = None
    ,figsize = figsize
    ,title = ''
    ,xlabel = ''
    ,ylabel = ''
    ,savepath = None
    ,show = True
    ,close = True) -> mpl.figure:
    if xticks is None:
        xticks = np.array(range(len(data)))
    if xticklabels is None:
        xticklabels = np.array(range(len(data)))
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(1, 1, 1)
    barcontainer = ax.bar(x = xticks, height = data)
    ax.set_title(title)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    formatxticklabels(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, format = 'png')
    if show:
        plt.show()
    if close:
        plt.close()

    return fig

if __name__ == '__main__':
    sp.call('cls', shell = True)
    plt.close('all')

    # ----------------------------------------
    # histogram
    # ----------------------------------------
    with open('.\\data\\pima.pkl','rb') as fl:
        df = pk.load(fl)

    data = df.iloc[:, 4]
    fig = hist(data, show = False, close = False)

    # xticklabels are formatted to have only 3 decimal places
    fig.axes[0].xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.3f}'))

    # ----------------------------------------
    # barplot
    # ----------------------------------------
    with open('.\\data\\iris.pkl', 'rb') as fl:
        df = pk.load(fl)

    # get non-numeric data
    data = df.loc[:, df.select_dtypes([object]).columns].values.astype('str')

    # count unique values
    valsunq, counts = np.unique(data, return_counts = True)
    fig = bar(data = counts, xticklabels = valsunq, title = 'Iris Classes')

    plt.show()
    plt.close('all')