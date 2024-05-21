import math
import os
import pickle as pkl

import dfutl
import matplotlib.cm as cm
import matplotlib.figure as fgr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython import embed
from scipy import stats

colors = cm.get_cmap("tab20").colors
RED = colors[6]
LIGHT_RED = colors[7]
BLUE = colors[0]
LIGHT_BLUE = colors[1]
GREEN = colors[4]
LIGHT_GREEN = colors[5]
ORANGE = colors[2]
LIGHT_ORANGE = colors[3]
PURPLE = colors[9]
LIGHT_PURPLE = colors[10]
GRAY = colors[15]
LIGHT_GRAY = colors[16]


# ----------------------------------------
# helper function to format xticklabels
# ----------------------------------------
def formatxticklabels(
    ax,
    horizontalalignment: str = "right",
    rotationmode: str = "anchor",
    xticklabelrotation: int = 30,
    xticklabelfontsize: int = 10,
):
    for ticklabel in ax.get_xticklabels():
        ticklabel.set_horizontalalignment("right")
        ticklabel.set_rotation_mode("anchor")
        ticklabel.set_rotation(xticklabelrotation)
        ticklabel.set_fontsize(xticklabelfontsize)


# ----------------------------------------
# boxplot
# ----------------------------------------
def boxplot(
    df,
    figsize: tuple = (14.4, 9),
    title: str = None,
    save: bool = False,
    savepath: str = ".\\boxplot.png",
    show: bool = False,
    close: bool = False,
):

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    ax = sns.boxplot(data=df, ax=ax)
    ax.set_title(title)

    formatxticklabels(ax)

    if save:
        if savepath is not None and savepath[-1] == "\\":
            savepath = savepath + "boxplot.png"
        plt.savefig(savepath, format="png")

    if show:
        plt.show()

    if close:
        plt.close()

    return fig


# ----------------------------------------
# plot a histogram of certain variables
# ----------------------------------------
# Specification:
# - Take Dataframe as input and plot all columns.
# - Specify the number of bins to use between min and max.
# - If binWidth is provided, use this to determine the number of bins instead.
def histogram(
    df,
    fig=None,
    figsize: tuple = (14.4, 9),
    ax=None,
    numBins: int = 10,
    binWidth: float = None,
    ylim: list = None,
    xlabel: list = None,
    xlabelfontsize: int = 10,
    ylabel: list = None,
    xticklabelrotation: int = 30,
    tightLayout=True,
    title: str = None,
    save: bool = False,
    savepath: str = ".\\png\\histogram.png",
    show: bool = False,
    close: bool = False,
):

    numVar = len(df.columns)

    if ax is not None and fig is not None:
        plotOne = True
    else:
        plotOne = False

    # infer data types of the input DataFrame
    isNumeric = np.vectorize(lambda x: np.issubdtype(x, np.number))
    colNumeric = isNumeric(df.dtypes)

    # if inputs are valid
    if numVar > 0:
        # determine the number of rows and columns of subplots
        # cap number of columns at 4 columns
        ncols = min(int(math.ceil(math.sqrt(numVar))), 3)
        numplots = 0
        nrows = 0
        while numplots < numVar:
            nrows = nrows + 1
            numplots = nrows * ncols

        # Modify figsize. Every 3 plots = 9 in in height.
        if not (plotOne):
            figsize = (14.4, int(nrows * 3))
            fig = plt.figure(figsize=figsize)

        # loop through all variables and plot them on the corresponding axes
        for cntAx in range(0, numVar):

            # get the series for which the histogram is to be made
            srs = df.iloc[:, cntAx]

            # column is numeric - use histogram
            if colNumeric[cntAx]:

                # ----------------------------------------
                # infer the bins through the binWidth
                # ----------------------------------------
                if binWidth is not None:
                    # segregate data by the thickness of each bin
                    bins = list()
                    binStopVal = srs.max() + binWidth
                    binStartVal = srs.min()
                    bins = np.arange(binStartVal, binStopVal, binWidth)

                # ----------------------------------------
                # infer the bins through the number of bins
                # ----------------------------------------
                elif numBins is not None:
                    # segregate data by the number of bins
                    bins = np.linspace(srs.min(), srs.max(), numBins + 1)

                # ----------------------------------------
                # create the figure and plot
                # ----------------------------------------
                if not (plotOne):
                    ax = fig.add_subplot(nrows, ncols, cntAx + 1)
                lsVals, lsBins, _ = ax.hist(srs, bins=bins)

                # ----------------------------------------
                # format the plot
                # ----------------------------------------
                ax.set_xticks(lsBins)
                ax.set_xticklabels(np.round(lsBins, 4))
                ax.grid(linewidth=0.5)
                ax.set_title(title)

                ax.set_ylim((lsVals.min(), lsVals.max()))

                if xlabel is not None:
                    ax.set_xlabel(xlabel[cntAx])
                if ylabel is not None:
                    ax.set_ylabel(ylabel[cntAx])

                if title is not None:
                    ax.set_title(title)
                else:
                    ax.set_title(df.columns[cntAx])

            else:
                # column is not numeric - use barplot
                # create the figure and plot
                ax = fig.add_subplot(nrows, ncols, cntAx + 1)
                x = np.array(list(set(srs)))
                y = df.iloc[:, [cntAx]].groupby(df.columns[cntAx]).size()
                barplot(
                    x=x,
                    y=y,
                    fig=fig,
                    ax=ax,
                    grid=True,
                    title=df.columns[cntAx],
                    tightLayout=True,
                )

            # format xticklabels
            formatxticklabels(ax, xticklabelrotation=xticklabelrotation)

            if plotOne:
                break

        if fig is not None and tightLayout:
            fig.tight_layout()

        if save:
            if savepath is not None and savepath[-1:] == "\\":
                savepath = savepath + "histogram.png"
            plt.savefig(savepath, format="png")

        if show:
            plt.show()

        if close:
            plt.close()


# ----------------------------------------
# scatter matrix plot
# ----------------------------------------
def scattermatrix(
    df,
    figsize: tuple = (14.4, 9),
    title: str = None,
    save: bool = False,
    savepath: str = ".\\scattermatrix.png",
    show: bool = False,
    close: bool = False,
):

    dfTemp = df.loc[:, dfutl.numericColumns(df)]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    axes = pd.plotting.scatter_matrix(dfTemp, ax=ax)

    if title is not None:
        ax.set_title(title)

    # format x and y axis labels
    for x in range(axes.shape[0]):
        for y in range(axes.shape[1]):
            ax = axes[x, y]
            ax.xaxis.label.set_rotation(30)
            ax.yaxis.label.set_rotation(0)
            ax.yaxis.labelpad = 50

    if save:
        if savepath is not None and savepath[-1:] == "\\":
            savepath = savepath + "scattermatrix.png"
        plt.savefig(savepath, format="png")

    if show:
        plt.show()

    if close:
        plt.close()


# ----------------------------------------
# plot a heat map
# ----------------------------------------
# Specification:
# - Computes the correlation of every column wrt to every other column in the Dataframe.
# - Set correlations to 0 when within a certain threshold.
def heatmap(
    df,
    figsize: tuple = (14.4, 9),
    correlation: float = None,
    xcolumns: list = None,
    ycolumns: list = None,
    title: str = None,
    save: bool = False,
    savepath: str = ".\\heatmap.png",
    show: bool = False,
    close: bool = False,
):

    # prepare variables for rows and columns
    if xcolumns is None:
        xcolumns = dfutl.numericColumns(df)
    else:
        xcolumns = dfutl.numericColumns(df.loc[:, xcolumns])
    if ycolumns is None:
        ycolumns = dfutl.numericColumns(df)
    else:
        ycolumns = dfutl.numericColumns(df.loc[:, ycolumns])

    # calculate correlations
    dfCorr = df.loc[:, dfutl.numericColumns(df)].corr()
    dfCorr = dfCorr.loc[xcolumns, ycolumns]

    # bi-directionally mask correlations that are less than a certain threshold
    if correlation is not None:
        mask = dfCorr <= correlation
        mask = mask & (dfCorr >= correlation * -1)
        dfCorrMask = dfCorr.mask(mask, 0)
    else:
        dfCorrMask = dfCorr

    # heat map of correlations
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    ax = sns.heatmap(
        data=dfCorrMask, vmin=-1, vmax=1, annot=True, annot_kws=dict([("fontsize", 6)])
    )
    ax.set_yticks([x + 0.5 for x in range(0, len(dfCorrMask.index))])
    ax.set_yticklabels(dfCorrMask.index)
    ax.set_xticks([x + 0.5 for x in range(0, len(dfCorrMask.columns))])
    ax.set_xticklabels(dfCorrMask.columns)

    formatxticklabels(ax)

    if title is None and correlation is not None:
        title = "Correlation Threshold = {0:.3f}".format(correlation)
    ax.set_title(title)

    if save:
        if savepath is not None and savepath[-1:] == "\\":
            savepath = savepath + "heatmap.png"
        plt.savefig(savepath, format="png")

    if show:
        plt.show()

    if close:
        plt.close()


# ----------------------------------------
# simple scatter plot between two quantities
# ----------------------------------------
def scatter(
    x,
    y,
    fig=None,
    ax=None,
    figsize: tuple = (14.4, 9),
    ylim=None,
    xlabel: str = None,
    ylabel: str = None,
    marker: str = "o",
    markersize: int = 5,
    markeredgewidth: float = 0.4,
    markeredgecolor: tuple = BLUE,
    linewidth: int = 0,
    color: tuple = BLUE,
    grid: bool = False,
    tightLayout: bool = True,
    title: str = None,
    save: bool = False,
    savepath: str = ".\\scatterplot.png",
    show: bool = False,
    close: bool = False,
):
    if fig is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
    elif ax is not None and fig is not None:
        pass
    else:
        ax = fig.get_axes()[0]

    # check shape of data passed in
    oneX = len(x.shape) == 1 or min(x.shape) == 1
    multX = len(x.shape) == 2 and x.shape[1] > 1
    oneY = len(y.shape) == 1 or min(y.shape) == 1
    multY = len(y.shape) == 2 and y.shape[1] > 1

    if oneX:
        if oneY:
            lines = ax.plot(
                x,
                y,
                marker=marker,
                markersize=markersize,
                linewidth=linewidth,
                markeredgewidth=markeredgewidth,
                markeredgecolor=markeredgecolor,
                color=color,
            )
        elif multY:
            numY = y.shape[1]
            colors = cm.brg(np.linspace(0, 1, numY))
            for cnt in range(0, numY):
                lines = ax.plot(
                    x,
                    y[:, cnt],
                    marker=marker,
                    markersize=markersize,
                    linewidth=linewidth,
                    markeredgewidth=markeredgewidth,
                    markeredgecolor=markeredgecolor,
                    color=colors[cnt],
                )
        else:
            print("Invalid input shapes x = {0}, y = {1}".format(x.shape, y.shape))
    elif multX and multY:
        numX = x.shape[1]
        numY = y.shape[1]
        if numX == numY:
            colors = cm.brg(np.linspace(0, 1, numY))
            for cnt in range(0, numY):
                lines = ax.plot(
                    x[:, cnt],
                    y[:, cnt],
                    marker=marker,
                    markersize=markersize,
                    linewidth=linewidth,
                    markeredgewidth=markeredgewidth,
                    markeredgecolor=colors[cnt],
                    color=colors[cnt],
                )
        else:
            print("Invalid input shapes x = {0}, y = {1}".format(x.shape, y.shape))
    else:
        print("Invalid input shapes x = {0}, y = {1}".format(x.shape, y.shape))

    if ylim == None:
        pass
    else:
        ax.set_ylim(ylim)

    if title is None:
        title = "{} vs {}".format(ylabel, xlabel)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.xaxis.label.set_text(xlabel)
    if ylabel is not None:
        ax.yaxis.label.set_text(ylabel)
    if grid:
        ax.grid(grid, linewidth=0.5)

    if save:
        if savepath[-1:] == "\\":
            savepath = savepath + "scatterplot.png"
        plt.savefig(savepath, format="png")
    if show:
        plt.show()

    if close:
        plt.close()

    return (fig, ax)


# ----------------------------------------
# plot a bar chart with text labels
# ----------------------------------------
# TODO: rotation of xticklabels
def barplot(
    x=None,
    y=None,
    yLine=None,
    fig=None,
    figsize: tuple = (14.4, 9),
    ax=None,
    ylim=None,
    width: float = 1,
    xticklabels=None,
    xlabel: str = None,
    ylabel: str = None,
    yLineLabel: str = None,
    yLineLim=None,
    align="center",
    grid=False,
    title: str = None,
    tightLayout=False,
    edgecolor=None,
    save: bool = False,
    savepath: str = ".\\barplot.png",
    show: bool = False,
    close: bool = False,
):

    # Internal Function
    # Attach a text label above each bar in *rects*, displaying its height."""
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                "{}".format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    # ----------------------------------------------------------------------
    # process inputs
    # ----------------------------------------------------------------------
    if y is None:
        y = np.arange(0, 10)
        x = np.arange(len(y))
    elif x is None:
        x = np.arange(len(y))

    # ----------------------------------------------------------------------
    # actual plot
    # ----------------------------------------------------------------------
    if fig is None:
        fig = plt.figure(figsize=figsize)
    if ax is None:
        ax = fig.add_subplot(1, 1, 1)
    else:
        ax = ax
    rects = ax.bar(x, y, width=width, align=align, edgecolor=edgecolor)

    # plot a line if appropriate
    if yLine is not None:
        ax2 = ax.twinx()
        ax2.plot(x, yLine, color=RED)
        if yLineLabel is not None:
            ax2.set_ylabel(yLineLabel)
        if yLineLim is not None:
            ax2.set_ylim(yLineLim)

    # axis titles
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # set y limit
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        yMin = 0
        yMax = y.max() * 1.1
        ax.set_ylim([yMin, yMax])

    # grid lines
    if grid:
        ax.grid(True, linewidth=0.5)

    # x-axis tick marks
    ax.set_xticks(x)

    # format xticklabels
    if xticklabels is None:
        xticklabels = x
    ax.set_xticklabels(xticklabels)
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)

    # title and labels
    if title is not None:
        ax.set_title(title)

    autolabel(rects, ax)

    if tightLayout:
        fig.tight_layout()

    if save:
        if savepath[-1:] == "\\":
            savepath = savepath + "barplot.png"
        plt.savefig(savepath, format="png")
    if show:
        plt.show()

    if close:
        plt.close()

    return (fig, ax)


# ----------------------------------------
# plot a scatter plot between two quantities colored by a third
# ----------------------------------------
def colorscatter(
    x,
    y,
    z,
    fig=None,
    figsize: tuple = (14.4, 9),
    xname: str = "x",
    yname: str = "y",
    zname: str = "z",
    numBins: int = 10,
    title: str = None,
    save: bool = False,
    savepath: str = ".\\colorscatterplot.png",
    show: bool = False,
    close: bool = False,
):

    if fig is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
    else:
        ax = fig.get_axes()[0]

    # create bins of z, 1 bin of z = 1 color shade of z
    valStart = z.min()
    valEnd = z.max()
    vals = np.linspace(valStart, valEnd, numBins + 1)
    colors = cm.seismic(np.linspace(0, 1, len(vals)))

    # loop through all bins
    for cnt in range(0, len(vals) - 1):

        # create a a boolean indexer for z values in current bin
        idx = (z >= vals[cnt]) & (z < vals[cnt + 1])

        # get all x and y values associated with this bin of z values
        xbin = x.loc[idx]
        ybin = y.loc[idx]

        # create a scatter plot
        fig = scatter(
            x=xbin,
            y=ybin,
            fig=fig,
            xname=xname,
            yname=yname,
            markersize=6,
            color=tuple(colors[cnt]),
            title=title,
            save=save,
            savepath=savepath,
            show=show,
            close=False,
        )

    if title is not None:
        ax.set_title(title)
    if xname is not None:
        ax.xaxis.label.set_text(xname)
    if yname is not None:
        ax.yaxis.label.set_text(yname)

    if save:
        if savepath[-1:] == "\\":
            savepath = savepath + "colorscatter.png"
        plt.savefig(savepath, format="png")

    if show:
        plt.show()

    if close:
        plt.close()

    return fig


# ----------------------------------------
# stem and leaf plot
# ----------------------------------------
# Use Case 01: Take a DataFrame and plot all columns.
# Use Case 02: Specify maximum number of bins.
def stemleaf(
    df,
    numBins: int = 20,
    title: str = None,
    save: bool = False,
    savepath: str = ".\\stemleaf.txt",
    show: bool = False,
):

    retall = list()

    for col in dfutl.numericColumns(df):
        vals = df.loc[:, col]

        # determine the number of stems (bins)
        if vals.min() == vals.max():
            return None
        else:
            # keep multiplying by 10 until the target number of bins is exceeded
            valmin = vals.min()
            valmax = vals.max()
            exp10 = 0
            while math.ceil(valmax) - math.floor(valmin) < numBins:
                vals = vals * 10
                exp10 = exp10 + 1
                valmin = vals.min()
                valmax = vals.max()

            # infer the bin width from min and max values
            currNBins = math.ceil(vals.max()) - math.floor(vals.min())
            binw = math.ceil(currNBins / numBins)

            # infer value to start binning at
            valstart = math.floor(valmin)

            ## for debugging
            # print('exp10 = {0}'.format(exp10))
            # print('valmin = {0}'.format(valmin))
            # print('valmax = {0}'.format(valmax))
            # print('valstart = {0}'.format(valstart))
            # print('binw = {0}'.format(binw))
            # print('currNBins = {0}'.format(currNBins))

            ## for debugging print the value and its inferred bin
            # for val in vals:
            #    print('{0} - {1}'.format(val, str((((val - valstart) // binw) * binw) + valstart)))

            # determine the bin of each value
            bins = [int((((val - valstart) // binw) * binw)) + valstart for val in vals]

            # create a series object and group each value by its bin
            srs = pd.Series(vals.astype(int))
            grouped = srs.groupby(bins)

            # for debugging - print the values in each group
            # print(grouped.apply(lambda x: sorted([val for val in x])))
            aggregated = grouped.apply(lambda x: sorted([str(val)[-1] for val in x]))

            # determine the number of spaces for each stem
            ndigits = math.ceil(math.log10(aggregated.index.max())) - 1
            line = "{0: <" + str(ndigits + 1) + "}| "

            # print index except last character and and the last characters of list of values
            # associated with the index
            idxstr = [
                line.format(str(idx)[0:-1]) if len(str(idx)) > 1 else line.format("0")
                for idx in aggregated.index
            ]
            valstr = ["".join(vals) for vals in aggregated]
            N = len(aggregated.index)
            ret = [idxstr[idx] + valstr[idx] for idx in range(N)]

            # build a list of every line for this column
            ret = [line.format("x") + format("y", "<30") + "x.y"] + ret
            if title is not None:
                ret = [title + " " + col] + ret

            # add to overall list of strings
            retall = retall + ret
            retall = retall + ["\n\n"]

    # join each line in the list with a newline
    retall = "\n".join(retall)

    # save if applicable
    if save:
        if savepath is not None and savepath[-1] == "\\":
            savepath = savepath + "stemleaf.txt"
        with open(savepath, "w") as fl:
            fl.write(retall)

    # show if applicable
    if show:
        print(retall)

    return retall


# ----------------------------------------
# normal probability plot
# ----------------------------------------
def probplot(
    df,
    fig=None,
    figsize: tuple = (14.4, 9),
    ax=None,
    title: str = None,
    tightLayout: bool = True,
    save: bool = False,
    savepath: str = ".\\probplot.png",
    show: bool = False,
    close: bool = False,
):

    colNumeric = dfutl.numericColumns(df)
    numVar = len(colNumeric)
    df = df.loc[:, colNumeric]

    # if inputs are valid
    if numVar > 0:
        # determine the number of rows and columns of subplots
        # cap number of columns at 4 columns
        ncols = min(int(math.ceil(math.sqrt(numVar))), 3)
        numplots = 0
        nrows = 0
        while numplots < numVar:
            nrows = nrows + 1
            numplots = nrows * ncols

        # Modify figsize. Every 3 plots = 9 in in height.
        if numVar > 1:
            figsize = (14.4, int(nrows * 3))
        if fig is None:
            fig = plt.figure(figsize=figsize)

        if fig is not None and ax is not None:
            plotOne = True
        else:
            plotOne = False

        # loop through all variables and plot them on the corresponding axes
        for cntAx in range(0, numVar):

            # get the series for which the histogram is to be made
            x = df.iloc[:, cntAx].copy()
            x.sort_values(inplace=True)
            x.reset_index(drop=True, inplace=True)
            n = len(x.index)
            j = ((pd.Series(x.index) + 1) - 0.5) / n
            jmu = j.mean()
            jstd = j.std()
            z = stats.norm.ppf(j)

            # use values between the 25th and 75th percentile to plot a line
            idx = list(range(math.ceil(0.25 * n), math.floor(0.75 * n) + 1))
            xline = x[idx]
            if min(xline) == max(xline):
                # Ignore if the same value
                continue
            yline = z[idx]
            m, b, _, _, _ = stats.linregress(xline, yline)
            yreg = (m * x) + b

            # add an axes to the figure
            if ax is None:
                ax = fig.add_subplot(nrows, ncols, cntAx + 1)
            lines1 = ax.plot(x, z, marker="o", linewidth=0)
            lines2 = ax.plot(x, yreg, color=RED, linewidth=2)

            formatxticklabels(ax)
            ax.set_xlabel(colNumeric[cntAx])
            ax.set_ylabel("z", rotation=0)

            if plotOne:
                break

        if title is not None:
            ax.set_title(title)

        if tightLayout:
            fig.tight_layout()

        if save:
            if savepath is not None and savepath[-1:] == "\\":
                savepath = savepath + "probplot.png"
            plt.savefig(savepath, format="png")

        if show:
            plt.show()

        if close:
            plt.close()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../../data/bankMarketing.pkl"), "rb") as fl:
        df = pkl.load(fl)
    probplot(
        df,
        save=True,
        savepath="probplot.png",
        close=True,
    )
