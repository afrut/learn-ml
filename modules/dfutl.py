import math
import os
import pickle as pkl
import warnings

import numpy as np
import pandas as pd
from IPython import embed


# ------------------------------------------------------------
# check for null values in a DataFrame
# ------------------------------------------------------------
def checkNull(
    df,
    lsCols: list = None,
    droprows: bool = False,
    dropcols: bool = False,
    printCols: list = None,
    printNullRows: bool = False,
    printNullColumns: bool = False,
    indent: str = "",
):
    if lsCols is not None:
        dfTemp = df.loc[:, lsCols]
    else:
        dfTemp = df

    mask = dfTemp.isna().any(axis="columns")
    rowsWithNulls = dfTemp.loc[mask, :]
    mask = dfTemp.isna().any(axis="rows")
    colsWithNulls = mask.index[mask]

    nrow = len(rowsWithNulls)
    ncol = len(colsWithNulls)
    print("{}Number of rows with missing data: {}".format(indent, nrow))
    print("{}Number of columns with missing data: {}".format(indent, ncol))

    if printNullColumns:
        print("{}Columns with nulls:".format(indent))
        for col in colsWithNulls:
            print("{}{}{}".format(indent, indent, col))
    if printNullRows:
        print("{}Rows with nulls:".format(indent))
        for row in rowsWithNulls.index:
            print("{}{}{}".format(indent, indent, row))

    if droprows and nrow > 0:
        print("{}Dropped the following rows:".format(indent))
        if printCols is None or len(printCols) == 0:
            for idx in rowsWithNulls.index:
                print("{}{}{}".format(indent, indent, idx))
        else:
            dfPrint = rowsWithNulls.loc[:, printCols]
            for tpl in dfPrint.itertuples():
                msg = "{}".format(tpl[0])
                for cnt in range(1, len(tpl)):
                    msg = msg + " - {}".format(tpl[cnt])
                print("{}{}{}".format(indent, indent, msg))
        df.drop(labels=rowsWithNulls.index, axis="rows", inplace=True)

    if dropcols and ncol > 0:
        print("{}Dropped the following columns:".format(indent))
        for idx in colsWithNulls:
            print("{}{}{}".format(indent, indent, idx))
        df.dropna(labels=colsWithNulls, axis="columns", inplace=True)


# ------------------------------------------------------------
# apply some filters to the DataFrame
# ------------------------------------------------------------
def applyFilters(df, lsFilters):
    # instantiate an array of all True
    idx = np.array((np.squeeze(np.ones((len(df.index), 1), dtype=bool), 1)))

    for fil in lsFilters:
        idx = np.logical_and(idx, np.array(fil))
    return df.loc[idx, :]


# ------------------------------------------------------------
# function to group data into bins
# ------------------------------------------------------------
def categorize(
    df, col, cat, binWidth, scale, numCatRet=None, retBinCol=True, encode=True
):
    # df - pandas DataFrame containing data
    # col - column label on which to split the data set
    # cat - a string that is used as a column label to indicate the bin that a sample belongs to
    # binWidth - the width of each bin in the unit of the quantity
    # scale - used for how many decimal places to truncate when determining the starting value
    # numCatRet - the number of categories to return
    # retBinCol - add the the categories to the original DataFrame?
    # decode - use integers instead of intervals to signify the categories instead
    #
    # return 1 - the original DataFrame with a column indicating the bin the sample belongs to
    # return 2 - a DataFrame containing the number of samples for each bin
    # return 3 - a DataFrame containing the number of samples for each bin

    # create bins for col with name cat
    binStartVal = math.trunc(df.loc[:, col].min() * scale) / scale
    binStopVal = df.loc[:, col].max() + binWidth
    bins = np.arange(binStartVal, binStopVal, binWidth)
    grouped = None

    if len(bins) > 1:
        # bin the thicknesses
        binned = pd.cut(x=df.loc[:, col], bins=bins, right=False)

        # if the bins are to be encoded
        dctCode = dict()
        if encode:
            # get unique values of the bins
            unqBins = sorted(list(set(binned)))

            # create dictionaries for encoding and decoding
            dctCode["toCode"] = {unqBins[cnt]: cnt for cnt in range(0, len(unqBins))}
            dctCode["fromCode"] = {cnt: unqBins[cnt] for cnt in range(0, len(unqBins))}

            # map the values of the binned series according to dictionaries
            binned = binned.astype(pd.Interval).map(dctCode["toCode"])

        # add to the original DataFrame
        df.loc[:, cat] = binned

        # count how many data points in each bin and sort
        grouped = df.loc[:, [col, cat]].groupby([cat]).count()
        grouped.sort_values(by=[col], ascending=False, inplace=True)

        # extract data for the first thinkess bin, which has the most records
        if numCatRet is not None:
            idxAll = np.zeros(
                [
                    len(df.index),
                ],
                dtype=bool,
            )
            for cnt in range(0, numCatRet):
                if not encode:
                    idx = np.array(df.loc[:, col] >= grouped.index[cnt].left)
                    idx = np.logical_and(
                        idx, np.array(df.loc[:, col] < grouped.index[cnt].right)
                    )
                    idxAll = (idxAll) | (idx)
                else:
                    idx = np.array(
                        df.loc[:, col] >= dctCode["fromCode"][grouped.index[cnt]].left
                    )
                    idx = np.logical_and(
                        idx,
                        np.array(
                            df.loc[:, col]
                            < dctCode["fromCode"][grouped.index[cnt]].right
                        ),
                    )
                    idxAll = (idxAll) | (idx)
            df = df.loc[idxAll, :].copy()
        else:
            df = df.copy()

        # check if the column indicating the bin should be returned
        if not retBinCol:
            df.drop([cat], inplace=True, axis=1)

    return (df, grouped, dctCode)


# returns the column names that are of numeric types
def numericColumns(df):
    isNumeric = np.vectorize(lambda x: np.issubdtype(x, np.number))
    colNumeric = isNumeric(df.dtypes)
    return list(df.columns[colNumeric])


def one_hot_encode(df: pd.DataFrame, columns: list[str]):
    one_hot_encoded = pd.get_dummies(df[columns], prefix=columns).astype(int)
    df = df.drop(columns, axis=1)
    return pd.concat([df, one_hot_encoded], axis=1)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "../data"))

    # Test one-hot encoding
    with open(os.path.join(data_dir, "adult.pkl"), "rb") as fl:
        df = pkl.load(fl)
        columns_to_ohe = ["workclass", "education", "marital-status", "sex"]
        df = one_hot_encode(df, columns=columns_to_ohe)
        embed()
