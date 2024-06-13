#exec(open('.\\templates\\data_manipulation.py').read())
# TODO: pd.DataFrame.pivot
# TODO: pd.DataFrame.melt
# TODO: np.count_nonzero
# TODO: np.repeat
import subprocess as sp
import pandas as pd
import numpy as np
import random
import math
import itertools

def func(a: np.array):
    return (2 * a[0]) + math.sin(a[1]) + math.sqrt(a[2]) + a[3]

def threshold(a: float):
    "Return 0 if value is greater than 5, else return value"
    if a > 5:
        return 0
    else:
        return a

def load():
    with open('.\\data\\iris\\iris.data', 'rt') as fl:
        df = pd.read_csv(fl
            ,names = ['sepal_length','sepal_width','petal_length','petal_width','class']
            ,header = None
            ,index_col = False)
        return df

if __name__ == '__main__':
    sp.call('cls', shell = True)

    # load some data
    df = load()

    # using zip to pair elements of 2 different tuples
    # zip takes a variable number of sequences as arguments
    # it is possible to pass a single sequence of sequences ls by using zip(*ls)
    ls = list(zip((1,2,3), (4,5,6)))        # ls = [(1, 4), (2, 5), (3, 6)]
    ls = list(zip(*ls))                     # ls = [(1, 2, 3), (4, 5, 6)]

    # get all numeric columns
    numeric = df.select_dtypes([np.number]).columns
    nonNumeric = df.select_dtypes([object]).columns
    print('Numeric columns:')
    for val in numeric:
        print('  ' + val)
    print('Non-numeric columns:')
    for val in nonNumeric:
        print('  ' + val)
    print('')

    # generate a random number in index to set to nan
    row = random.randint(df.index[0], df.index[-1])
    col = random.randint(0, len(df.columns) - 1)
    df.iloc[row, col] = np.nan

    # check if any value in the dataframe is na
    print('df.isna():\n{0}\n'.format(df.isna()))

    # for every column, check if there is at least one value that is na
    print('df.isna().any():\n{0}\n'.format(df.isna().any()))

    # for every row, check if there is at least one value that is na
    print('df.isna().any(axis = 1):\n{0}\n'.format(df.isna().any(axis = 1)))

    # boolean indexing
    srs = df.loc[df.loc[:, 'sepal_length'] < 5, :]
    print('rows with sepal_length < 5:\n{0}\n'.format(srs))

    # convert a pandas series to a numpy array
    sepal = df.loc[:, 'sepal_length'].copy().to_numpy()
    print('numpy array of sepal lengths:\n{0}\n'.format(sepal))

    # convert numpy array to Python list
    lssepal = sepal.tolist()
    print('list of sepal lengths:\n{0}\n'.format(lssepal))

    # find row and column with nan value
    row = df.isna().any(axis = 1)
    row = row[row].index[0]
    col = df.isna().any()
    col = col[col].index[0]
    print('row {0} and column {1} has value: {2}\n'.format(row, col, df.loc[row, col]))
    df.loc[row, col] = 0

    # find first index of an element in a Python list
    print('index of sepal_length == 5.4: {0}\n'.format(lssepal.index(5.4)))

    # indexes of a numpy array that fit a criteria
    print('indexes of values of sepal_length < 5:\n{0}\n'.format(np.where(sepal < 5)[0]))

    # apply a function to a series or dataframe
    print('columns without null values:\n{0}\n'.format(df.isna().any().apply(np.logical_not)))
    
    # define and use a vectorized user-defined function and return results as float
    f = np.vectorize(threshold, otypes = [float])
    print('using vectorize function to set all values of sepal_length > 5 to 0:\n{0}\n'.format(f(sepal)))
    
    # update the values of a DataFrame based on a second one
    # create a second dataframe with only odd rows
    df2 = pd.DataFrame(f(sepal), columns = ['sepal_length'])
    df2 = df2.loc[(df2.index % 2) == 1, :]
    df.update(df2)
    print('updated dataframe using another dataframe:\n{0}\n'.format(df))

    # merge one dataframe with another
    srs = df.loc[:, 'petal_length'].copy()
    srs[srs.index % 2 == 0] = 0
    df2 = pd.DataFrame(srs, columns = ['petal_length'])
    df = df.merge(df2
        ,how = 'left'
        ,left_index = True
        ,right_index = True
        #,left_on = columnname
        #,right_on = columnname
        ,suffixes = ['_old', '_new']
        ,copy = False)
    print('merged dataframe:\n{0}\n'.format(df))

    # replace all rows in the DataFrame where sepal_length <= 5 with nan
    df2 = load()
    df2 = df2.where(df2.loc[:,'sepal_length'] > 5)
    print(df2)

    # replace all rows in the DataFrame where sepal_length > 5 with nan
    df2 = load()
    df2 = df2.mask(df2.loc[:,'sepal_length'] > 5)
    print(df2)

    # relational operators in pandas.DataFrame
    df2 = load()
    df2 = df2.loc[:, df2.select_dtypes([np.number]).columns]
    df2.lt(5)   # < 5
    df2.eq(5)   # == 5
    df2.gt(5)   # > 5
    df2.le(5)   # <= 5
    df2.ge(5)   # >= 5

    # group by/aggregate the data on a column(s)
    grouped = df.groupby(['class'])
    print('number of elements in each class:\n{0}\n'.format(grouped.size()))

    # create an array by specifying a start, stop and interval
    arr = np.arange(0, 5, 0.5)
    print('arr:\n{0}\n'.format(arr))

    # create an array by specifying a start, stop, and the number of
    # equally-spaced values
    arr = np.linspace(0, 4, 17)
    print('arr:\n{0}\n'.format(arr))

    # adding an axis with np.newaxis
    print('arr.shape: {0}'.format(arr.shape))
    arr = arr[:, np.newaxis]
    print('arr[:, np.newaxis].shape : {0}'.format(arr.shape))
    print('')

    # removing an axis with np.ravel
    print('arr.shape: {0}'.format(arr.shape))
    print('np.ravel(arr).shape: {0}'.format(np.ravel(arr).shape))
    print('')

    # removing an axis with np.squeeze
    print('arr.shape: {0}'.format(arr.shape))
    print('np.squeeze(arr, axis = 1).shape: {0}'.format(np.squeeze(arr, axis = 1).shape))
    print('')

    # stacking arrays - joining arrays on a new axis
    arr1 = np.ravel(arr)
    arr2 = np.linspace(1, 5, 17)
    print('arr1.shape: {0}'.format(arr1.shape))
    print('arr2.shape: {0}'.format(arr2.shape))
    print('np.stack([arr1, arr2], axis = 0).shape: {0}'.format(np.stack([arr1, arr2], axis = 0).shape))
    print('np.stack([arr1, arr2], axis = 1).shape: {0}'.format(np.stack([arr1, arr2], axis = 1).shape))
    print('np.vstack([arr1, arr2]).shape: {0}'.format(np.vstack([arr1, arr2]).shape))
    print('np.hstack([arr1, arr2]).shape: {0}'.format(np.hstack([arr1, arr2]).shape))
    print('np.hstack([arr1[:, np.newaxis], arr2[:, np.newaxis]]).shape: {0}'\
        .format(np.hstack([arr1[:, np.newaxis], arr2[:, np.newaxis]]).shape))
    print('')

    # concatenating arrays - joining arrays on an existing axis
    arr1 = arr1[:, np.newaxis]
    arr2 = arr2[:, np.newaxis]
    print('arr1.shape: {0}'.format(arr1.shape))
    print('arr2.shape: {0}'.format(arr2.shape))
    print('np.concatenate([arr1,arr2], axis = 0).shape: {0}'.format(np.concatenate([arr1,arr2], axis = 0).shape))
    print('np.concatenate([arr1,arr2], axis = 1).shape: {0}'.format(np.concatenate([arr1,arr2], axis = 1).shape))
    print('')

    # generating every combination of elements between multiple sequences
    x1 = [1,2,3]
    x2 = [4,5]
    x3 = [6,7,8,9,10]
    print('x1: {0}'.format(x1))
    print('x2: {0}'.format(x2))
    print('x3: {0}'.format(x3))
    sequences = [x1,x2,x3]
    combinations = np.array(np.meshgrid(*sequences)).T.reshape(-1,len(sequences))
    print('All combinations:\n{0}\n'.format(combinations))

    # sort an array by its first column, then second column, third column, ...
    reversecol = combinations[:, [x for x in range(combinations.shape[1] - 1, -1, -1)]]     # reverse columns of array
    transpose = reversecol.T                                                                # get transpose of array
    idx = np.lexsort(transpose)                                                             # sort
    print('Sorted array:\n{0}\n'.format(combinations[idx]))

    # replace all values < 3 with 0
    arr1 = df2.values
    arr1 = np.where(arr1 < 3, 0, arr1)
    print('Any value < 3 replaced by 0:\n{0}\n'.format(arr1))

    # apply a custom function along an axis of an array
    # axis = 0 means loop through every columna and apply the function on rows
    # axis = 1 means loop through every row and apply the function on columns
    arr = df2.values
    arr = np.apply_along_axis(func1d = func, axis = 1, arr = arr1)
    print('Function applied on every row:\n{0}\n'.format(arr))

    # reshape an array
    arr = np.arange(0,12,1)
    print('Original array:\n{0}'.format(arr))
    print('Reshape into 4 rows and 3 columns:\n{0}'.format(arr.reshape(4, 3)))
    print('Reshape into 6 rows and 2 columns:\n{0}'.format(arr.reshape(6, 2)))
    print('Reshape into 2 rows and balance of columns:\n{0}'.format(arr.reshape(2, -1)))
    print('')
    
    # count unique elements in an array
    df2 = load()
    arr = df2.loc[:, 'class'].values.astype('str')
    valsUnq, counts = np.unique(arr, return_counts = True)
    print('Counts of each class:')
    for cnt in range(len(valsUnq)):
        print('    {0}: {1}'.format(valsUnq[cnt], counts[cnt]))