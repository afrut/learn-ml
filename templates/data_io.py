#exec(open('.\\templates\\data_io.py').read())
import subprocess as sp
import pickle as pk
import pandas as pd
import os

if __name__ == '__main__':
    sp.call('cls', shell = True)

    # load a DataFrame from csv file
    # open file in 'r' (read-only) and as 't' (text)
    with open('.\\data\\iris\\iris.data', 'rt') as fl:
        df = pd.read_csv(fl                             # file object
            ,names = ['sepal_length','sepal_width'
                ,'petal_length','petal_width','class']  # column names
            ,header = None                              # first row in the file is not a header of column names
            ,index_col = False)                         # don't use a column as index

    # save the data to a pickle file
    # open file in 'w' (write) and as 'b' (binary)
    with open('.\\sample.pkl', 'wb') as fl:
        pk.dump(df, fl)

    # check if a file exists
    filename = '.\\sample.pkl'
    print('Does \'{0}\' exist? {1}'.format(filename, os.path.exists(filename)))

    # load data from a pickle file
    with open('.\\sample.pkl','rb') as fl:
        df = pk.load(fl)
    print('Shape of DataFrame: {0}'.format(df.shape))

    # delete a file
    os.remove('.\\sample.pkl')