#exec(open('templates\\preproc.py').read())
import subprocess as sp
import pandas as pd
import sklearn.preprocessing as pp
import numpy as np

if __name__ == '__main__':
    sp.call('cls', shell = True)

    # load some data
    with open('.\\data\\iris\\iris.data', 'rt') as fl:
        df = pd.read_csv(fl
            ,names = ['sepal_length','sepal_width','petal_length','petal_width','class']
            ,header = None
            ,index_col = False)

    # get numeric columns
    dfnumeric = df[df.select_dtypes([np.number]).columns]

    # get arra of numeric values
    X = dfnumeric.values

    # define a scaler
    scaler = pp.MinMaxScaler(feature_range = (0, 1)).fit(X)

    # transform data
    rescaledX = scaler.transform(X)

    # display
    np.set_printoptions(precision = 3)
    print('MinMaxScaler transformed data:\n{0}\n'.format(rescaledX[0:5,:]))

    # ----------------------------------------
    # multiple transformations
    # ----------------------------------------
    scalers = dict()

    # transform so that smallest data is 0, and largest is 1
    scaler = pp.MinMaxScaler(feature_range = (0, 1)).fit(X)
    scalers['MinMaxScaler'] = (scaler, scaler.transform(X))

    # transform so that data is standard normal Gaussian distribution; ie, mean
    # of 0 and standard deviation of 1
    scaler = pp.StandardScaler().fit(X)
    scalers['StandardScaler'] = (scaler, scaler.transform(X))

    # transform so that the length of each observation (row) has a length of 1
    # (unit vector)
    scaler = pp.Normalizer().fit(X)
    scalers['Normalizer'] = (scaler, scaler.transform(X))

    # transform so that all values above a threshold are 1 and all values below
    # a threshold are 0
    scaler = pp.Binarizer(threshold = 2).fit(X)
    scalers['Binarizer'] = (scaler, scaler.transform(X))

    # display results of transformations
    for entry in scalers.items():
        print('{0} transformed data:\n{1}\n'.format(entry[0], entry[1][1][0:5,:]))