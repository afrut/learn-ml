#exec(open('templates\\preproc_encoding.py').read())
import subprocess as sp
import pickle as pk
import sklearn.preprocessing as pp
import numpy as np

if __name__ == '__main__':
    sp.call('cls', shell = True)

    # load some data
    with open('.\\data\\adult.pkl', 'rb') as fl:
        df = pk.load(fl)

    # get education data
    X = df['education'].values[:, np.newaxis]

    # create ordinal encoder without specifying order of labels
    enc = pp.OrdinalEncoder()
    enc.fit(X)
    Xtrans = np.ravel(enc.transform(X))
    print('OrdinalEncoder categories used:\n{0}\n'.format(enc.categories_[0]))
    print('OrdinalEncoder transformed data:\n{0}\n'.format(Xtrans[0:10]))

    # create the ordinal encoder specifying the order of the labels
    # specified increasing order
    educat = [' Preschool',' 1st-4th',' 5th-6th',' 7th-8th',' 9th',' 10th',' 11th',' 12th',' HS-grad',' Prof-school',' Assoc-acdm',' Assoc-voc',' Some-college',' Bachelors',' Masters',' Doctorate']
    enc = pp.OrdinalEncoder(categories = [educat])
    enc.fit(X)
    Xtrans = np.ravel(enc.transform(X))
    print('OrdinalEncoder categories used with specified order:\n{0}\n'.format(enc.categories_[0]))
    print('OrdinalEncoder transformed data with specified order:\n{0}\n'.format(Xtrans[0:10]))
    
    # one-hot encoding
    X = df['native-country'].values[:, np.newaxis]
    enc = pp.OneHotEncoder(sparse = False)
    enc.fit(X)
    Xtrans = enc.transform(X)
    print('Number of categories: {0}'.format(len(enc.categories_[0])))
    print('Shape of transformed data: {0}'.format(Xtrans.shape))
    print('OneHotEncoder categories used:\n{0}\n'.format(enc.categories_[0]))
    print('OneHotEncoder transformed data:\n{0}\n'.format(Xtrans[0:10, :]))

    # dummy encoding using OneHotEncoder by dropping (alphabetically) first category
    # handle_unknown = 'ignore' means if a new category is encountered in the test set, ignore the value and continue.
    X = df['native-country'].values[:, np.newaxis]
    enc = pp.OneHotEncoder(sparse = False, drop = 'first', handle_unknown = 'ignore')
    enc.fit(X)
    Xtrans = enc.transform(X)
    print('Number of categories: {0}'.format(len(enc.categories_[0])))
    print('Shape of transformed data: {0}'.format(Xtrans.shape))
    print('Dummy encoding categories used:\n{0}\n'.format(enc.categories_[0]))
    print('Dummy encoding transformed data:\n{0}\n'.format(Xtrans[0:10, :]))

    # check that the excluded variable is the first variable
    excluded = enc.categories_[0][0]
    idx = Xtrans.sum(axis = 1) == 0     # find all rows that don't fit in another category
    print('Labels of the dropped category:\n{0}\n'.format(X[idx][0:10]))
    assert np.all(X[idx] == excluded)