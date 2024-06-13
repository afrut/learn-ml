#exec(open('templates\\metrics_regression.py').read())
# assessing a regression model's efficacy
import subprocess as sp
import numpy as np
import pickle as pk
import sklearn.model_selection as ms
import sklearn.linear_model as sl

if __name__ == '__main__':
    sp.call('cls', shell = True)

    # load some data
    with open('.\\data\\bostonHousing.pkl', 'rb') as fl:
        df = pk.load(fl)

    # specify the x and y matrices
    ycols = ['PRICE']
    xcols = list(set(df.columns) - set(ycols))
    X = df.loc[:, xcols].values
    y = np.ravel(df.loc[:, ycols].values)

    k = 10                                                                  # number of folds
    cvsplitter = ms.KFold(n_splits = k, shuffle = True, random_state = 0)   # cross-validation splitter
    model = sl.LinearRegression()                                           # select a model to use
    scoring = 'neg_mean_absolute_error'                                     # metric to use
    score = ms.cross_val_score(model, X, y, cv = cvsplitter, scoring = scoring)
    print('{0} mean: {1:.4f}'.format(scoring, score.mean()))
    print('{0} std: {1:.4f}'.format(scoring, score.std()))

    scoring = 'neg_mean_squared_error'
    score = ms.cross_val_score(model, X, y, cv = cvsplitter, scoring = scoring)
    print('{0} mean: {1:.4f}'.format(scoring, score.mean()))
    print('{0} std: {1:.4f}'.format(scoring, score.std()))

    scoring = 'r2'
    score = ms.cross_val_score(model, X, y, cv = cvsplitter, scoring = scoring)
    print('{0} mean: {1:.4f}'.format(scoring, score.mean()))
    print('{0} std: {1:.4f}'.format(scoring, score.std()))