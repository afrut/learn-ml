#exec(open('templates\\tuning_pipeline.py').read())
# template for nested tuning of a pipeline within cross-validation
# this is used to estimate the generalization performance of (an estimator with
# hyperparameter tuning) in every fold of cross-validation
import subprocess as sp
import numpy as np
import pickle as pk
import sklearn.model_selection as sms
import sklearn.neighbors as neighbors
import sklearn.preprocessing as pp
import sklearn.pipeline as pipeline

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

    # define a pipeline
    estimators = list()
    estimators.append(('standardize', pp.StandardScaler()))
    estimators.append(('knn', neighbors.KNeighborsRegressor()))
    ppl = pipeline.Pipeline(estimators)

    # tuning
    # parameters of pipelines can be set using ‘__’ separated parameter names
    param_grid = {'knn__n_neighbors': [x for x in range(1, 20)]}
    search = sms.GridSearchCV(estimator = ppl, param_grid = param_grid)

    # specify cross-validation
    k = 10
    cvsplitter = sms.KFold(n_splits = k, shuffle = True, random_state = 0)

    # execute cross-validation
    scoring = 'neg_mean_absolute_error'
    score = score = -1 * sms.cross_val_score(search, X, y, cv = cvsplitter, scoring = scoring)
    print('Mean absolute error: {0:.4f}'.format(score.mean()))