#exec(open('templates\\tuning.py').read())
# template for algorithm performance with tuning
import subprocess as sp
import numpy as np
import pickle as pk
import sklearn.model_selection as sm
import sklearn.linear_model as sl
import scipy.stats as stats

if __name__ == '__main__':
    sp.call('cls', shell = True)

    # load some data
    with open('.\\data\\pima.pkl', 'rb') as fl:
        df = pk.load(fl)

    # specify the x and y matrices
    ycols = ['class']
    xcols = list(set(df.columns) - set(ycols))
    X = df.loc[:, xcols].values
    y = np.ravel(df.loc[:, ycols].values)

    # specify cross-validation
    k = 10
    cvsplitter = sm.KFold(n_splits = k, shuffle = True, random_state = 0)

    # array of hyperparameter values to test
    alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
    param_grid = {'alpha': alphas}

    # create the model and execute grid search
    model = sl.Ridge()
    search = sm.GridSearchCV(estimator = model, param_grid = param_grid)
    search.fit(X, y)
    print('Grid search score of best hyperparameter value: {0:.4f}'.format(search.best_score_))
    print('Grid search best hyperparameter value: {0:.4f}'.format(search.best_estimator_.alpha))
    print('Grid search best hyperparameter values:')
    for tpl in search.best_params_.items():
        print('    {0:<10}: {1:.6f}'.format(tpl[0], tpl[1]))
    print('')
    
    # use randomized search by selecting hyperparameter values randomly from a
    # uniform distribution 100 times
    param_grid = {'alpha': stats.uniform()}
    search = sm.RandomizedSearchCV(estimator = model
        , param_distributions = param_grid, n_iter = 100)
    search.fit(X, y)
    print('Randomized search score of best hyperparameter value: {0:.4f}'.format(search.best_score_))
    print('Randomized search best hyperparameter value: {0:.4f}'.format(search.best_estimator_.alpha))
    print('Randomized search best hyperparameter values:')
    for tpl in search.best_params_.items():
        print('    {0:<10}: {1:.6f}'.format(tpl[0], tpl[1]))
    print('')