#exec(open('templates\\ensembles_boosting_classification.py').read())
# testing different classification algorithms
import subprocess as sp
import numpy as np
import pickle as pk
import sklearn.model_selection as sm
import sklearn.ensemble as ensemble
import sklearn.tree

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
    cvsplitter = ms.KFold(n_splits = k, shuffle = True, random_state = 0)

    # adaboost classifier with 30 decision trees
    model = ensemble.AdaBoostClassifier(n_estimators = 30, random_state = 0)
    scores = sm.cross_val_score(model, X, y, cv = cvsplitter)
    print('AdaBoostClassifer accuracy mean: {0:.4f}'.format(scores.mean()))
    print('AdaBoostClassifer accuracy std: {0:.4f}'.format(scores.std()))

    # stochastic gradient boosting aka gradient boosting machines
    # create a gradient boosting machine with 30 trees
    model = ensemble.GradientBoostingClassifier(n_estimators = 30, random_state = 0)
    scores = sm.cross_val_score(model, X, y, cv = cvsplitter)
    print('GradientBoostingClassifier accuracy mean: {0:.4f}'.format(scores.mean()))
    print('GradientBoostingClassifier accuracy std: {0:.4f}'.format(scores.std()))