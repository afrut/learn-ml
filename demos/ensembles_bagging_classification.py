#exec(open('templates\\ensembles_bagging_classification.py').read())
# testing different classification algorithms
import subprocess as sp
import numpy as np
import pickle as pk
import sklearn.model_selection as ms
import sklearn.ensemble as ensemble
import sklearn.tree as tree

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

    # create a bagged classifier
    model = ensemble.BaggingClassifier(base_estimator = tree.DecisionTreeClassifier()       # the base algorithm
        ,n_estimators = 100                                                                 # number of base estimators to construct when bagging
        ,random_state = 0)
    scores = ms.cross_val_score(model, X, y, cv = cvsplitter)
    print('Bagged decision tree accuracy mean: {0:.4f}'.format(scores.mean()))
    print('Bagged decision tree accuracy std: {0:.4f}'.format(scores.std()))

    # random forest classifier
    model = ensemble.RandomForestClassifier(n_estimators = 100  # number of trees to construct
        ,max_features = 3)                                      # number of features to consider when looking for the best split   
    scores = ms.cross_val_score(model, X, y, cv = cvsplitter)
    print('Random forest accuracy mean: {0:.4f}'.format(scores.mean()))
    print('Random forest accuracy std: {0:.4f}'.format(scores.std()))

    # extra trees classifier
    model = ensemble.ExtraTreesClassifier(n_estimators = 100  # number of trees to construct
        ,max_features = 7)                                      # number of features to consider when looking for the best split   
    scores = ms.cross_val_score(model, X, y, cv = cvsplitter)
    print('Extra tree classifier accuracy mean: {0:.4f}'.format(scores.mean()))
    print('Extra tree classifier accuracy std: {0:.4f}'.format(scores.std()))