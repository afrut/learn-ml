#exec(open('templates\\ensembles_stacking_classification.py').read())
# template for a stacking classifier
import subprocess as sp
import numpy as np
import pickle as pk
import sklearn.model_selection as ms
import sklearn.ensemble as ensemble
import sklearn.tree as tree
import sklearn.neighbors as neighbors
import sklearn.linear_model as slm

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

    # create list of estimators
    estimators = list()
    estimators.append(('knn', neighbors.KNeighborsClassifier()))
    estimators.append(('logistic', slm.LogisticRegression(max_iter = 1000)))

    # create a stacking classifier
    model = ensemble.StackingClassifier(estimators = estimators)
    scores = ms.cross_val_score(model, X, y, cv = cvsplitter)
    print('StackingClassifier accuracy mean: {0:.4f}'.format(scores.mean()))