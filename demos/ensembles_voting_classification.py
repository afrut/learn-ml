#exec(open('templates\\ensembles_voting_classification.py').read())
import subprocess as sp
import numpy as np
import pickle as pk
import sklearn.model_selection as sm
import sklearn.linear_model as sl
import sklearn.tree as tree
import sklearn.svm as svm
import sklearn.ensemble as ensemble

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

    # create a multiple models to participate in the voting
    estimators = list()
    estimators.append(('logistic', sl.LogisticRegression(max_iter = 1000)))
    estimators.append(('cart', tree.DecisionTreeClassifier()))
    estimators.append(('svm', svm.SVC()))

    # create voting ensemble model
    model = ensemble.VotingClassifier(estimators)
    results = sm.cross_val_score(model, X, y, cv = cvsplitter)
    print('Voting ensemble accuracy mean: {0:.4f}'.format(results.mean()))
    print('Voting ensemble accuracy std: {0:.4f}'.format(results.std()))