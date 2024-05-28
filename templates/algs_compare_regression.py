#exec(open('templates\\algs_compare_regression.py').read())
# testing different classification algorithms
import subprocess as sp
import pandas as pd
import sklearn.model_selection as ms
import sklearn.linear_model as sl
import sklearn.metrics as sm
import sklearn.discriminant_analysis as da
import sklearn.neighbors as neighbors
import sklearn.naive_bayes as nb
import sklearn.tree as tree
import sklearn.svm as svm
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sp.call('cls', shell = True)
    plt.close('all')

    # load some data
    with open('.\\data\\bostonHousing.pkl', 'rb') as fl:
        df = pk.load(fl)

    # specify the x and y matrices
    ycols = ['PRICE']
    xcols = list(set(df.columns) - set(ycols))
    X = df.loc[:, xcols].values
    y = np.ravel(df.loc[:, ycols].values)

    # specify cross-validation
    k = 10                                                                  # number of folds
    cvsplitter = ms.KFold(n_splits = k, shuffle = True, random_state = 0)   # cross-validation splitter
    
    # specify all models
    models = list()
    models.append(('LR', sl.LinearRegression()))
    models.append(('RIDGE', sl.Ridge()))
    models.append(('LASSO', sl.Lasso()))
    models.append(('EN', sl.ElasticNet()))
    models.append(('KNN', neighbors.KNeighborsRegressor()))
    models.append(('CART', tree.DecisionTreeRegressor()))
    models.append(('SVM', svm.SVR()))

    # fit and compute scores
    scoring = 'neg_mean_squared_error'
    algs = list()
    scores = list()
    for entry in models:
        score = -1 * ms.cross_val_score(entry[1], X, y, cv = cvsplitter, scoring = scoring)
        scores.append(score)
        algs.append(entry[0])
        #print('{0} - {1:.4f} - {2:.4f}'.format(entry[0], np.mean(score), np.std(score, ddof = 1)))

    # boxplot of results
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.boxplot(scores)
    ax.set_xticklabels(algs)
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('Mean Squared Error of Different Classifiers')

    # table of results
    scores = np.array(scores)
    dfScores = pd.DataFrame(index = algs)
    dfScores['mean'] = np.mean(scores, axis = 1)
    dfScores['std'] = np.std(scores, ddof = 1, axis = 1)
    print('Mean and standard deviation of MSE for different algorithms:')
    print(dfScores)

    plt.show()
    plt.close('all')