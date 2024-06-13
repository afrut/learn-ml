#exec(open('templates\\algs_compare_classification.py').read())
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
    with open('.\\data\\pima.pkl', 'rb') as fl:
        df = pk.load(fl)

    # specify the x and y matrices
    ycols = ['class']
    xcols = list(set(df.columns) - set(ycols))
    X = df.loc[:, xcols].values
    y = np.ravel(df.loc[:, ycols].values)

    # specify cross-validation
    k = 10                                                                  # number of folds
    cvsplitter = ms.KFold(n_splits = k, shuffle = True, random_state = 0)   # cross-validation splitter
    
    # specify all models
    models = list()
    models.append(('LR', sl.LogisticRegression(max_iter = 1000)))
    models.append(('LDA', da.LinearDiscriminantAnalysis()))
    models.append(('KNN', neighbors.KNeighborsClassifier()))
    models.append(('NB', nb.GaussianNB()))
    models.append(('CART', tree.DecisionTreeClassifier()))
    models.append(('SVM', svm.SVC()))

    # fit and compute scores
    scoring = 'accuracy'
    algs = list()
    scores = list()
    for entry in models:
        score = ms.cross_val_score(entry[1], X, y, cv = cvsplitter, scoring = scoring)
        scores.append(score)
        algs.append(entry[0])
        #print('{0} - {1:.4f} - {2:.4f}'.format(entry[0], np.mean(score),
        #np.std(score, ddof = 1)))

    # boxplot of results
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.boxplot(scores)
    ax.set_xticklabels(algs)
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracies of Different Classifiers')

    # table of results
    scores = np.array(scores)
    dfScores = pd.DataFrame(index = algs)
    dfScores['mean'] = np.mean(scores, axis = 1)
    dfScores['std'] = np.std(scores, ddof = 1, axis = 1)
    print('Mean and standard deviation of accuracies for different algorithms:')
    print(dfScores)

    plt.show()
    plt.close('all')