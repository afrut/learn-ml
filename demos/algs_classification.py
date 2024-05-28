#exec(open('templates\\algs_classification.py').read())
# testing different classification algorithms
import subprocess as sp
import sklearn.model_selection as ms
import sklearn.linear_model as sl
import sklearn.discriminant_analysis as da
import sklearn.neighbors as neighbors
import sklearn.naive_bayes as nb
import sklearn.tree as tree
import sklearn.svm as svm
import numpy as np
import pickle as pk

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
    k = 10                                                                  # number of folds
    cvsplitter = ms.KFold(n_splits = k, shuffle = True, random_state = 0)   # cross-validation splitter

    # logistic regression
    model = sl.LogisticRegression(max_iter = 1000)
    score = ms.cross_val_score(model, X, y, cv = cvsplitter)
    print('Logistic regression mean accuracy: {0:.4f}'.format(score.mean()))

    # linear discriminant analysis
    model = da.LinearDiscriminantAnalysis()
    score = ms.cross_val_score(model, X, y, cv = cvsplitter)
    print('Linear discriminant analysis mean accuracy: {0:.4f}'.format(score.mean()))
    
    # k-nearest neighbor classifier
    model = neighbors.KNeighborsClassifier()
    score = ms.cross_val_score(model, X, y, cv = cvsplitter)
    print('{0}-nearest neighbor classifier mean accuracy: {1:.4f}'.format(model.n_neighbors, score.mean()))
    
    # naive bayes classifier
    model = nb.GaussianNB()
    score = ms.cross_val_score(model, X, y, cv = cvsplitter)
    print('Gaussian Naive-Bayes classifier mean accuracy: {0:.4f}'.format(score.mean()))
    
    # Classification and Regression Tree (CART) classifier
    model = tree.DecisionTreeClassifier()
    score = ms.cross_val_score(model, X, y, cv = cvsplitter)
    print('Decision tree classifier mean accuracy: {0:.4f}'.format(score.mean()))
    
    # support vector machines (SVM)
    model = svm.SVC()
    score = ms.cross_val_score(model, X, y, cv = cvsplitter)
    print('SVM classifier mean accuracy: {0:.4f}'.format(score.mean()))