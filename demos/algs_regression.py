#exec(open('templates\\algs_regression.py').read())
# testing different regression algorithms
import subprocess as sp
import sklearn.model_selection as ms
import sklearn.linear_model as sl
import sklearn.neighbors as neighbors
import sklearn.tree as tree
import sklearn.svm as svm
import numpy as np
import pickle as pk

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

    # specify cross-validation
    k = 10                                                                  # number of folds
    cvsplitter = ms.KFold(n_splits = k, shuffle = True, random_state = 0)   # cross-validation splitter

    # linear regression
    model = sl.LinearRegression()
    score = ms.cross_val_score(model, X, y, cv = cvsplitter, scoring = 'neg_mean_squared_error')
    print('Linear regression neg_mean_squared_error mean: {0:.4f}'.format(score.mean()))

    # ridge regression, penalty: sum squared values of coefficients aka L2-norm
    model = sl.Ridge()
    score = ms.cross_val_score(model, X, y, cv = cvsplitter, scoring = 'neg_mean_squared_error')
    print('Ridge regression neg_mean_squared_error mean: {0:.4f}'.format(score.mean()))
    
    # lasso regression, penalty: sum of absolute value of coefficient values aka
    # L1-norm
    model = sl.Lasso()
    score = ms.cross_val_score(model, X, y, cv = cvsplitter, scoring = 'neg_mean_squared_error')
    print('Lasso regression neg_mean_squared_error mean: {0:.4f}'.format(score.mean()))
    
    # ElasticNet regression, penalty: combination of ridge and lasso; penalize
    # both magnitude and number of coefficients
    model = sl.ElasticNet()
    score = ms.cross_val_score(model, X, y, cv = cvsplitter, scoring = 'neg_mean_squared_error')
    print('ElasticNet regression neg_mean_squared_error mean: {0:.4f}'.format(score.mean()))
    
    # k-nearest neighbor regression
    model = neighbors.KNeighborsRegressor()
    score = ms.cross_val_score(model, X, y, cv = cvsplitter, scoring = 'neg_mean_squared_error')
    print('{0}-nearest neighbor regressor neg_mean_squared_error mean: {1:.4f}'.format(model.n_neighbors, score.mean()))
    
    # decision tree regression
    model = tree.DecisionTreeRegressor()
    score = ms.cross_val_score(model, X, y, cv = cvsplitter, scoring = 'neg_mean_squared_error')
    print('Decision tree regression neg_mean_squared_error mean: {0:.4f}'.format(score.mean()))
    
    # support vector machine regression
    model = svm.SVR()
    score = ms.cross_val_score(model, X, y, cv = cvsplitter, scoring = 'neg_mean_squared_error')
    print('SVM regression neg_mean_squared_error mean: {0:.4f}'.format(score.mean()))