#exec(open('templates\\modelsel_test_train.py').read())
# model evaluation by splitting data into training and test set
import subprocess as sp
import numpy as np
import sklearn.model_selection as sm
import sklearn.linear_model as sl

if __name__ == '__main__':
    sp.call('cls', shell = True)

    # load some data
    with open('.\\data\\iris\\iris.data', 'rt') as fl:
        df = pd.read_csv(fl
            ,names = ['sepal_length','sepal_width','petal_length','petal_width','class']
            ,header = None
            ,index_col = False)

    # specify the x and y matrices
    xcols = ['sepal_length','sepal_width','petal_length','petal_width']
    ycols = ['class']
    X = df.loc[:, xcols].values
    y = np.ravel(df.loc[:, ycols].values)

    # split the data into test and training sets
    Xtrain, Xtest, ytrain, ytest = sm.train_test_split(X, y, test_size = 0.33
        ,random_state = 0)

    # fit a model to training data
    model = sl.LogisticRegression().fit(Xtrain, ytrain)

    # evaluate model on test set
    score = model.score(Xtest, ytest)
    print('LogisticRegression accuracy: {0}'.format(score))