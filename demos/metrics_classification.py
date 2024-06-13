#exec(open('templates\\metrics_classification.py').read())
# assessing a classification model's efficacy
import subprocess as sp
import numpy as np
import pickle as pk
import sklearn.model_selection as ms
import sklearn.linear_model as sl
import sklearn.metrics as sm

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

    k = 10                                                                  # number of folds
    cvsplitter = ms.KFold(n_splits = k, shuffle = True, random_state = 0)   # cross-validation splitter
    model = sl.LogisticRegression(max_iter = 1000)                          # select a model to use
    scoring = 'accuracy'                                                    # metric to use
    score = ms.cross_val_score(model, X, y, cv = cvsplitter, scoring = scoring)
    print('{0} mean: {1:.4f}'.format(scoring, score.mean()))
    print('{0} std: {1:.4f}'.format(scoring, score.std()))

    # logloss - reward or punish predictions proportional to the confidence of
    # the prediction
    # smaller logloss is better; 0 is perfect logloss
    # in sklearn, larger is better
    scoring = 'neg_log_loss'
    score = ms.cross_val_score(model, X, y, cv = cvsplitter, scoring = scoring)
    print('{0} mean: {1:.4f}'.format(scoring, score.mean()))
    print('{0} std: {1:.4f}'.format(scoring, score.std()))

    # area under roc curve - tradeoff between sensitivity and specificty
    # an area of 1 represents making all predictions perfectly
    scoring = 'roc_auc'
    score = ms.cross_val_score(model, X, y, cv = cvsplitter, scoring = scoring)
    print('{0} mean: {1:.4f}'.format(scoring, score.mean()))
    print('{0} std: {1:.4f}'.format(scoring, score.std()))
    print('')

    # confusion matrix - compare prediction with actual
    Xtrain, Xtest, ytrain, ytest = ms.train_test_split(X, y, test_size = 0.33
        ,random_state = 0)
    model = sl.LogisticRegression(max_iter = 1000)
    model.fit(Xtrain, ytrain)
    predicted = model.predict(Xtest)
    confmat = sm.confusion_matrix(ytest, predicted)     # create confusion matrix based on test set prediction vs actual
    print('Confusion Matrix:')
    print(confmat)
    print('Number of predictions where ytest = 0, prediction = 0: {0}'.format(confmat[0, 0]))
    print('Number of predictions where ytest = 0, prediction = 1: {0}'.format(confmat[0, 1]))
    print('Number of predictions where ytest = 1, prediction = 0: {0}'.format(confmat[1, 0]))
    print('Number of predictions where ytest = 1, prediction = 1: {0}'.format(confmat[1, 1]))
    print('')

    # classification report
    report = sm.classification_report(ytest, predicted)
    print('Classification Report:\n{0}\n'.format(report))