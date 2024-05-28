#exec(open('templates\\proj_template_classification.py').read())
import subprocess as sp
import matplotlib.pyplot as plt
import pickle as pk
import numpy as np
import math
import pandas as pd
import sklearn.model_selection as sms
import sklearn.pipeline as pipeline
import sklearn.preprocessing as pp
import sklearn.metrics as metrics
import sklearn.gaussian_process as gp
import sklearn.gaussian_process.kernels as gpk
import sklearn.linear_model as slm
import sklearn.naive_bayes as nb
import sklearn.neighbors as neighbors
import sklearn.neural_network as nn
import sklearn.svm as svm
import sklearn.tree as tree
import sklearn.discriminant_analysis as da
import sklearn.ensemble as ensemble
from sklearn.experimental import enable_hist_gradient_boosting

if __name__ == '__main__':
    sp.call('cls', shell = True)
    plt.close('all')

    # ----------------------------------------
    # Data loading and formatting
    # ----------------------------------------
    with open('.\\data\\pima.pkl', 'rb') as fl:
        df = pk.load(fl)
    # check that there are no missing values
    assert(np.all(np.logical_not(df.isna()))), 'Nan values present'

    # ----------------------------------------
    # Constants
    # ----------------------------------------
    np.set_printoptions(precision = 4, suppress = True)
    pd.options.display.float_format = '{:10,.4f}'.format
    seed = 29

    # ----------------------------------------
    # Descriptive statistics
    # ----------------------------------------
    print('Number of Rows: {0}'.format(df.shape[0]))
    print('Number of Columns: {0}'.format(df.shape[1]))
    print('')

    print('Column Names:')
    for col in df.columns:
        print('    ' + col)
    print('')

    print('First 20 rows:')
    print(df.head(20))
    print('')

    print('Last 20 rows:')
    print(df.tail(20))
    print('')

    print('Data types:')
    datatypes = df.dtypes
    for idx in datatypes.index:
        print('    {0} - {1}'.format(idx, datatypes[idx]))
    print('')

    print('Statistical Summary:')
    print(df.describe())
    print('')

    print('Correlations between variables:')
    print(df.corr())
    print('')

    # numbers closer to 0 mean the distribution is closer to Gaussian
    print('Skew of variables:')
    print(df.skew())
    print('')

    # ----------------------------------------
    # Descriptive plots
    # ----------------------------------------
    # get numeric types
    numerics = df.select_dtypes([np.number]).columns.to_numpy()

    # get non-numeric types
    nonnum = list(set(df.columns) - set(numerics))
    nonnum = np.array(nonnum)

    # determine layout of univariate plots
    numvar = len(numerics)
    numrows = int(math.sqrt(numvar))
    numcols = numrows
    while(numrows * numcols < numvar):
        numcols = numcols + 1
    layout = (numrows, numcols)

    # get only numeric data
    dfnumeric = df.loc[:, numerics]

    # matrix of histograms
    ret = dfnumeric.hist()

    # matrix of probability density functions
    ret = dfnumeric.plot(kind = 'density', subplots = True, layout = layout, sharex = False)
    
    # box and whisker plot for all numeric quantities
    ret = dfnumeric.plot(kind = 'box', subplots = True, layout = layout, sharex = False, sharey = False)
    
    # matrix/heatmap of correlations
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # plot the matrix of correlations, returns the colorbar image
    axesimage = ax.matshow(dfnumeric.corr(), vmin = -1, vmax = 1)

    # plot the colorbar image
    fig.colorbar(axesimage)

    # set x and y-axis ticks and ticklabels
    ticks = np.arange(0, len(dfnumeric.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(dfnumeric.columns)
    ax.set_yticklabels(dfnumeric.columns)

    # scatter matrix plot
    pd.plotting.scatter_matrix(dfnumeric)

    #plt.show()
    plt.close('all')

    # ----------------------------------------
    # Specify variables and target
    # ----------------------------------------
    # TODO: set the target variable here
    ycols = ['class']
    xcolsnumeric = list(set(df.select_dtypes([np.number]).columns) - set(ycols))
    xcolsnonnumeric = list(set(df.select_dtypes([object]).columns) - set(ycols))
    xcols = xcolsnumeric + xcolsnonnumeric
    X = df.loc[:, xcols].values
    y = np.ravel(df.loc[:, ycols].values)
    
    # ----------------------------------------
    # Validation set
    # ----------------------------------------
    validationSize = 0.2
    Xtrain, Xvalid, ytrain, yvalid = sms.train_test_split(X, y
        ,test_size = validationSize
        ,random_state = seed)

    # ----------------------------------------
    # specify cross-validation
    # ----------------------------------------
    k = 10                                                                   # number of folds
    cvsplitter = sms.KFold(n_splits = k, shuffle = True, random_state = 0)   # cross-validation splitter

    # ----------------------------------------
    # Try different piplines
    # - base model
    # - standardized/normalized/min-max-scaled
    # - one-hot encoding
    # - feature selection pipeline
    # - tuning pipeline
    # ----------------------------------------
    # define estimator parameters
    # format: models[name] = (constructor, constructor_args, hyperparameter_grid)
    # TODO: comment/uncomment as needed
    models = dict()
    #models['GPC'] = (gp.GaussianProcessClassifier, {'kernel': 1.0 *gpk.RBF(1.0)}, {}) # this is kind of slow
    models['LR'] = (slm.LogisticRegression, {'max_iter': 1000}, {})
    models['PAC'] = (slm.PassiveAggressiveClassifier, {}, {})
    models['PERCPT'] = (slm.Perceptron, {}, {})
    models['RIDGE'] = (slm.RidgeClassifier, {}, {})
    models['SGD'] = (slm.SGDClassifier, {}, {})
    models['BernNB'] = (nb.BernoulliNB, {}, {})
    #models['CatNB'] = (nb.CategoricalNB, {}, {}) # look into this further, does not allow negative values
    #models['CompNB'] = (nb.ComplementNB, {}, {}) # does not allow negative values
    models['GaussNB'] = (nb.GaussianNB, {}, {})
    #models['MultinNB'] = (nb.MultinomialNB, {}, {}) # does not allow negative values
    models['KNN'] = (neighbors.KNeighborsClassifier, {}, {})
    #models['RNN'] = (neighbors.RadiusNeighborsClassifier, {'radius': 10}, {})
    models['MLP'] = (nn.MLPClassifier, {'max_iter': 10000}, {})
    #models['LinearSVC'] = (svm.LinearSVC, {'max_iter': 10000}, {})
    #models['NuSVC'] = (svm.NuSVC, {}, {})
    models['SVC'] = (svm.SVC, {}, {})
    models['TREE'] = (tree.DecisionTreeClassifier, {}, {})
    models['EXTREE'] = (tree.ExtraTreeClassifier, {}, {})
    models['QDA'] = (da.QuadraticDiscriminantAnalysis, {}, {})
    models['LDA'] = (da.LinearDiscriminantAnalysis, {}, {})
    models['BAGTREE'] = (ensemble.BaggingClassifier, {'random_state': seed, 'base_estimator': tree.DecisionTreeClassifier(), 'n_estimators': 30}, {})
    models['ET'] = (ensemble.ExtraTreesClassifier, {}, {})
    models['ADA'] = (ensemble.AdaBoostClassifier, {}, {})
    models['GBM'] = (ensemble.GradientBoostingClassifier, {}, {})
    models['RF'] = (ensemble.RandomForestClassifier, {}, {})
    models['HISTGBM'] = (ensemble.HistGradientBoostingClassifier, {}, {})

    # ----------------------------------------
    # Pipeline definition
    # ----------------------------------------
    pipelines = dict()
    print('Pipeline creation:')
    for entry in models.items():
        name = entry[0]
        model = entry[1][0]
        args = entry[1][1]
        params = entry[1][2]

        # ----------------------------------------
        # Pipeline for current model without scaling
        # Tune hyperparameter on entire training set. Then use best hyperparameter
        # value in cross-validation. Alternatively, use nested hyperparameter
        # tuning with cross-validation to estimate generalization performance
        # ----------------------------------------
        exclude = set(['KNN','RNN','SGD','LinearSVC'])
        if name not in exclude:   # nearest neighbors always needs scaling
            print('    Creating pipeline for {0: <16} - '.format(name), end = '')
            ppl = pipeline.Pipeline([(name, model(**args))])

            # add in hyperparameter tuning if applicable
            if len(params) > 0:
                param_grid = dict()
                for tpl in params.items():
                    param_grid[name + '__' + tpl[0]] = tpl[1]
                ppl = sms.GridSearchCV(estimator = ppl, param_grid = param_grid)

            # add pipeline to collection of piplines to try
            pipelines[name] = ppl
            print('done')

        # ----------------------------------------
        # pipeline for current model with scaling
        # ----------------------------------------
        exclude = set(['CatNB','CompNB','MultinNB'])
        if name not in exclude:
            print('    Creating pipeline for {0: <16} - '.format('Scaled' + name), end = '')
            ppl = pipeline.Pipeline([('Scaler', pp.StandardScaler()), (name, model(**args))])

            # add in hyperparameter tuning if applicable
            if len(params) > 0:
                param_grid = dict()
                for tpl in params.items():
                    param_grid[name + '__' + tpl[0]] = tpl[1]
                ppl = sms.GridSearchCV(estimator = ppl, param_grid = param_grid)

            # add pipeline to collection of piplines to try
            pipelines['Scaled' + name] = ppl
            print('done')
    print('')

    # create voting and stacking classifiers
    # TODO: specify estimators for voting classifier
    estimators = list()
    estimators.append(('LR', pipelines['LR']))
    estimators.append(('ADA', pipelines['ADA']))
    estimators.append(('GBM', pipelines['GBM']))
    estimators.append(('RF', pipelines['RF']))
    estimators.append(('ScaledKNN', pipelines['ScaledKNN']))
    estimators.append(('RIDGE', pipelines['RIDGE']))
    estimators.append(('SVC', pipelines['SVC']))
    pipelines['VOTE'] = pipeline.Pipeline([('model', ensemble.VotingClassifier(estimators = estimators))])
    pipelines['STACK'] = pipeline.Pipeline([('model', ensemble.StackingClassifier(estimators = estimators))])

    # ----------------------------------------
    # pipeline fitting and scoring
    # ----------------------------------------
    print('Pipleine fitting and scoring progress: name - mean accuracy - std accuracy')
    scoring = 'neg_mean_absolute_error'
    pipelinenames = list()
    scores = list()
    for entry in pipelines.items():
        name = entry[0]
        print('    {0:<20}'.format(name), end = '')
        ppl = entry[1]
        score = -1 * sms.cross_val_score(ppl, Xtrain, ytrain, cv = cvsplitter, scoring = scoring)
        scores.append(score)
        pipelinenames.append(entry[0])
        print('{0:.4f} - {1:.4f}'.format(np.mean(score), np.std(score, ddof = 1)))
    print('')

    # boxplot of results
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.boxplot(scores)
    ax.set_xticklabels(pipelinenames)
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Mean Absolute Error of Different Algorithms')

    # format every xticklabel
    for ticklabel in ax.get_xticklabels():
        ticklabel.set_horizontalalignment('right')  # center, right, left
        ticklabel.set_rotation_mode('anchor')       # None or anchor
        ticklabel.set_rotation(60)                  # angle of rotation
        ticklabel.set_fontsize(12)                  # float

    fig.tight_layout()
    plt.show()
    plt.close('all')

    # table of results for cross validation
    arrscores = np.array(scores)
    dfScores = pd.DataFrame(index = pipelinenames)
    dfScores['mean'] = np.mean(arrscores, axis = 1)
    dfScores['std'] = np.std(arrscores, ddof = 1, axis = 1)
    print('Mean and standard deviation of MAE for different algorithms:')
    print(dfScores.sort_values(by = ['mean']))
    print('')

    # table of results on validation set/holdout set
    scores = list()
    pipelinenames = list()
    scorer = metrics.get_scorer(scoring)
    print('Validation set scoring:')
    for entry in pipelines.items():
        name = entry[0]
        print('    {0:<20}'.format(name), end = '')
        ppl = entry[1]
        ppl.fit(Xtrain, ytrain)
        score = -1 * scorer(ppl, Xvalid, yvalid)
        scores.append(score)
        pipelinenames.append(entry[0])
        print('{0:.4f}'.format(score))
    print('')

    # table of results for validation set/holdout set
    arrscoresvalid = np.array(scores)
    dfScoresValid = pd.DataFrame(index = pipelinenames)
    dfScoresValid['mae'] = arrscoresvalid
    print('MAE for different algorithms:')
    print(dfScoresValid.sort_values(by = ['mae']))
    print('')

    # table of results on entire dataset
    scores = list()
    pipelinenames = list()
    scorer = metrics.get_scorer(scoring)
    print('All data scoring:')
    for entry in pipelines.items():
        name = entry[0]
        print('    {0:<20}'.format(name), end = '')
        ppl = entry[1]
        ppl.fit(X, y)
        score = -1 * scorer(ppl, X, y)
        scores.append(score)
        pipelinenames.append(entry[0])
        print('{0:.4f}'.format(score))
    print('')

    # table of results on entire dataset
    arrscoresall = np.array(scores)
    dfScoresAll = pd.DataFrame(index = pipelinenames)
    dfScoresAll['mae'] = arrscoresall
    print('MAE for different algorithms:')
    print(dfScoresAll.sort_values(by = ['mae']))
    print('')