#exec(open('templates\\proj_template_regression.py').read())
# TODO: list out hyperparameters
import subprocess as sp
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import math
import pandas as pd
import sklearn.model_selection as sms
import sklearn.pipeline as pipeline
import sklearn.linear_model as slm
import sklearn.neighbors as neighbors
import sklearn.svm as svm
import sklearn.ensemble as ensemble
import sklearn.tree as tree
import sklearn.preprocessing as pp
import sklearn.kernel_ridge as kr
import sklearn.gaussian_process as gp
import sklearn.cross_decomposition as cd
import sklearn.neural_network as nn
import sklearn.isotonic as si
import sklearn.metrics as metrics
from sklearn.experimental import enable_hist_gradient_boosting

if __name__ == '__main__':
    sp.call('cls', shell = True)
    plt.close('all')

    # ----------------------------------------
    # Data loading and formatting
    # ----------------------------------------
    with open('.\\data\\bostonHousing.pkl', 'rb') as fl:
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
    ycols = ['PRICE']
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
    models['LR'] = (slm.LinearRegression, {}, {})
    models['RIDGE'] = (slm.Ridge, {'random_state': seed}, {'alpha': np.logspace(-6, 6, 11)})
    models['LASSO'] = (slm.Lasso, {'random_state': seed}, {'alpha': np.logspace(-6, 6, 11)})
    #models['MTLASSO'] = (slm.MultiTaskLasso, {}, {})
    models['EN'] = (slm.ElasticNet, {'random_state': seed}, {'alpha': np.logspace(-6, 6, 11)})
    #models['MTEN'] = (slm.MultiTaskElasticNet, {}, {})
    models['LARS'] = (slm.Lars, {'random_state': seed}, {})
    models['LASSOLARS'] = (slm.LassoLars, {'random_state': seed}, {'alpha': np.logspace(-6, 6, 11)})
    #models['ISOR'] = (si.IsotonicRegression, {}, {})
    models['OMP'] = (slm.OrthogonalMatchingPursuit, {}, {})
    models['BRIDGE'] = (slm.BayesianRidge, {}, {})
    models['ARD'] = (slm.ARDRegression, {}, {})
    models['TW'] = (slm.TweedieRegressor, {'max_iter': 10000}, {})
    models['POISSON'] = (slm.PoissonRegressor, {'max_iter': 10000}, {})
    #models['GAMMA'] = (slm.GammaRegressor, {}, {})
    models['SGD'] = (slm.SGDRegressor, {}, {}) # always standard-scale this
    models['PA'] = (slm.PassiveAggressiveRegressor, {}, {})
    models['HUBER'] = (slm.HuberRegressor, {'max_iter': 10000}, {})
    models['RANSAC'] = (slm.RANSACRegressor, {'random_state': seed}, {})
    models['TH'] = (slm.TheilSenRegressor, {'random_state': seed}, {})
    models['KRR'] = (kr.KernelRidge, {}, {})
    models['GPR'] = (gp.GaussianProcessRegressor, {'random_state': seed}, {})
    models['PLS'] = (cd.PLSRegression, {}, {}) # don't include this in the voting regressor
    models['KNN'] = (neighbors.KNeighborsRegressor, {}, {})
    #models['RADIUSNN'] = (neighbors.RadiusNeighborsRegressor, {}, {})
    models['SVM'] = (svm.SVR, {}, {})
    #models['LSVM'] = (svm.LinearSVR, {'max_iter': 100000}, {})
    models['TREE'] = (tree.DecisionTreeRegressor, {}, {})
    models['EXTREE'] = (tree.ExtraTreeRegressor, {}, {})
    models['BAGTREE'] = (ensemble.BaggingRegressor, {'random_state': seed, 'base_estimator': tree.DecisionTreeRegressor(), 'n_estimators': 30}, {})
    models['AB'] = (ensemble.AdaBoostRegressor, {'random_state': seed}, {})
    models['GBM'] = (ensemble.GradientBoostingRegressor, {'random_state': seed}, {})
    models['HISTGBM'] = (ensemble.HistGradientBoostingRegressor, {'random_state': seed}, {})
    models['RF'] = (ensemble.RandomForestRegressor, {'random_state': seed}, {})
    models['ET'] = (ensemble.ExtraTreesRegressor, {'random_state': seed}, {})
    models['NN'] = (nn.MLPRegressor, {'max_iter': 10000, 'random_state': seed}, {})

    # create a voting regressor out of all the regressors
    estimators = list()
    for entry in models.items():
        name = entry[0]
        model = entry[1][0]
        args = entry[1][1]
        if name != 'PLS' and name != 'SGD':
            estimators.append((name, model(**args)))
    models['VOTE'] = (ensemble.VotingRegressor, {'estimators': estimators}, {})

    # create a stacking regressor out of all the regressors
    estimators = list()
    for entry in models.items():
        name = entry[0]
        model = entry[1][0]
        args = entry[1][1]
        if name != 'PLS' and name != 'SGD':
            estimators.append((name, model(**args)))
    models['STACK'] = (ensemble.StackingRegressor, {'estimators': estimators}, {})

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
        if name != 'SGD':   # SGD always needs scaling
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