#exec(open('templates\\featsel_regression.py').read())
import subprocess as sp
import pandas as pd
import sklearn.feature_selection as sfs
import numpy as np
import pickle as pk
import sklearn.ensemble as ensemble

if __name__ == '__main__':
    sp.call('cls', shell = True)

    # load some data
    with open('.\\data\\bostonHousing.pkl', 'rb') as fl:
        df = pk.load(fl)

    # ----------------------------------------
    # Constants
    # ----------------------------------------
    np.set_printoptions(precision = 4, suppress = True)
    seed = 29
    figsize = (16, 10)

    # specify the x and y matrices
    ycols = ['PRICE']
    xcolsnum = list(set(df.select_dtypes([np.number]).columns) - set(ycols))
    xcolsnonnum = list(set(df.select_dtypes([object]).columns) - set(ycols))
    xcols = xcolsnum + xcolsnonnum
    X = df.loc[:, xcols].values
    y = np.ravel(df.loc[:, ycols].values)

    # number of features to select
    k = 2

    # ----------------------------------------
    # Select-k-best algorithms
    # ----------------------------------------
    selectors = dict()
    selectors['f_regression'] = sfs.SelectKBest(score_func = sfs.f_regression, k = k)
    selectors['mutual_info_regression'] = sfs.SelectKBest(score_func = sfs.mutual_info_regression, k = k)
    results = dict()
    for entry in selectors.items():
        selector = entry[1]
        selector.fit(X, y)
        srs = pd.Series(selector.scores_, index = xcols)
        srs.sort_values(ascending = False, inplace = True)
        results[entry[0]] = srs
        print('{0} feature rankings:'.format(entry[0]))
        for idx in srs.index:
            print('    {0: <20} - {1:.4f}'.format(idx, srs[idx]))
        print('')

    # ----------------------------------------
    # Feature importance
    # ----------------------------------------
    model = ensemble.ExtraTreesRegressor(random_state = seed)
    model.fit(X, y)
    srs = pd.Series(model.feature_importances_, index = xcols)
    srs.sort_values(ascending = False, inplace = True)
    results['ExtraTrees'] = srs
    print('ExtraTreesClassifier feature importance rankings:')
    for idx in srs.index:
        print('    {0: <20} - {1:.4f}'.format(idx, srs[idx]))
    print('')

    model = ensemble.RandomForestRegressor(random_state = seed)
    model.fit(X, y)
    srs = pd.Series(model.feature_importances_, index = xcols)
    srs.sort_values(ascending = False, inplace = True)
    results['RandomForest'] = srs
    print('RandomForestClassifier feature importance rankings:')
    for idx in srs.index:
        print('    {0: <20} - {1:.4f}'.format(idx, srs[idx]))
    print('')

    # ----------------------------------------
    # Recursive Feature Elimination
    # ----------------------------------------
    # specify any model
    model = ensemble.ExtraTreesRegressor(random_state = seed)
    selector = sfs.RFE(model, n_features_to_select = k).fit(X, y)
    srs = pd.Series(selector.ranking_, index = xcols)
    srs.sort_values(ascending = True, inplace = True)
    results['rfe'] = srs
    print('Recursive feature elimination feature rankings:')
    for idx in srs.index:
        print('    {0: <20} - {1:.4f}'.format(idx, srs[idx]))
    print('')

    # get best features according to all ranking schemes
    dfresults = pd.DataFrame(index = xcols)
    rank = np.arange(0, len(xcols))
    for entry in results.items():
        dfresults[entry[0]] = pd.Series(rank.copy(), index = entry[1].index)
    srs = dfresults.sum(axis = 1).sort_values()
    print('All features ranked:')
    for idx in srs.index:
        print('    {0}'.format(idx))
    print('')
    print('Best {0} features:'.format(k))
    for cnt in range(k):
        print('    {0}'.format(srs.index[cnt]))