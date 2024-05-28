#exec(open('.\\templates\\preproc_column_transformer.py').read())
import subprocess as sp
import importlib as il
import pickle as pk
import numpy as np
import sklearn.compose as sc
import sklearn.preprocessing as pp
import sklearn.pipeline as pl
import sklearn.ensemble as ensemble
import sklearn.model_selection as ms

import datacfg

if __name__ == '__main__':
    sp.call('cls', shell = True)
    il.reload(datacfg)

    with open(datacfg.datacfg()['adult']['filepath'], 'rb') as fl:
        df = pk.load(fl)

    # Set feature and target columns.
    ycols = set(['class'])
    xcols = set(df.columns) - ycols

    # Set numeric and non-numeric columns.
    numerics = set(df.select_dtypes([np.number]).columns)
    nonnumerics = xcols - numerics
    # xcols = xcols - set(['native-country'])
    xcols = list(xcols)
    idxnumerics = [xcols.index(col) for col in numerics]
    idxnonnumerics = [xcols.index(col) for col in nonnumerics]

    # Designate data.
    X = df.loc[:, xcols].values
    y = np.ravel(df.loc[:, ycols].values)

    # Split data.
    Xtrain, Xtest, ytrain, ytest = ms.train_test_split(X, y, test_size = 0.33
        ,random_state = 0)

    # Cross-validation.
    k = 3
    cvsplitter = ms.KFold(n_splits = k, shuffle = True, random_state = 0)

    # Apply a transformation for each column.
    transformers = list()
    transformers.append(('StandardScaler', pp.StandardScaler(), idxnumerics))
    transformers.append(('OneHotEncoder', pp.OneHotEncoder(sparse = False, drop = 'first', handle_unknown = 'ignore'), idxnonnumerics))
    ct = sc.ColumnTransformer(transformers, remainder = 'passthrough')
    ct.fit(Xtrain)
    Xtrain_transformed = ct.transform(Xtrain)
    print('Feature Names: {0}'.format(ct.get_feature_names_out()))

    # Use the transformer in a pipeline.
    estimators = list()
    estimators.append(('ColumnTransformer', sc.ColumnTransformer(transformers, remainder = 'passthrough')))
    estimators.append(('RandomForestClassifier', ensemble.RandomForestClassifier(n_estimators = 100, max_features = 3)))
    ppl = pl.Pipeline(estimators)
    accuracy = ms.cross_val_score(ppl, Xtrain, ytrain, cv = cvsplitter)
    print('Accuracy of pipeline: {0:.2f}'.format(accuracy.mean()))