#exec(open('templates\\pipelines.py').read())
# demonstrate usage of pipleines to chain together sklearn workflows
import subprocess as sp
import numpy as np
import pickle as pk
import sklearn.model_selection as ms
import sklearn.linear_model as sl
import sklearn.discriminant_analysis as da
import sklearn.preprocessing as pp
import sklearn.pipeline as pipeline
import sklearn.decomposition as decomp
import sklearn.feature_selection as fs
import sklearn.base as sb

# A non-sensical custom transformer that multiplies data with a slope and intercept:
class CustomTransformer(sb.BaseEstimator, sb.TransformerMixin):
    def __init__(self, m, b):
        self.m = m
        self.b = b
    def fit(self, X, y = None):
        # No fitting method needed
        return self
    def transform(self, X, y = None):
        return (self.m * X) + self.b

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

    # define the steps in the pipeline
    estimators = list()
    estimators.append(('custom', CustomTransformer(1,0)))       # add a custom transformer
    estimators.append(('standardize', pp.StandardScaler()))     # preprocess by standardizing data
    estimators.append(('lda', da.LinearDiscriminantAnalysis())) # run LDA on the standardized data
    model = pipeline.Pipeline(estimators)                       # create a model from the steps in the pipeline

    # specify cross-validation
    k = 10                                                                  # number of folds
    cvsplitter = ms.KFold(n_splits = k, shuffle = True, random_state = 0)   # cross-validation splitter
    score = ms.cross_val_score(model, X, y, cv = cvsplitter)
    print('Standardized linear discriminant analysis mean accuracy: {0:.4f}'.format(score.mean()))

    # define steps in a feature selection pipeline
    features = list()
    features.append(('pca', decomp.PCA(n_components = 3)))      # use PCA to select 3 of the best features
    features.append(('select_best', fs.SelectKBest(k = 6)))     # use chi-squared test to select 6 of the best features
    feature_union = pipeline.FeatureUnion(features)             # create the feature selection pipeline
    estimators = list()
    estimators.append(('feature_union', feature_union))                         # add the feature selection pipleine to a new pipeline
    estimators.append(('logistic', sl.LogisticRegression(max_iter = 1000)))     # use logistic regression as the model
    model = pipeline.Pipeline(estimators)                                       # logistic regression with automatic feature selection by pca and chi-squared test

    # specify cross-validation
    score = ms.cross_val_score(model, X, y, cv = cvsplitter)
    print('Logistic regression with automatic feature selection by PCA and chi2 test mean accuracy: {0:.4f}'.format(score.mean()))
    