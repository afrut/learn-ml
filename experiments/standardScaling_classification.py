# exec(open('experiments\\standardScaling_classification.py').read())
# What effect does standard scaling have on different classification algorithms?
# The following algorithms are sort of "indifferent" to standard scaling:
# - ensemble.AdaBoostClassifier
# - ensemble.BaggingClassifier with tree.DecisionTreeClassifier as base estimator
# - ensemble.ExtraTreesClassifier
# - tree.ExtraTreeClassifier
# - ensemble.GradientBoostingClassifier
# - ensemble.HistGradientBoostingClassifier
# - neighbors.KNeighborsClassifier (but in theory this should always be better with scaling)
# - discriminant_analysis.LinearDiscriminantAnalysis
# - discriminant_analysis.QuadraticDiscriminantAnalysis (but it is better in the adult dataset)
# - ensemble.RandomForestClassifier
# - linear_model.RidgeClassifier
# - tree.DecisionTreeClassifier
# The following algorithms are better with standard scaling
# - naive_bayes.GaussianNB (but only on the adult dataset)
# - neural_network.MLPClassifier
# - linear_model.PassiveAggressiveClassifier
# - linear_model.Perceptron (but not better on the adult dataset)
# - linear_model.SGDClassifier (scaling is necessary for this one)
# The following algorithms are better off WITHOUT standard scaling
# - naive_bayes.BernoulliNB (but this is kind of dependent on the dataset)
# - svm.SVC (slightly worse; worse in the wineCultivar dataset)
import os
import pickle as pk

import datacfg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.discriminant_analysis as da
import sklearn.ensemble as ensemble
import sklearn.linear_model as slm
import sklearn.model_selection as sms
import sklearn.naive_bayes as nb
import sklearn.neighbors as neighbors
import sklearn.neural_network as nn
import sklearn.pipeline as pipeline
import sklearn.preprocessing as pp
import sklearn.svm as svm
import sklearn.tree as tree
from sklearn.experimental import enable_hist_gradient_boosting

if __name__ == "__main__":
    # ----------------------------------------
    # Constants
    # ----------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    np.set_printoptions(precision=4, suppress=True)
    pd.options.display.float_format = "{:10,.4f}".format
    seed = 29
    figsize = (16, 10)

    # ----------------------------------------
    # specify cross-validation
    # ----------------------------------------
    k = 10  # number of folds
    cvsplitter = sms.KFold(
        n_splits=k, shuffle=True, random_state=0
    )  # cross-validation splitter

    # ----------------------------------------
    # Model Definitions
    # ----------------------------------------
    models = dict()
    # models['GPC'] = (gp.GaussianProcessClassifier, {'kernel': 1.0 * gpk.RBF(1.0)}, {})      # kind of slow
    # models['LR'] = (slm.LogisticRegression, {'max_iter': 100000}, {})                       # depending on dataset, needs high max iterations
    models["PAC"] = (slm.PassiveAggressiveClassifier, {}, {})
    models["PERCPT"] = (slm.Perceptron, {}, {})
    models["RIDGE"] = (slm.RidgeClassifier, {}, {})
    models["SGD"] = (slm.SGDClassifier, {}, {})
    models["BernNB"] = (nb.BernoulliNB, {}, {})
    # models['CatNB'] = (nb.CategoricalNB, {}, {}) # look into this further, does not allow negative values
    # models['CompNB'] = (nb.ComplementNB, {}, {}) # does not allow negative values
    models["GaussNB"] = (nb.GaussianNB, {}, {})
    # models['MultinNB'] = (nb.MultinomialNB, {}, {}) # does not allow negative values
    models["KNN"] = (neighbors.KNeighborsClassifier, {}, {})
    # models['RNN'] = (neighbors.RadiusNeighborsClassifier, {'radius': 10}, {})
    models["MLP"] = (nn.MLPClassifier, {"max_iter": 100000}, {})
    # models['LinearSVC'] = (svm.LinearSVC, {'max_iter': 10000}, {})
    # models['NuSVC'] = (svm.NuSVC, {}, {})
    models["SVC"] = (svm.SVC, {}, {})
    models["TREE"] = (tree.DecisionTreeClassifier, {}, {})
    models["EXTREE"] = (tree.ExtraTreeClassifier, {}, {})
    models["QDA"] = (da.QuadraticDiscriminantAnalysis, {}, {})
    models["LDA"] = (da.LinearDiscriminantAnalysis, {}, {})
    models["BAGTREE"] = (
        ensemble.BaggingClassifier,
        {
            "random_state": seed,
            "estimator": tree.DecisionTreeClassifier(),
            "n_estimators": 30,
        },
        {},
    )
    models["ET"] = (ensemble.ExtraTreesClassifier, {}, {})
    models["ADA"] = (ensemble.AdaBoostClassifier, {}, {})
    models["GBM"] = (ensemble.GradientBoostingClassifier, {}, {})
    models["RF"] = (ensemble.RandomForestClassifier, {}, {})
    models["HISTGBM"] = (ensemble.HistGradientBoostingClassifier, {}, {})

    modelnames = list(models.keys())

    # ----------------------------------------
    # Pipeline definition
    # ----------------------------------------
    pipelines = dict()
    for entry in models.items():
        modelname = entry[0]
        model = entry[1][0]
        args = entry[1][1]
        params = entry[1][2]

        exclude = set(["SGD"])
        if modelname not in exclude:
            pipelines[modelname] = pipeline.Pipeline([(modelname, model(**args))])
        exclude = set(["CompNB", "MultinNB", "NuSVC"])
        if modelname not in exclude:
            pipelines["Scaled" + modelname] = pipeline.Pipeline(
                [("Scaler", pp.StandardScaler()), (modelname, model(**args))]
            )

    # ----------------------------------------
    # Data loading and formatting
    # ----------------------------------------
    datasets = datacfg.datacfg()
    dfScores = None
    dfScoresAll = None
    dfScoresScaled = None
    datasetnames = ["adult"]
    datasetnames = [
        "abalone",
        "bankMarketing",
        "wineCultivar",
        "winequality-red",
        "winequality-white",
        "iris",
        "pima",
        "adult",
    ]
    for datasetname in datasetnames:
        if "classification" in datasets[datasetname]["probtype"]:
            with open(datasets[datasetname]["filepath"], "rb") as fl:
                df = pk.load(fl)

            # check that there are no missing values
            assert np.all(np.logical_not(df.isna())), "Nan values present"
            ycols = datasets[datasetname]["targets"]
            xcolsnum = list(set(df.select_dtypes([np.number]).columns) - set(ycols))
            xcolsnonnum = list(set(df.select_dtypes([object]).columns) - set(ycols))

            if len(xcolsnonnum) > 0:
                # one-hot encoding of any categorical variables
                Xnonnum = df.loc[:, xcolsnonnum].values
                ohe = pp.OneHotEncoder(sparse_output=False, drop="first")
                ohe.fit(Xnonnum)
                XnonnumOhe = ohe.transform(Xnonnum)

                # check that the excluded variable is the first variable
                excluded = ohe.categories_[0][0]
                idx = (
                    XnonnumOhe.sum(axis=1) == 0
                )  # find all rows that don't fit in another category
                assert np.all(Xnonnum[idx] == excluded)

            # concatenate to arrive at final arrays
            xcols = xcolsnum + xcolsnonnum
            X = df.loc[:, xcolsnum].values
            y = np.ravel(df.loc[:, ycols].values)
            if len(xcolsnonnum) > 0:
                X = np.concatenate((X, XnonnumOhe), axis=1)

                # modify labels of features after one-hot encoding
                xcols = xcolsnum.copy()
                for cat in ohe.categories_[0][1:]:
                    xcols.append(cat)

            # ----------------------------------------
            # pipeline fitting and scoring
            # ----------------------------------------
            print(
                "Pipleine fitting and scoring progress: name - mean accuracy - std accuracy"
            )
            scoring = "accuracy"
            pipelinenames = list()
            scoresAll = list()
            for entry in pipelines.items():
                pipelinename = entry[0]
                print("    {0:<20}".format(pipelinename), end="")
                ppl = entry[1]
                score = -1 * sms.cross_val_score(
                    ppl, X, y, cv=cvsplitter, scoring=scoring
                )
                scoremean = np.mean(score)
                scorestd = np.std(score)
                print("{0:.4f} - {1:.4f}".format(np.mean(score), np.std(score, ddof=1)))
                scoresAll.append(score)
                pipelinenames.append(entry[0])
            print("")

            # boxplot of results
            plt.close("all")
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1)
            ax.boxplot(scoresAll)
            ax.set_xticklabels(pipelinenames)
            ax.set_xlabel("Algorithm")
            ax.set_ylabel("Mean Absolute Error")
            ax.set_title("Mean Absolute Error of Different Algorithms")

            # format every xticklabel
            for ticklabel in ax.get_xticklabels():
                ticklabel.set_horizontalalignment("right")  # center, right, left
                ticklabel.set_rotation_mode("anchor")  # None or anchor
                ticklabel.set_rotation(60)  # angle of rotation
                ticklabel.set_fontsize(12)  # float

            fig.tight_layout()
            fig.savefig(
                f"{script_dir}/standardScaling_classification_" + datasetname + ".png",
                format="png",
            )
            # plt.show()
            plt.close("all")

            # write results to a dataframe
            if dfScoresAll is None:
                dfScoresAll = pd.DataFrame(index=pipelinenames)
            dfScoresAll[datasetname] = np.array(scoresAll).mean(axis=1)

    # compare metrics of different models on different datasets for scaled
    # and non-scaled
    modelnames = list()
    modelnamesScaled = list()
    vals = list()
    valsScaled = list()
    for tpl in dfScoresAll.itertuples():
        pipelinename = tpl[0]
        val = tpl[1:]
        if "Scaled" in pipelinename:
            modelnamesScaled.append(pipelinename[6:])
            valsScaled.append(val)
        else:
            modelnames.append(pipelinename)
            vals.append(val)
    dfScores = pd.DataFrame(data=vals, index=modelnames, columns=datasetnames)
    dfScoresScaled = pd.DataFrame(
        data=valsScaled, index=modelnamesScaled, columns=datasetnames
    )
    dfImprovement = (dfScores - dfScoresScaled) / dfScores * 100
    print(dfImprovement)
    with open(f"{script_dir}/standardScaling_classification.pkl", "wb") as fl:
        pk.dump(dfImprovement, fl)
