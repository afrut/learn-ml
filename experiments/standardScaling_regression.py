# exec(open('experiments\\standardScaling_regression.py').read())
# What effect does standard scaling have on different regression algorithms?
# The following algorithms are sort of "indifferent" to standard scaling:
# - linear_model.LinearRegression
# - linear_model.Ridge
# - linear_model.Lasso (check this on the dataset)
# - linear_model.ElasticNet
# - linear_model.Lars
# - linear_model.LassoLars
# - linear_model.OrthogonalMatchingPursuit
# - linear_model.ARDRegression
# - linear_model.BayesianRidge
# - linear_model.HuberRegressor
# - linear_model.RANSACRegressor
# - tree.DecisionTreeRegressor
# - tree.ExtraTreeRegressor
# - ensemble.AdaBoostRegressor
# - ensemble.BaggingRegressor with tree.DecisionTreeRegressor
# - ensemble.ExtraTreesRegressor
# - ensemble.GradientBoostingRegressor
# - ensemble.HistGradientBoostingRegressor
# - ensemble.RandomForestRegressor
# - cross_decomposition.PLSRegression
# - linear_model.TheilSenRegressor (check this on the dataset)
# - linear_model.TweedieRegressor (check this on the dataset)
# The following algorithms are better with standard scaling
# - linear_model.SGDRegressor (always standard scale this)
# - gaussian_process.GaussianProcessRegressor (check this on the dataset)
# - neighbors.KNeighborsRegressor
# - neural_network.MLPRegressor
# - linear_model.PassiveAggressiveRegressor
# - svm.SVR
# The following algorithms are better off WITHOUT standard scaling
# - kernel_ridge.KernelRidge
import pickle as pk
import subprocess as sp

import datacfg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.cross_decomposition as cd
import sklearn.ensemble as ensemble
import sklearn.gaussian_process as gp
import sklearn.kernel_ridge as kr
import sklearn.linear_model as slm
import sklearn.model_selection as sms
import sklearn.neighbors as neighbors
import sklearn.neural_network as nn
import sklearn.pipeline as pipeline
import sklearn.preprocessing as pp
import sklearn.svm as svm
import sklearn.tree as tree
from sklearn.experimental import enable_hist_gradient_boosting

if __name__ == "__main__":
    sp.call("cls", shell=True)

    # ----------------------------------------
    # Constants
    # ----------------------------------------
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
    models["LR"] = (slm.LinearRegression, {}, {})
    models["RIDGE"] = (slm.Ridge, {"random_state": seed}, {})
    models["LASSO"] = (slm.Lasso, {"random_state": seed}, {})
    # models['MTLASSO'] = (slm.MultiTaskLasso, {}, {})
    models["EN"] = (slm.ElasticNet, {"random_state": seed}, {})
    # models['MTEN'] = (slm.MultiTaskElasticNet, {}, {})
    models["LARS"] = (slm.Lars, {"random_state": seed}, {})
    models["LASSOLARS"] = (slm.LassoLars, {"random_state": seed}, {})
    # models['ISOR'] = (si.IsotonicRegression, {}, {})
    models["OMP"] = (slm.OrthogonalMatchingPursuit, {}, {})
    models["BRIDGE"] = (slm.BayesianRidge, {}, {})
    models["ARD"] = (slm.ARDRegression, {}, {})
    models["TW"] = (slm.TweedieRegressor, {"max_iter": 10000}, {})
    # models['POISSON'] = (slm.PoissonRegressor, {'max_iter': 10000}, {})
    # models['GAMMA'] = (slm.GammaRegressor, {}, {})
    models["SGD"] = (slm.SGDRegressor, {}, {})  # always standard-scale this
    models["PA"] = (slm.PassiveAggressiveRegressor, {}, {})
    models["HUBER"] = (slm.HuberRegressor, {"max_iter": 10000}, {})
    models["RANSAC"] = (slm.RANSACRegressor, {"random_state": seed}, {})
    models["TH"] = (slm.TheilSenRegressor, {"random_state": seed}, {})
    models["KRR"] = (kr.KernelRidge, {}, {})
    models["GPR"] = (gp.GaussianProcessRegressor, {"random_state": seed}, {})
    models["PLS"] = (
        cd.PLSRegression,
        {},
        {},
    )  # don't include this in the voting regressor
    models["KNN"] = (neighbors.KNeighborsRegressor, {}, {})
    # models['RADIUSNN'] = (neighbors.RadiusNeighborsRegressor, {}, {})
    models["SVM"] = (svm.SVR, {}, {})
    # models['LSVM'] = (svm.LinearSVR, {'max_iter': 100000}, {})
    models["TREE"] = (tree.DecisionTreeRegressor, {}, {})
    models["EXTREE"] = (tree.ExtraTreeRegressor, {}, {})
    models["BAGTREE"] = (
        ensemble.BaggingRegressor,
        {
            "random_state": seed,
            "estimator": tree.DecisionTreeRegressor(),
            "n_estimators": 30,
        },
        {},
    )
    models["AB"] = (ensemble.AdaBoostRegressor, {"random_state": seed}, {})
    models["GBM"] = (ensemble.GradientBoostingRegressor, {"random_state": seed}, {})
    models["HISTGBM"] = (
        ensemble.HistGradientBoostingRegressor,
        {"random_state": seed},
        {},
    )
    models["RF"] = (ensemble.RandomForestRegressor, {"random_state": seed}, {})
    models["ET"] = (ensemble.ExtraTreesRegressor, {"random_state": seed}, {})
    models["NN"] = (nn.MLPRegressor, {"max_iter": 10000, "random_state": seed}, {})

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

        if modelname != "SGD":
            pipelines[modelname] = pipeline.Pipeline([(modelname, model(**args))])
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
    datasetnames = [
        "abalone",
        "bostonHousing",
        "autoMpg",
        "heartDiseaseWisconsin",
        "winequality-red",
        "winequality-white",
    ]
    for datasetname in datasetnames:
        if "regression" in datasets[datasetname]["probtype"]:
            with open(datasets[datasetname]["filepath"], "rb") as fl:
                df = pk.load(fl)

            # dataset-specific pre-processing
            if datasetname == "heartDiseaseWisconsin":
                # remove all rows with missing data
                idx = np.any(df.mask(df == "?").isna().values, axis=1)
                idx = np.logical_not(idx)
                df = df.loc[idx, :]
                df["ca"] = df["ca"].astype(np.float64)
                df["thal"] = df["thal"].astype(np.float64)

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
            scoring = "neg_mean_absolute_error"
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
                ".\\experiments\\standardScaling_regression_" + datasetname + ".png",
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
    with open(".\\experiments\\standardScaling_regression.pkl", "wb") as fl:
        pk.dump(dfImprovement, fl)
