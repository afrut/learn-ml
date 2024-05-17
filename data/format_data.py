import os
import pickle as pk
import re

import numpy as np
import pandas as pd
from IPython import embed

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # initialize data configuration dictionary
    datacfg = dict()

    # iris dataset
    dfIris = pd.read_csv(
        os.path.join(script_dir, "iris/iris.data"),
        names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"],
        header=None,
        index_col=False,
    )
    dfIris.loc[34, "petal_width"] = 0.2
    dfIris.loc[37, "sepal_width"] = 3.6
    dfIris.loc[37, "petal_length"] = 1.4
    pk.dump(dfIris, open(os.path.join(script_dir, "iris.pkl"), "wb"))

    # boston housing datasetl
    data_file = os.path.join(script_dir, "bostonHousing/data")
    raw_df = pd.read_csv(
        data_file,
        sep="\s+",
        skiprows=22,
        header=None,
    )
    pattern = re.compile(r"^\s+([A-Z]+).*$")
    with open(data_file, "rt") as fl:
        n = 0
        column_names = []
        for line in fl.readlines():
            n += 1
            if n > 7:
                column_names.append(re.match(pattern, line).groups()[0])
            if n > 20:
                break
    column_names.pop(column_names.index("MEDV"))
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    X = data
    y = target
    features = [feature for feature in column_names]
    targets = ["MEDV"]
    data = np.concatenate([X, y[:, np.newaxis]], axis=1)
    dfBoston = pd.DataFrame(data=data, columns=features + targets)
    pk.dump(dfBoston, open(os.path.join(script_dir, "bostonHousing.pkl"), "wb"))

    # wine quality dataset
    dfWineQualityRed = pd.read_csv(
        os.path.join(script_dir, "wineQuality/winequality-red.csv"), header=0
    )
    pk.dump(
        dfWineQualityRed, open(os.path.join(script_dir, "winequality-red.pkl"), "wb")
    )
    dfWineQualityWhite = pd.read_csv(
        os.path.join(script_dir, "wineQuality/winequality-white.csv"), header=0
    )
    pk.dump(
        dfWineQualityWhite,
        open(os.path.join(script_dir, "winequality-white.pkl"), "wb"),
    )

    # abalone dataset
    dfAbalone = pd.read_csv(
        os.path.join(script_dir, "abalone/abalone.data"),
        names=[
            "sex",
            "length",
            "diameter",
            "height",
            "whole weight",
            "shucked weight",
            "viscera weight",
            "shell weight",
            "rings",
        ],
    )
    pk.dump(dfAbalone, open(os.path.join(script_dir, "abalone.pkl"), "wb"))

    # auto mpg dataset
    dfAutoMpg = pd.read_csv(
        os.path.join(script_dir, "autoMpg/auto-mpg.data"),
        names=[
            "mpg",
            "cylinders",
            "displacement",
            "horsepower",
            "weight",
            "acceleration",
            "model year",
            "origin",
            "car name",
        ],
    )
    dfAutoMpg.drop(["car name"], axis=1, inplace=True)
    idx = np.logical_not(dfAutoMpg.loc[:, "horsepower"] == "?")
    dfAutoMpg = dfAutoMpg.loc[idx, :]
    dfAutoMpg.loc[:, "horsepower"] = dfAutoMpg.loc[:, "horsepower"].astype(np.float64)
    pk.dump(dfAutoMpg, open(os.path.join(script_dir, "autoMpg.pkl"), "wb"))

    # bank marketing dataset
    dfBankMarketing = pd.read_csv(
        os.path.join(script_dir, "bankMarketing/bank.csv"), header=0
    )
    pk.dump(dfBankMarketing, open(os.path.join(script_dir, "bankMarketing.pkl"), "wb"))

    # heart disease wisconsin data set
    dfHeartDisease = pd.read_csv(
        os.path.join(script_dir, "heartDiseaseWisconsin/processed.cleveland.data"),
        names=[
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
            "num",
        ],
    )
    pk.dump(
        dfHeartDisease,
        open(os.path.join(script_dir, "heartDiseaseWisconsin.pkl"), "wb"),
    )

    # wine cultivars data set
    dfWineCult = pd.read_csv(
        os.path.join(script_dir, "wineCultivar/wine.data"),
        names=[
            "cultivar",
            "alcohol",
            "malic acid",
            "ash",
            "alcalinity of ash  ",
            "magnesium",
            "total phenols",
            "flavanoids",
            "nonflavanoid phenols",
            "proanthocyanins",
            "color intensity",
            "hue",
            "od280/od315 of diluted wines",
            "proline",
        ],
    )
    pk.dump(dfWineCult, open(os.path.join(script_dir, "wineCultivar.pkl"), "wb"))

    # pima indians diabetes dataset
    dfPima = pd.read_csv(
        os.path.join(script_dir, "pima/pima-indians-diabetes.csv"),
        names=[
            "pregnancies",
            "glucose",
            "bloodPressure",
            "skinThickness",
            "insulin",
            "bmi",
            "diabetesPedigreef",
            "age",
            "class",
        ],
    )
    pk.dump(dfPima, open(os.path.join(script_dir, "pima.pkl"), "wb"))

    # adult census income dataset
    dfAdult = pd.read_csv(
        os.path.join(script_dir, "adult/adult_all.data"),
        names=[
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
            "class",
        ],
    )
    pk.dump(dfAdult, open(os.path.join(script_dir, "adult.pkl"), "wb"))
