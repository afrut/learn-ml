# TODO: complete this
import subprocess as sp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.datasets as ds
import sklearn.linear_model as lm
import sklearn.model_selection as ms
from IPython import embed

sp.call("cls", shell=True)

# **********************************************************************
#
# DATA PREPARATION
#
# **********************************************************************
# load the diabetes dataset
diabetes = ds.load_diabetes()  # diabetes is an sklearn.utils.Bunch type
X = pd.DataFrame(diabetes.data)  # features in a DataFrame
X.columns = diabetes.feature_names
Y = pd.Series(diabetes.target)  # target as pandas.Series
Y.columns = ["price"]
# print( 'Keys: \n', str( diabetes.keys() ) )                    # keys of the diabetes dictionary
print("Shape: \n", str(diabetes.data.shape), "\n")  # shape of dataset
print("Features: \n", str(diabetes.feature_names), "\n")  # features
# print( 'Description: \n' , diabetes.DESCR , '\n' )              # description
# print( 'Head: \n', str( X.head() ) )                            # first few rows of the dataset
# print( 'Tail: \n', str( X.tail() ) )                            # first few rows of the dataset

# **********************************************************************
#
# MODEL FITTING
#
# **********************************************************************
lr = lm.LinearRegression()  # create LinearRegression object
X = np.array(X)
Y = np.array(Y)
lr.fit(X, Y)  # fit a linear model

# create a DataFrame of coefficients and feature names
coef = pd.DataFrame(
    list(zip(diabetes.feature_names, lr.coef_)), columns=["features", "coefficients"]
)

# append intercept to DataFrame
coef = pd.concat(
    [
        coef,
        pd.DataFrame(list(zip(["intercept"], [lr.intercept_])), columns=coef.columns),
    ],
    axis=0,
    ignore_index=True,
)
print("Estimated intercept coefficient: ", str(lr.intercept_) + "\n")
print("Number of coefficients: ", str(len(lr.coef_)) + "\n")
print("Coefficients:\n", str(coef), "\n")

# row array of coefficients and last element is intercept
beta = np.array(coef.coefficients)[:, np.newaxis]

# append a row of ones to X for the intercept
X = np.append(X, np.ones([X.shape[0], 1]), axis=1)

# predict the dependent variable based on fitted estimates
Y_pred = np.dot(X, beta)

# alternatively, use sklearn's linear regression object
# but drop the ones for the intercept first
X = np.delete(X, X.shape[1] - 1, axis=1)
Y_pred = lr.predict(X)

# **********************************************************************
#
# MODEL WITH DATA SPLITTING
#
# **********************************************************************
# split the data set into training and testing sets
X_train, X_test, Y_train, Y_test = ms.train_test_split(
    diabetes.data, diabetes.target, test_size=0.33, random_state=5
)

# refit based on the training data set
lr = lm.LinearRegression()
lr.fit(X_train, Y_train)
Y_pred_train = lr.predict(X_train)
Y_pred_test = lr.predict(X_test)

# calculate squared errors
Y_pred_train_sse = round(np.sum((Y_train - Y_pred_train) ** 2), 4)
Y_pred_test_sse = round(np.sum((Y_test - Y_pred_test) ** 2), 4)
Y_pred_sse = round(np.sum((Y - Y_pred) ** 2), 4)
Y_pred_train_mse = round(np.mean((Y_train - Y_pred_train) ** 2), 4)
Y_pred_test_mse = round(np.mean((Y_test - Y_pred_test) ** 2), 4)
Y_pred_mse = round(np.mean((Y - Y_pred) ** 2), 4)
print("Y_pred_train_sse    =", Y_pred_train_sse)
print("Y_pred_test_sse     =", Y_pred_test_sse)
print("Y_pred_sse          =", Y_pred_sse)
print("Y_pred_train_mse    =", Y_pred_train_mse)
print("Y_pred_test_mse     =", Y_pred_test_mse)
print("Y_pred_mse          =", Y_pred_mse)

# **********************************************************************
#
# PLOTTING DIAGNOSTICS
#
# **********************************************************************
# create lists to store figure and axes handles
lsFig = list()
lsAx = list()

# prepare X and Y DataFrames for plotting
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
Y = pd.DataFrame(diabetes.target, columns=["prog"])
Y_pred = pd.DataFrame(Y_pred, columns=["prog"])

# scatter plot of Disease Progression against BMI
fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(X.bmi, Y.prog)
ax.set_xlabel("BMI of patient (bmi)")
ax.set_ylabel("Disease progression (prog)")
ax.set_title("Relationship of BMI and Disease Progression")

# scatter plot of predicted disease progress
# vs real disease progression
fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(Y.prog, Y_pred.prog)
ax.set_xlabel("Disease Progression: $Y_i$")
ax.set_ylabel("Predicted Disease Progression: $\hat{Y}_i$")
ax.set_title(
    "Disease Progression vs Predicted" + "Disease Progression: $Y_i$ vs $\hat{Y}_i$"
)

# create a residual plot
fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(Y_pred.prog, Y_pred.prog - Y.prog, c="b", s=40, alpha=0.5)
plt.hlines(y=0, xmin=0, xmax=300)
ax.set_title("Residual Plot")
ax.set_xlabel("Predicted Disease Progression")
ax.set_ylabel("Residuals")

# show all plots
plt.show()
