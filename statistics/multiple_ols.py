import os
import pickle as pk

import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model as lm

# Load the data
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.abspath(os.path.join(script_dir, "../data/bostonHousing.pkl"))
with open(data_path, "rb") as fl:
    df = pk.load(fl)
print(df.columns)

# **********************************************************************
#
# MODEL FITTING
#
# **********************************************************************
# create the features DataFrame
X = df.drop("MEDV", axis=1)

# create a LinearRegression object
lr = lm.LinearRegression()

# fit a model to the data
lr.fit(X, df.MEDV)

print("Estimated intercept coefficient: " + str(lr.intercept_) + "\n")
print("Number of coefficients: " + str(len(lr.coef_)) + "\n")

# create a DataFrame of coefficients and feature names
coef = pd.DataFrame(
    list(zip(X.columns, lr.coef_)), columns=["features", "estimatedCoefficients"]
)
print("Coefficients:\n" + str(coef) + "\n")

# **********************************************************************
#
# MODEL FITTING WITH DATA SPLITTING
#
# **********************************************************************
# split the data set into training and testing sets
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
    X, df.MEDV, test_size=0.33, random_state=5
)

# refit based on the training data set
lr = lm.LinearRegression()
lr.fit(X_train, Y_train)
pred_train = lr.predict(X_train)
pred_test = lr.predict(X_test)

print(
    "Fit a model X_train, and calculate MSE with Y_train:",
    np.mean((Y_train - lr.predict(X_train)) ** 2),
)
print(
    "Fit a model X_train, and calculate MSE with X_test, Y_test:",
    np.mean((Y_test - lr.predict(X_test)) ** 2),
)

# **********************************************************************
#
# DIAGNOSTICS
#
# **********************************************************************
# plt.scatter( df.RM, df.MEDV )
# plt.xlabel( 'Average number of rooms per dwelling (RM)' )
# plt.ylabel( 'Housing Price' )
# plt.title( 'Relationship between RM and Price' )
# plt.show()

# predict house prices using the model
lr.predict(X)[0:5]

# plt.scatter( df.MEDV, lr.predict(X) )
# plt.xlabel( 'Prices: $Y_i$' )
# plt.ylabel( 'Predicted prices: $\hat{Y}_i$' )
# plt.title( 'Prices vs Predicted Prices: $Y_i$ vs $\hat{Y}_i$' )
# plt.show()

# calculate the mean squared error
mse = np.mean((df.MEDV - lr.predict(X)) ** 2)

# create a residual plot
# plt.scatter( lr.predict(X_train)
#            , lr.predict(X_train) - Y_train
#            , c = 'b'
#            , s = 40
#            , alpha = 0.5 )
# plt.scatter( lr.predict(X_test)
#            , lr.predict(X_test) - Y_test
#            , c = 'g'
#            , s = 40 )
# plt.hlines( y = 0, xmin = 0, xmax = 50 )
# plt.title( 'Residual Plot using training (blue) and test (green) data' )
# plt.ylabel( 'Residuals' )
# plt.show()

# **********************************************************************
#
# PLOTTING
#
# **********************************************************************
# box and whisker plots of the data
# df.plot( kind = 'box'
#        , subplots = True
#        , layout = (4,4)
#        , sharex = False
#        , sharey = False )
# plt.show()

# histograms of the data
# df.hist()
# plt.show()

# scatter plot matrix
# pd.plotting.scatter_matrix( df )
# plt.show()
