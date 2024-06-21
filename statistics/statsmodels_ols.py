import numpy as np
from statsmodels import api as sm

# *********************************************************************
#
# DATA CREATION
#
# *********************************************************************
n = 100  # define number of samples
x = np.linspace(0, 10, n)  # create independent variable from 0 to 10
X = np.column_stack((x, x**2))  # create two columns from the tuple of arrays
beta = np.array([1, 0.1, 10])  # specify the true value of the parameters
e = np.random.normal(size=n)  # simulate random error from a normal distribution

# add constant to the independent variable matrix for regression intercept
X = sm.add_constant(X)

# take sum-product of X matrix and simulated true parameters and add error
y = np.dot(X, beta) + e

# *********************************************************************
#
# MODEL FITTING
#
# *********************************************************************
model = sm.OLS(y, X)  # create the model object
results = model.fit()  # perform ordinary least squares fitting
print(results.summary())
print("Parameters: ", results.params)
print("R2: ", results.rsquared)
