"""
Bias vs. Variance 
> Methods explained how to fix these problems
> Regularized vs Non-regularized
"""

# libs
from ast import Lambda
from json import load
import numpy as np
from matplotlib import pyplot as plt
import os 
from scipy import optimize
from scipy.io import loadmat
import utils 

"""
Part I:
> Regularized Linear Regression
"""

# Import data and split into 3 parts: 
data = loadmat(os.path.join("Data", "ex5data1.mat"))
# X = change in the water level, y = water flowing out
"""
Splitting data into:
1) Training example - set that your model will learn obtaining theta parameters
2) Cross Validation - set for determining the regularization parameter
3) Test set - evaluating performance of model - unseen examples
"""
# Training data
X, y = data["X"], data["y"][:, 0] # first part of the column only

# CV data
X_val, y_val = data["Xval"], data["yval"][:, 0]

# Test data
X_test, y_test = data["Xtest"], data["ytest"][:, 0]

m, n = X.shape

"""
Application of Regularized Linear Regression
"""
def costFunctionReg(theta, X, y, lambda_):
    m, n = X.shape
    J = 0
    grad = np.zeros(theta.shape)

    h = np.dot(X, theta)

    # Cost Function Linear Regression
    temp = theta 
    temp[0] = 0

    reg = (lambda_ / (2 * m)) * np.sum(np.square(theta[1:]))
    J = (1 / (2 * m)) * np.sum(np.square(h - y)) + reg 

    grad = (1 / m) * np.dot((h - y), X) + (lambda_ / m) * temp
    
    return J, grad

# Concatenate X
X_intercept = np.concatenate([np.ones((m, 1)), X], axis=1)
theta = np.ones(X_intercept.shape[1])
cost, grad = costFunctionReg(theta, X_intercept, y, lambda_ = 1)

res_x = utils.trainLinearReg(costFunctionReg, X_intercept, y, lambda_=0)

fig = plt.figure()
plt.plot(X, y, 'ro', ms=10, mec='k', mew=1)
plt.xlabel("Change in water level (x)")
plt.ylabel("Water flowing out of the dam (y)")
plt.plot(X, np.dot(X_intercept, res_x), '--', lw=2)
plt.show()
