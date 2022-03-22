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
X_val, y_val = data['Xval'], data['yval'][:, 0]


# Test data
X_test, y_test = data["Xtest"], data["ytest"][:, 0]

m = y.size
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

    reg = (lambda_ / (2 * m)) * np.sum(np.square(temp))
    J = (1 / (2 * m)) * np.sum(np.square(h - y)) + reg 

    grad = (1 / m) * np.dot((h - y), X) + (lambda_ / m) * temp
    
    return J, grad

# Concatenate X
X_intercept = np.concatenate([np.ones((m, 1)), X], axis=1)
theta = np.ones(X_intercept.shape[1])
cost, grad = costFunctionReg(theta, X_intercept, y, lambda_ = 1)

res_x = utils.trainLinearReg(costFunctionReg, X_intercept, y, lambda_=0)

# fig = plt.figure()
# plt.plot(X, y, 'ro', ms=10, mec='k', mew=1)
# plt.xlabel("Change in water level (x)")
# plt.ylabel("Water flowing out of the dam (y)")
# plt.plot(X, np.dot(X_intercept, res_x), '--', lw=2)

def learningCurve(X, y, X_val, y_val, lambda_=0):
    m = y.size
    error_train = np.zeros(m) # amount of m (training samples)
    error_val = np.zeros(m)

    # loop through the numbers of examples starting from 1 
    # Obtain J_train and J_val for each m 
    for i in range(1, m + 1):
        # obtain the the ideal theta parameters from training data for each m 
        
        theta_t = utils.trainLinearReg(costFunctionReg, X[:i], y[:i], lambda_ = lambda_)

        # first set is the training set using trainLinearReg
        cost_train, grad_train = costFunctionReg(theta_t, X[:i], y[:i], lambda_=lambda_)

        # second set is the CV set 
        cost_CV, grad_CV = costFunctionReg(theta_t, X_val, y_val, lambda_=lambda_)

        error_train[i-1] = cost_train
        error_val[i-1] = cost_CV


    return error_train, error_val

"""
Plotting and validating results
"""
X_val_intercept = np.concatenate([np.ones((X_val.shape[0], 1)), X_val], axis=1)
error_train, error_val = learningCurve(X_intercept, y, X_val_intercept, y_val, lambda_=0)

plt.plot(np.arange(1, m+1), error_train, np.arange(1, m+1), error_val, lw=2)
plt.title('Learning curve for linear regression')
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 150])
plt.show() 

print('# Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print('  \t%d\t\t%f\t%f' % (i+1, error_train[i], error_val[i]))

