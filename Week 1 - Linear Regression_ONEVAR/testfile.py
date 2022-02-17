# libs
import os
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read comma separated data
data = np.loadtxt(os.path.join("data", "ex1data1.txt" ), delimiter=',')
X, y = data[:,0], data[:,1]

m = y.size # number of training examples 

"""
Plotting the Data 
"""

def plotData(X, y):
    fig = plt.figure() # first configure plt object 
    plt.plot(X, y, 'ro', ms=10, mec='k')  
    plt.ylabel("Profit in $10,000")
    plt.xlabel("Population of City in 10,000s")
    plt.show()

# plotData(X, y)

"""
Gradient Descent

1)
The objective of Linear Regression is to minimize the Cost Function
J(theta) = (1/2m) * SUM(h_theta(x) - y)^2
-> from the cost function you can make a graph as well 

2)
h_theta is the hypothesis equation given the linear model 
- this model does not have to be linear
h_theta(x) = theta_transpose * x = theta_0 (const) + theta_1 * x1 etc. 

3) How to get these theta values?
> Make use of the batch gradient descent algorithm - iterative method
theta_j = theta_j - alpha * (1/m) * SUM(h_theta(x)-y) * xj 
"""

# add a column for theta_0 constant in X
X_theta0 = np.stack([np.ones(m), X], axis=1)

# Convergence of J function - minimalisation

def computeCost(X, y, theta):
    m = y.size
    h = np.dot(X, theta)
    J = (1/(2*m)) * np.sum(np.square(h - y))

    return J

"""
Gradient Descent - update theta for each iteration

> theta_j = theta_j - alpha * (1/m) * SUM(h_theta(x)-y) * xj 
> a list of new J values - iteration 
"""

def gradientDescent(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn `theta`. Updates theta by taking `num_iters`
    gradient steps with learning rate `alpha`.
    
    Parameters
    ----------
    X : array_like
        The input dataset of shape (m x n+1).
    
    y : arra_like
        Value at given features. A vector of shape (m, ).
    
    theta : array_like
        Initial values for the linear regression parameters. 
        A vector of shape (n+1, ).
    
    alpha : float
        The learning rate.
    
    num_iters : int
        The number of iterations for gradient descent. 
    
    Returns
    -------
    theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).
    
    J_history : list
        A python list for the values of the cost function after each iteration.
    
    Instructions
    ------------
    Peform a single gradient step on the parameter vector theta.

    While debugging, it can be useful to print out the values of 
    the cost function (computeCost) and gradient here.
    """
    # Initialize some useful values
    m = y.shape[0]  # number of training examples
    
    # make a copy of theta, to avoid changing the original array, since numpy arrays
    # are passed by reference to functions
    theta = theta.copy()
    
    J_history = [] # Use a python list to save cost in every iteration
    
    for i in range(num_iters):
        # ==================== YOUR CODE HERE =================================
        theta = theta - (alpha / m) * (np.dot(X, theta) - y).dot(X)

        # =====================================================================
        
        # save the cost J in every iteration
        J_history.append(computeCost(X, y, theta))
    
    return theta, J_history

# initialize fitting parameters
theta = np.zeros(2)

# some gradient descent settings
iterations = 1500
alpha = 0.01

theta, J_history = gradientDescent(X_theta0 ,y, theta, alpha, iterations)
print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))
print('Expected theta values (approximately): [-3.6303, 1.1664]')

print(theta[0])