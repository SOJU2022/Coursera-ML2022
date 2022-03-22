# libs 
import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt

def trainLinearReg(costFunctionReg, X, y, lambda_, maxiter=200):
    # Initialize Theta
    initial_theta = np.zeros(X.shape[1])

    # Create "short hand" for the cost function to be minimized
    costFunction = lambda t: costFunctionReg(t, X, y, lambda_)

    # Now, costFunction is a function that takes in only one argument
    options = {'maxiter': maxiter}

    # Minimize using scipy
    res = optimize.minimize(costFunction, initial_theta, jac=True, method='TNC', options=options)
    
    return res.x