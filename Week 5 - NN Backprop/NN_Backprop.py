"""
Week 5 - Neural Networks using Backpropagation
"""

# libs
import matplotlib
import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt
from scipy.io import loadmat
import os
import utils

# data import and set initial parameters
data = loadmat(os.path.join("Data", "ex4data1.mat")) # 5000 pictures - 20x20 pixels
X, y = data["X"], data["y"].ravel() 

y[y == 10] = 0 # change every index value to 0 where the ouput is equal to 10 
m, n = X.shape

# Visualization of random X
rand_indices = np.random.choice(m, size=100, replace=False)
sel_series = X[rand_indices, :]

utils.displayData(sel_series)

"""
Step 1 - Feedforward propagation to obtain hypothesis h_theta
"""

# Neural Network layering - 3 layers (1 hidden)
# Initial Parameters - Need to randomly initialize the thetas 
input_layer_size = 400
hidden_layer_size = 25 # 25 nodes in the hidden layer
num_labels = 10 

# Initial thetas (with bias term)
weights = loadmat(os.path.join("Data", "ex4weights.mat")) # usually randomized by code
Theta1, Theta2 = weights["Theta1"], weights["Theta2"]
Theta2 = np.roll(Theta2, 1, axis=0)

# Unroll the parameters into vecotrs
nn_params = np.concatenate([Theta1.ravel(), Theta2.ravel()]) # put them in the 1 axis 

def costFunctionFF(theta, X, y, input_layer_size, hidden_layer_size, num_labels, lambda_): # regularized form

    """
    Process to obtain J and gradients for each layer and node
    1) FeedForward Propagation by normal standard in order to obtain hypothesis h
        > This hypothesis h is known for each layer and node given by a's
    2) Use each of these hypothesis to obtain J(theta) given input theta
    3) Calculate gradients ? 
    """

    if X.ndim == 1: 
        X = X[None] # when X is (m,) > turn into (m, 1)

    m, n = X.shape

    # Reshape nn_params back into the parameters Theta1 and Theta2
    Theta1 = np.reshape(theta[:hidden_layer_size * (input_layer_size+1)], (hidden_layer_size, input_layer_size + 1)) # (25, 401)
    Theta2 = np.reshape(theta[hidden_layer_size * (input_layer_size + 1):], (num_labels, (hidden_layer_size + 1))) # (10, 26)
    
    # output shape
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape) 

    # obtain J by obtaining the hypothesis h first
    X = np.concatenate([np.ones((m, 1)), X], axis=1) # 5000 x 401
    # hidden layer
    z2 = np.dot(X, Theta1.T) # 5000 x 25+1
    a2 = utils.sigmoidFunc(z2) # 5000 x 26
    a2 = np.concatenate([np.ones((a2.shape[0], 1)), a2], axis=1)

    # output layer
    z3 = np.dot(a2, Theta2.T) # Theta2 = 10x26
    a3 = utils.sigmoidFunc(z3) # 5000 x 10 - no bias term

    # output y
    y_matrix = y.reshape(-1)
    y_matrix = np.eye(num_labels)[y_matrix]

    # add regularization term
    temp1 = Theta1
    temp1[0] = 0 
    temp2 = Theta2
    temp2[0] = 0 

    reg = (lambda_ / (2 * m)) * (np.sum(np.square(temp1)) + np.sum(np.square(temp2))) 

    # Cost Function
    J = (-1 / m) * np.sum((np.log(a3) * y_matrix) + np.log(1 - a3) * (1 - y_matrix)) + reg

    # backpropagation
    delta_3 = a3 - y_matrix
    delta_2 = np.dot(delta_3, Theta2)[:, 1:] * utils.sigmoidGradient(z2)

    gradDelta_3 = np.dot(delta_3.T, a2)
    gradDelta_2 = np.dot(delta_2.T, X)

    # add regularization term to DVec
    Theta1_grad = (1 / m) * gradDelta_2
    Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + (lambda_ / m) * Theta1[:, 1:]
    
    Theta2_grad = (1 / m) * gradDelta_3
    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + (lambda_ / m) * Theta2[:, 1:]

    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])

    return J, grad

lambda_ = 0
J, grad = costFunctionFF(nn_params, X, y, input_layer_size, hidden_layer_size,
                   num_labels, lambda_)
print('Cost at parameters (loaded from ex4weights): %.6f ' % J)
print('The cost should be about                   : 0.287629.')






