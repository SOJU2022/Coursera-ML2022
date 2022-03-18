"""
Week 5 - Neural Networks using Backpropagation
"""

# libs
from mimetypes import init
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
    3) Calculate gradients using backpropagation
    """

    if X.ndim == 1: 
        X = X[None] # when X is (m,) > turn into (m, 1)

    m, n = X.shape

    # Reshape nn_params back into the parameters Theta1 and Theta2
    # Because layer have different amount of nodes - it needs be rolled into a vector 
    # reshape turns it back into individual matrices
    Theta1 = np.reshape(theta[:hidden_layer_size * (input_layer_size+1)], (hidden_layer_size, input_layer_size + 1)) # (25, 401)
    Theta2 = np.reshape(theta[hidden_layer_size * (input_layer_size + 1):], (num_labels, (hidden_layer_size + 1))) # (10, 26)
    
    # output shape
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape) 

    # obtain J by obtaining the hypothesis h first
    X = np.concatenate([np.ones((m, 1)), X], axis=1) # 5000 x 401
    # hidden layer
    z2 = np.dot(X, Theta1.T) # 5000 x 25
    a2 = utils.sigmoidFunc(z2) # 5000 x 25
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
    delta_3 = a3 - y_matrix # delta error
    delta_2 = np.dot(delta_3, Theta2)[:, 1:] * utils.sigmoidGradient(z2)

    gradDelta_3 = np.dot(delta_3.T, a2)
    gradDelta_2 = np.dot(delta_2.T, X)

    # add regularization term to DVec
    # DVec = D_ij for each layer - Theta_grad in this case
    Theta1_grad = (1 / m) * gradDelta_2 + (lambda_ / m) * temp1 # with temp you remove j=0 from regularizing
    Theta2_grad = (1 / m) * gradDelta_3 + (lambda_ / m) * temp2 

    DVec = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])

    return J, DVec


"""
Validating the Cost Function and Gradient of J 
"""
# J, DVec = costFunctionFF(nn_params, X, y, input_layer_size, hidden_layer_size,
#                    num_labels, lambda_=0)
# print('Cost at parameters (loaded from ex4weights): %.6f ' % J)
# print('The cost should be about                   : 0.287629.')

# # Weight regularization parameter (we set this to 1 here).
# J_reg, DVec_reg = costFunctionFF(nn_params, X, y, input_layer_size, hidden_layer_size,
#                       num_labels, lambda_=1)

# print('Cost at parameters (loaded from ex4weights): %.6f' % J_reg)
# print('This value should be about                 : 0.383770.')

"""
In this section some good practices will be conducted regarding Backpropagation
> Gradient Checking
> Random Initialization of Theta - cannot start with 0 as previously done for symmetry breaking
"""

def randInitializeWeights(L_in, L_out): # L_in = s_l (nodes in layer l) and L_out = s_l+1
    # Strat from course docs
    epsilon_init = np.sqrt(6) / np.sqrt(L_in + L_out)

    W = np.zeros((L_out, 1 + L_in)) # 1 adds the bias term?

    # for i in range(W.shape[0]):
    #     for j in range(W.shape[1]):
    #         W[i, j] = np.random.rand() * (2 * epsilon_init) - epsilon_init

    # or take the easy way without frooty loops
    W = np.random.rand(L_out, 1 + L_in) * (2 * epsilon_init) - epsilon_init

    return W 

print('Initializing Neural Network Parameters ...')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()])

"""
Gradient Checking 
> Turn off when you are not using it - heavy computational resources needed
"""
# #  Check gradients by running checkNNGradients from utils
# lambda_ = 3
# utils.checkNNGradients(costFunctionFF, lambda_)

# # Also output the costFunction debugging values
# debug_J, debug_DVec  = costFunctionFF(nn_params, X, y, input_layer_size,
#                           hidden_layer_size, num_labels, lambda_)

# print('\n\nCost at (fixed) debugging parameters (w/ lambda = %f): %f ' % (lambda_, debug_J))
# print('(for lambda = 3, this value should be about 0.576051)')

"""
Using scipy.optimize.minimize func 
> To obtain theta for min J using initial_nn_theta
"""

options = {'maxiter': 100} # iteration number - the higher the # iter, the higher the accuracy
lambda_ = 1

# shorthand statements: statement lambda is helpful to write single line
# functions without naming a function
costFunction = lambda p: costFunctionFF(p, X, y, input_layer_size,
                                        hidden_layer_size, num_labels, lambda_)


# costFunction only takes one argument p
res = optimize.minimize(costFunction,
                        initial_nn_params,
                        jac=True,
                        method='TNC',
                        options=options)

nn_params = res.x

# Obtain Theta1 and Theta2 back from nn_params
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                    (hidden_layer_size, (input_layer_size + 1)))

Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                    (num_labels, (hidden_layer_size + 1)))

pred = utils.predict(Theta1, Theta2, X)
print('Training Set Accuracy: %f' % (np.mean(pred == y) * 100))

"""
Next: 
> Visualizing the hidden layer 
> Make the code more compact
> Understand the core mechanics of NN 
> We use the sigmoid function - which is a binary analysis hypothesis 
    * Check for examples and more practice using NN
    * Create a nice dashboard out of it? 
"""
