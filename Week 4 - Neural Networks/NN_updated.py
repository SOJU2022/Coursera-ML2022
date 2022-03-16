"""
Coursera Week 4 - Neural Networks
> Real Neural Networks without Logistic Regressors in the hidden layer
"""

# libs
from re import I
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import os 
from scipy import optimize
from scipy.io import loadmat # load .mat (matlab) files into python environment

import utils # import external file

# import data
data = loadmat(os.path.join("Data", "ex3data1.mat"))
X, y = data["X"], data["y"].ravel() # ravel change any matrix to a flat 1-D array

y[y == 10] = 0 # search the index with y == 10 and change this output to 0
m = y.size

# randomly permute examples - to be used for visualizing one picture at a time
indices = np.random.permutation(m)

# Randomly select 100 data points to display
rand_indices = np.random.choice(m, 100, replace=False)
sel = X[rand_indices, :]

utils.displayData(sel) # plot the 100 examples from sel

"""
In this part - the Theta arrays for each layer is already given
> In backpropagation next week one may derive / calculate the theta parameters
"""

# Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 0 to 9

# Load the .mat file, which returns a dictionary 
weights = loadmat(os.path.join('Data', 'ex3weights.mat'))

# get the model weights from the dictionary
# Theta1 has size 25 x 401
# Theta2 has size 10 x 26
Theta1, Theta2 = weights['Theta1'], weights['Theta2']

# swap first and last columns of Theta2, due to legacy from MATLAB indexing, 
# since the weight file ex3weights.mat was saved based on MATLAB indexing
Theta2 = np.roll(Theta2, 1, axis=0)

"""
Implement Feedforward Propagation and Prediction 
"""

# Similar to the OneVsAll Classification strategy
def predict(Theta1, Theta2, X):
    if X.ndim == 1: 
        X = X[None] # promote to 2D by adding a (,1)

    # parameters used for NN
    m, n = X.shape
    num_labels = Theta2.shape[0]
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    p = np.zeros(X.shape[0])

    # hidden layer 1
    z_output_layer1 = np.dot(X, Theta1.T) # num_examples x num_labels
    a2 = utils.sigmoidFunc(z_output_layer1) # first hidden layer - sigmoid func in this case
    a2 = np.concatenate([np.ones((a2.shape[0], 1)), a2], axis=1)


    # hidden layer 2
    output_layer2 = np.dot(a2, Theta2.T) # theta 2 - 10x26 matrix
    a3 = utils.sigmoidFunc(output_layer2)

    output_final = np.argmax(a3, axis=1)

    p = output_final

    return p 

# prediction = predict(Theta1, Theta2, X)

# get p with chosen data from X 
range_start = 3500
range_end = 4700
X_validation = X[range_start:range_end,:] # obtain 20 examples from X
prediction = predict(Theta1, Theta2, X_validation)
print(prediction)


# training accuracy
print('Training Set Accuracy: {:.1f}%'.format(np.mean(prediction == y[range_start:range_end]) * 100))

# Display number pictures with the given prediction for each 
if indices.size > 0:
    i, indices = indices[0], indices[1:]
    utils.displayData(X[i, :], figsize=(4, 4))
    pred = predict(Theta1, Theta2, X[i, :])
    print('Neural Network Prediction: {}'.format(*pred))
else:
    print('No more images to display!')