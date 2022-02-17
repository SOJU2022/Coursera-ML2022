# libs
import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# import data 
data = np.loadtxt(os.path.join("data", "ex1data2.txt"), delimiter=',')
X, y = data[:,:2], data[:,2]
m = y.size

# Print and validate few data points
print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:, 1]', 'y'))
print('-'*26)
for i in range(m): 
    print('{:8.0f}{:8.0f}{:10.0f}'.format(X[i, 0], X[i, 1], y[i]))

"""
Dispersion in scale / magnitude between the features 
> Apply Feature Normalization
"""
def featureNormalization(X):
    X_norm = X.copy() # make a copy of original X, otherwise you need to populate it with zeros or ones and replace them

    # for loop dependent on X size and not y size 
    X_size = X.shape[0]

    mu = np.array([np.mean(X[:,0]), np.mean(X[:,1])])
    sigma = np.array([np.std(X[:,0]), np.std(X[:,1])])

    for i in range(X_size):
        X_norm[i,0] = (X[i,0] - mu[0]) / sigma[0]
        X_norm[i,1] = (X[i,1] - mu[1]) / sigma[1]

    return X_norm, mu, sigma

# get the parameters from featureNormalization out of the function
X_norm, mu, sigma = featureNormalization(X)

# add intercept term to X (for theta)
# previously using np.stack
# X = np.stack([np.ones(m), X], axis=1) 
X = np.concatenate([np.ones((m, 1)), X_norm], axis=1) # (m, 1) is the shape, ie mx1 matrix. axis is the dimension either in the horizontal or vertical axis


"""
Gradient Descent Multivariate
"""

# calculates J for a specific theta
def computeCostMulti(X, y, theta):
    m = y.shape[0]
    h = np.dot(X, theta)
    z = h - y
    J = (1/(2*m)) * np.dot(z.T, z)
    
    return J

J = computeCostMulti(X, y, theta=np.array([0, 0, 0]))


# iteration of theta - not speficically getting the best outcome since it depends on # of iteration
def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    theta = theta.copy() # need copy for iterative process otherwise it jumps back to original - usually done when doing a recursive loop function afterwards 
    J_history = [] # create an empty array
    
    # copy needed for theta to populate the information
    for i in range(num_iters):
        h = np.dot(X, theta)
        z = h - y
        theta = theta - (alpha / m) * np.dot(z, X) 
        J_history.append(computeCostMulti(X, y, theta))

    return theta, J_history

theta, J_history = gradientDescentMulti(X, y, theta=np.zeros(3), alpha=0.01, num_iters=1500)
print("Theta found by gradient descent: {:.4f}, {:.4f}, {:.4f}".format(*theta))


"""
Some validation calulcations using
> house price with feature 1 in square feet and feature 2 # of bedrooms
"""
# # array with 1 square bracket - matrix if you use multiple square brackets
# X_val = np.array([[6000, 7], [12000, 15], [2000, 1], [3000, 2]])  # use of double brackets to make a matrix-like array 
# # X_val = X_val[None, :] # promote to a matrix of 1,X_val.size
# n = X_val.shape[0] # to decide to populate the constant theta0 term

# # apply featureNormalization 
# X_val_norm, mu_val, sigma_val = featureNormalization(X_val)

# X_val_norm = np.concatenate([np.ones((n, 1)), X_val_norm], axis=1)
# house_price = np.dot(X_val_norm, theta)
# print(house_price)        

"""
The above is a wrong approach because it recalculates mu and sigma with a different dataset
> Make use of existing data - mu, sigma etc.
"""

def normCalculation(X_val):
    n = X_val.shape[0]
    
    # need to populate X_val to be used
    # make a copy instead of populating it with zeros or ones
    X_val = X_val.copy()
    
    for i in range(n):
        X_val[i, 0] = (X_val[i, 0] - mu[0]) / sigma[0]
        X_val[i, 1] = (X_val[i, 1] - mu[1]) / sigma [1]

    return X_val

X_val_test = np.array([[4000, 2]])
X_val = normCalculation(X_val_test) # turn an array into a matrix-like + calculations
X_val = np.concatenate([np.ones((X_val.shape[0], 1)), X_val], axis=1)
house_price = np.dot(X_val, theta)
print(house_price)


"""
Visualization
"""

def plotData(X, y, theta):
    # data configuration
    X_feat1 = X[:,1]
    X_feat2 = X[:,2]

    # surface plot
    fig = plt.figure() # configure it
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_feat1, X_feat2, y)

    # regression line addition
    theta1_vals = np.linspace(-3, 3, 100)
    theta2_vals = np.linspace(-3, 3, 100)
    X_thetas = np.array([theta1_vals, theta2_vals])
    X_thetas = X_thetas.T
    X_thetas = np.concatenate([np.ones((X_thetas.shape[0], 1)), X_thetas], axis=1)
    y_value = np.dot(X_thetas, theta)
    ax.plot(theta1_vals, theta2_vals, y_value, 'r-')

    plt.xlabel("square feet") # feature 1
    plt.ylabel("# bedrooms") # feature 2
    plt.title("House Price")
    plt.show()

# call visualization
# plotData(X, y, theta)

"""
Selection of the best learning rate - alpha parameter
> Plot the number of iterations against Cost J for variety of alpha 
> Cost J should go down at increasing iterations - the quickness depends on the steepness
> Plot the curves of different alpha in the same figure
"""

# plot the alphas and their respective J data
def plotAlpha(J_history):
    fig = plt.figure()

    # loop through different J_history and plot them
    for i in range(len(J_history)):
        plt.plot(np.arange(len(J_history[i])), J_history[i,:], lw=2)
        
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost J")
    plt.legend("")    
    plt.show()

# separate J for each alpha
def multipleAlpha(alpha):
    alpha = alpha.copy()
    num_iters = 500
    J_history = []

    for i in range(alpha.shape[0]): # can use len as well for 1D arrays
        theta = np.zeros(3) # initial state for each iteration
        theta, J_history = gradientDescentMulti(X, y, theta, alpha[i], num_iters)
        J_history.append(J_history)
    
    return J_history


# setup the alpha 
alpha = np.array([0.01, 0.03, 0.1, 0.3]) # create an array of multiple alphas in log

# return a matrix of J_histories from different alphas - matrix of size: num_iters x alpha.size
J_history = multipleAlpha(alpha)

# call J-alpha plot
plotAlpha(J_history) 








