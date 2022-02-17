# libs
import os
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
    # plt.show()


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

J = computeCost(X_theta0, y, theta=np.array([0.0, 0.0]))
print('With theta = [0, 0] \nCost computed = %.2f' % J)
print('Expected cost value (approximately) 32.07\n')

# further testing of the cost function
J = computeCost(X_theta0, y, theta=np.array([-1, 2]))
print('With theta = [-1, 2]\nCost computed = %.2f' % J)
print('Expected cost value (approximately) 54.24')

"""
Gradient Descent - update theta for each iteration

> theta_j = theta_j - alpha * (1/m) * SUM(h_theta(x)-y) * xj 
> a list of new J values - iteration 
"""

def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    theta = theta.copy() # need copy for iterative process otherwise it jumps back to original
    J_history = [] # create an empty array
    
    for i in range(num_iters):
        h = np.dot(X, theta)
        theta = theta - (alpha / m) * np.dot((h - y), X) 
        J_history.append(computeCost(X, y, theta))

    return theta, J_history

theta, J_history = gradientDescent(X_theta0, y, theta=np.zeros(2), alpha=0.01, num_iters=1500)
print("Theta found by gradient descent: {:.4f}, {:.4f}".format(*theta))
print('Expected theta values (approximately): [-3.6303, 1.1664]')

"""
Use plotData function to add additional visualization to it 
"""

# plotData(X, y)
# plt.plot(X, np.dot(X_theta0, theta), '-')
# plt.legend(["Training Data", "Linear Regression"])
# plt.show()

"""
Some validation calulcations using
> profit in areas of 35,000 and 70,000 people
"""
X_validation = np.array([3.5, 7])
X_validation_theta0 = np.stack([np.ones(X_validation.size), X_validation], axis=1)
profit = np.dot(X_validation_theta0, theta)
print("Profit in area with a population of 35,000 and 70,000 people are respectively %.2f and %.2f" % (profit[0]*10000, profit[1]*10000))

"""
Visualizing J(theta)
"""

# grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100) # from -10 to 10 in 100 steps
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals containing 0s 
J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

# fill out J_vals
for i, theta0 in enumerate(theta0_vals): # using enumerate keeps counter in "i" without using count +1 in code
    for j, theta1 in enumerate(theta1_vals):
        # populate the given matrix J_vals using indices
        J_vals[i, j] = computeCost(X_theta0, y, [theta0, theta1])

J_vals = J_vals.T # transpose function

# surface plot
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='viridis')
plt.xlabel('theta0')
plt.ylabel('theta1')
plt.title('Surface')

# contour plot
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
ax = plt.subplot(122)
plt.contour(theta0_vals, theta1_vals, J_vals, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))
plt.xlabel('theta0')
plt.ylabel('theta1')
plt.plot(theta[0], theta[1], 'ro', ms=10, lw=2)
plt.title('Contour, showing minimum')

plt.show()