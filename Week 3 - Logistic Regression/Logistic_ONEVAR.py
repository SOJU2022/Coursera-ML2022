# libs
import numpy as np
import os
from matplotlib import pyplot as plt 
from scipy import optimize

# dataset 1
data = np.loadtxt(os.path.join("Datafile", "ex2data1.txt"), delimiter=",")
X, y = data[:,:2], data[:,2]
m = y.size
n = X.shape[0]

# dataset 2
data_2 = np.loadtxt(os.path.join("Datafile", "ex2data2.txt"), delimiter=",")
X_2, y_2 = data_2[:,:2], data_2[:,2]
m_2 = y.size
n_2 = X.shape[0]

"""
Visualize the data 
"""

def plotData(X, y):
    """
    Plots the data points X and y into a new figure with binary [0, 1] as output
    - Point with * for 1 and o for 0

    Parameters
    ----------
    X : array_like - An Mx2 matrix representing the dataset
    y : array_like - Label values for the dataset. A vector of size (M, )
    """

    # boolean statement
    pos = y == 1
    neg = y == 0

    fig = plt.figure()
    
    # plot multiple for y either 0 or 1
    """
    Interesting mechanics:
    > pos / neg outputs a boolean True or False
    > However, X[pos/neg], will pick up the value that is equal to True from the array
    """
    plt.plot(X[pos, 0], X[pos, 1], "k*", lw=2, ms=10) # lw = linewidth, ms = markersize
    plt.plot(X[neg, 0], X[neg, 1], "ko", mfc='y', ms=8, mec='k', mew=1)

    plt.xlabel("Exam 1 Score")
    plt.ylabel("Exam 2 Score")
    plt.legend(["Admitted", "Not Admitted"])

"""
Plot results including Decision boundary
""" 
def plotDecisionBoundary(plotData, X, y, theta):
    """
    Plots the data points X and y into a new figure with the decision boundary defined by theta.
    Plots the data points with * for the positive examples and o for  the negative examples.
    Parameters
    ----------
    plotData : func
        A function reference for plotting the X, y data.
    theta : array_like
        Parameters for logistic regression. A vector of shape (n+1, ).
    X : array_like
        The input dataset. X is assumed to be  a either:
            1) Mx3 matrix, where the first column is an all ones column for the intercept.
            2) MxN, N>3 matrix, where the first column is all ones.
    y : array_like
        Vector of data labels of shape (m, ).
    """
    # make sure theta is a numpy array
    theta = np.array(theta)

    # Plot Data (remember first column in X is the intercept)
    plotData(X[:, 1:3], y)

    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])

        # Calculate the decision boundary line
        plot_y = (-1. / theta[2]) * (theta[1] * plot_x + theta[0])

        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y)

        # Legend, specific for the exercise
        plt.legend(['Admitted', 'Not admitted', 'Decision Boundary'])
        plt.xlim([30, 100])
        plt.ylim([30, 100])
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((u.size, v.size))
        # Evaluate z = theta*x over the grid
        for i, ui in enumerate(u):
            for j, vj in enumerate(v):
                z[i, j] = np.dot(mapFeature(ui, vj), theta)

        z = z.T  # important to transpose z before calling contour
        # print(z)

        # Plot z = 0
        plt.contour(u, v, z, levels=[0], linewidths=2, colors='g')
        plt.contourf(u, v, z, levels=[np.min(z), 0, np.max(z)], cmap='Greens', alpha=0.4)


def computeCost(theta, X, y):
    theta = theta.copy()
    z = np.dot(X, theta) # X = (m x n+1) - n amount of features, theta = (n+1, 1)
    h = 1 / (1 + np.exp(-z)) # sigmoid function - h = (m, 1)
    calc_cf = -y * np.log(h) - (1-y) * np.log((1-h))
    calc_grad = np.dot((h-y), X)
    
    # return the following parameter
    J = 0 
    grad = np.zeros(theta.shape) # theta.shape = (n+1, 1)

    # Cost Function
    J = (1 / m) * np.sum(calc_cf)

    # gradient of the cost
    grad = (1 / m) * calc_grad

    return J, grad

# Add intercept term to X for theta_0
X_intercept = np.concatenate([np.ones((n, 1)), X], axis=1)
initial_theta = np.zeros((X_intercept.shape[1], 1))

J, grad = computeCost(initial_theta, X_intercept, y)

"""
Validating cost func + grad output
"""

# print("Cost at initial theta: %.3f" % J) # one way to print using parameters
# print("Expected cost (approx): 0.693 \n")

# print("Cost at initial theta: ", J) # one way to print using parameters
# print("Expected cost (approx): 0.693 \n")


# print("Gradient at initial theta: ")
# print("\t[{:.4f}, {:.4f}, {:.4f}]".format(*grad))
# print('Expected gradients (approx):\n\t[-0.1000, -12.0092, -11.2628]\n')


"""
Advanced Optimization - alternative for Gradient Descent
> Using function scipy.optimize.minimize 
> Minimize the J function for a certain theta 
    Parameters:
    -----------
    - costFunction (providing reference only)
    - initial_theta
    - (X, y)
    - jac: indication if the cost function returns the Jacobian 
    - method: optimization method / algo used
    - options: additional parameters
"""

# set options for optimize.minimize
options= {'maxiter': 400}

# this function returns an object - "OptimizeResult"
res = optimize.minimize(computeCost,
                        initial_theta,
                        (X_intercept, y),
                        jac=True,
                        method='TNC',
                        options=options)

# the fun property of `OptimizeResult` object returns
# the value of costFunction at optimized theta
cost = res.fun

# the optimized theta is in the x property
theta = res.x

# Print theta to screen
print('Cost at theta found by optimize.minimize: {:.3f}'.format(cost))
print('Expected cost (approx): 0.203\n')

print('theta:')
print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*theta))
print('Expected theta (approx):\n\t[-25.161, 0.206, 0.201]')


"""
Evaluating Logistic Regression 
    - The predict function will produce "1" or "0"  given a dataset and learned parameter vector theta
"""

def sigmoid(theta, X): # adding a sigmoid function 
    theta = theta.copy()
    z = np.dot(X, theta) # X = (m x n+1) - n amount of features, theta = (n+1, 1)
    h = 1 / (1 + np.exp(-z)) # sigmoid function - h = (m, 1)
    return h 

def predict(theta, X): # input X_intercept
    m = X.shape[0] # number of training examples
    p = np.zeros(m)

    h = sigmoid(theta, X)

    for i, value_sig in enumerate(h): 
        if value_sig >= 0.5:
            p[i] = 1
        else:
            p[i] = 0 

    return p 

"""
Validating Logistic Regression output
"""
# predict probability for a student with score 45 on exam 1
# and score 85 on exam 2
student_1 = np.array([1, 45, 85])
prob = sigmoid(theta, student_1)
print("For a student with scores 45 and 85, "
        "we predict an admission probability of {:.3f}".format(prob))
print("Expected value: 0.775 +/- 0.002 \n")

# Compute accuracy on our training set
p = predict(theta, X_intercept)
print('Train Accuracy: {:.2f} %'.format(np.mean(p == y) * 100))
print('Expected accuracy (approx): 89.00 %')

"""
Regularized Logistic Regression
> Generalize to new examples and new data
"""

# # Quick Visualisation
# plotData(X_2, y_2)
# # Labels and Legend
# plt.xlabel('Microchip Test 1')
# plt.ylabel('Microchip Test 2')

# # Specified in plot order
# plt.legend(['y = 1', 'y = 0'], loc='upper right')
# plt.show()

"""
Application of Feature Mapping
    - Linear decision boundary will not work for this particular data
    - Feature Mapping gives the optionality to extend your feature vector
    - func mapFeature
"""

def mapFeature(X1, X2, degree=6):
    """
    Maps the two input features to quadratic features used in the regularization exercise.
    Returns a new feature array with more features, comprising of
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    Parameters
    ----------
    X1 : array_like
        A vector of shape (m, 1), containing one feature for all examples.
    X2 : array_like
        A vector of shape (m, 1), containing a second feature for all examples.
        Inputs X1, X2 must be the same size.
    degree: int, optional
        The polynomial degree.
    Returns
    -------
    : array_like
        A matrix of of m rows, and columns depend on the degree of polynomial.
    """
    if X1.ndim > 0:
        out = [np.ones(X1.shape[0])]
    else:
        out = [np.ones(1)]

    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((X1 ** (i - j)) * (X2 ** j))

    if X1.ndim > 0:
        return np.stack(out, axis=1)
    else:
        return np.array(out)

# Cost Func + Gradient based on Regularization
X_2_intercept = mapFeature(X_2[:,0], X_2[:,1]) # define new X using mapFeature

def costFunctionReg(theta, X, y, lambda_):
    n_2 = y.size
    J = 0 
    grad = np.zeros(theta.shape)

    z = np.dot(X, theta) # X = (m x n+1) - n amount of features, theta = (n+1, 1)
    h = 1 / (1 + np.exp(-z)) # sigmoid function - h = (m, 1)
    calc_cf = -y * np.log(h) - (1-y) * np.log((1-h))

    # theta
    temp = theta
    temp[0] = 0 

    # calculation of gradients
    grad = (1 / n_2) * np.dot((h-y), X) # for j = 0
    grad = grad + (lambda_ / n_2) * temp

    # Cost Function
    J = (1 / m) * np.sum(calc_cf) + (lambda_ / (2*n_2)) * np.sum(np.square(temp))

    return J, grad

"""
Validate costFunctionReg
"""

# Initialize fitting parameters
initial_theta_2 = np.zeros(X_2_intercept.shape[1])

# set initial parameters
lambda_ = 1

J, grad = costFunctionReg(initial_theta_2, X_2_intercept, y_2, lambda_)

print('Cost at initial theta (zeros): {:.3f}'.format(J))
print('Expected cost (approx)       : 0.693\n')

print('Gradient at initial theta (zeros) - first five values only:')
print('\t[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*grad[:5]))
print('Expected gradients (approx) - first five values only:')
print('\t[0.0085, 0.0188, 0.0001, 0.0503, 0.0115]\n')

# Compute and display cost and gradient
# with all-ones theta and lambda = 10
test_theta = np.ones(X_2_intercept.shape[1])
J, grad = costFunctionReg(test_theta, X_2_intercept, y_2, 10)

print('------------------------------\n')
print('Cost at test theta    : {:.2f}'.format(J))
print('Expected cost (approx): 3.16\n')

print('Gradient at initial theta (zeros) - first five values only:')
print('\t[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*grad[:5]))
print('Expected gradients (approx) - first five values only:')
print('\t[0.3460, 0.1614, 0.1948, 0.2269, 0.0922]')

"""
Plotting the Decision Boundary for Regularization problem
"""

# set options for optimize.minimize
options= {'maxiter': 400}

# this function returns an object - "OptimizeResult"
res = optimize.minimize(costFunctionReg,
                        initial_theta_2,
                        (X_2_intercept, y_2, lambda_),
                        jac=True,
                        method='TNC',
                        options=options)

# the fun property of `OptimizeResult` object returns
# the value of costFunction at optimized theta
J_2 = res.fun

# the optimized theta is in the x property
theta_2 = res.x

plotDecisionBoundary(plotData, X_2_intercept, y_2, theta_2)
plt.show()

# Compute accuracy on our training set
p = predict(theta_2, X_2_intercept)

