"""
Coursera Week 4 - Neural Networks
> Making use of Logistic Regressors 
However Logistic Regression cannot form more complex hypotheses as it is only
a linear classifier - can add more features, such as polynomial features as done in Week 3
But this make it very expensive to train since this will lead to huge matrices
"""

# libs
import numpy as np
from matplotlib import pyplot as plt
import os 
from scipy import optimize
from scipy.io import loadmat # load .mat (matlab) files into python environment

"""
Dataset 
> X - Input a 20x20 pixel picture of a random number
between 0 and 9 (5000 pictures)
> y - Output (known number)
"""

# 20x20 input images of digits
input_layer_size = 400 # 20x20 = 400 individual pixels

# 10 label, from 1 to 10 
num_labels = 10

# training data stored in arrays X and y
data = loadmat(os.path.join("Data", "ex3data1.mat"))
X, y = data["X"], data["y"].ravel() # ravel change any matrix to a flat 1-D array

y[y == 10] = 0 # search the index with y == 10 and change this output to 0
m = y.size

"""
Visualize the data
"""

def displayData(X, example_width=None, figsize=(10, 10)):
    """
    Displays 2D data stored in X in a 2D grid layout 
    """

    # compute rows and cols
    if X.ndim == 2:
        m, n = X.shape
    elif X.ndim == 1:
        n = X.size
        m = 1 # only 1 row available
        X = X[None] # Promote to a 2D array 
    else:
        raise IndexError("Input X should be 1 or 2 dimensional.")

    example_width = example_width or int(np.round(np.sqrt(n)))
    example_height = n / example_width

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    fig, ax_array = plt.subplots(display_rows, display_cols, figsize=figsize)
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    ax_array = [ax_array] if m == 1 else ax_array.ravel()

    for i, ax in enumerate(ax_array):
        ax.imshow(X[i].reshape(example_width, example_width, order='F'),
                  cmap='Greys', extent=[0, 1, 0, 1])
        ax.axis('off')

# Randomly select 100 data points to display
rand_indices = np.random.choice(m, 100, replace=False)
sel = X[rand_indices, :]

displayData(sel)

"""
Vectorization of Data input X and Theta - to be used for each layer
"""

# Sigmoid Function
def sigmoidFunc(z):
    h = 1 / (1 + np.exp(-z))

    return h 

# Unregularized Cost Function using multiple Logistic Regression a(i)
def lrCostFunction(theta, X, y, lambda_):
    m = y.size

    # Vectorizing the cost function - Ultimately not necessary, but for validation sakes

    # convert labels to ints if their type is bool
    if  y.dtype == bool:
        y = y.astype(int)

    J = 0 # amount if J equal to amount of examples
    grad = np.zeros(theta.shape) # amount of features

    z = np.dot(X, theta) # for each label you have different theta weighting
    h = sigmoidFunc(z) # each theta input has a unique h output

    # regularization part
    temp = theta 
    temp[0] = 0 

    # Cost Func
    J = (1 / m) * np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h)) + (lambda_ / (2 * m)) * np.sum(np.square(temp))

    # gradient
    grad = (1 / m) * np.dot((h - y), X) # for j = 0 
    grad = grad + (lambda_ / m) * temp # for j >= 1

    return J, grad


# Test parameters + validation
theta_t = np.array([-2, -1, 1, 2], dtype=float)
X_t = np.concatenate([np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F')/10.0], axis=1)
y_t = np.array([1, 0, 1, 0, 1])
lambda_t = 3

cost, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)

print('Cost         : {:.6f}'.format(cost))
print('Expected cost: 2.534819')
print('-----------------------')
print('Gradients:')
print(' [{:.6f}, {:.6f}, {:.6f}, {:.6f}]'.format(*grad))
print('Expected gradients:')
print(' [0.146561, -0.548558, 0.724722, 1.398003]')

"""
One-vs-all Classification
> Training multiple logistic regression classifiers
> One for each of the K classes in our dataset (10 labels)
> Outputs the probability that the input matches with the labels using an array with 0 and 1s
"""

def oneVsAll(X, y, num_labels, lambda_):
    m, n = X.shape # number of features without bias

    all_theta = np.zeros((num_labels, n + 1)) # num_labels x features + 1 
    J = np.zeros((m, 1))

    X_intercept = np.concatenate([np.ones((m, 1)), X], axis=1)

    # you want to obtain J and grad for each label
    # Basically loop through the matrix i, j and fill them with the corresponding theta values
    for c in range(num_labels): 
        # Optimize func to obtain J and Theta
        initial_theta = np.zeros((n+1, 1)) # this include the bias term
        options = {'maxiter': 50}
        res = optimize.minimize(lrCostFunction,
                                initial_theta,
                                (X_intercept, (y == c), lambda_), # in this case y == c because we want to search for each label y
                                jac=True,
                                method='CG',
                                options=options)

        cost = res.fun 
        grad = res.x # shape (n+1, 1) as well

        # outcome of the optimize func
        all_theta[c] = grad 
        J[c] = cost

    return all_theta, J

lambda_ = 0.1
all_theta, cost = oneVsAll(X, y, num_labels, lambda_)


# prediction of the input 
def predictOneVsAll(all_theta, X_examples):
    m, n = X_examples.shape # where X is the input - Examples of 20x20 pixels 
    num_labels = all_theta.shape[0]

    p = np.zeros(m)
    X_examples = np.concatenate([np.ones((m, 1)), X_examples], axis=1)

    """
    Step for step brainstorm for the code
    1. Input a number of examples inside X - columnwise - 400 + intercept bias
    2. Calculate the predictions using all_theta for each example
    2. This calculation yields an array with output from each label for each example
    3. Need to obtain the number that is predicted from the array with index value 1 (argmax?)
    4. Put all the predicted number in "p" parameter
    """

    # step 1 - Done in validation
    # step 2 - yields a matrix for all examples with their predictions
    output = np.dot(X_examples, all_theta.T) # a matrix num_examples x num_labels

    # step 3
    output_index = np.argmax(output, axis=1) # obtain index of the max of each row

    # step 4 
    p = output_index

    return p 

# get p with chosen data from X 
X_validation = X[3500:3550,:] # obtain 20 examples from X
p = predictOneVsAll(all_theta, X)
print(p)

# predict the training set accuracy against the calculated Theta
print('Training Set Accuracy: {:.2f}%'.format(np.mean(p == y) * 100))


