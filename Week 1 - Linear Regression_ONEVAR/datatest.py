# libs
import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# import data 
data = np.loadtxt(os.path.join("data", "ex1data2.txt"), delimiter=',')
X, y = data[:,:2], data[:,2]
m = y.size

mu = np.mean(X[:,0])

print(mu)