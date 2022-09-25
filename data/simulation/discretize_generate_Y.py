import os,sys
import math

import numpy as np

from scipy.stats import multivariate_normal


n = int(sys.argv[1]) # number of instances

# top level X
cov_matrix = np.zeros((5,5))
for i in range(5):
    for j in range(5):
        if i == j:
            cov_matrix[i,j] = 1
        else:
            cov_matrix[i,j] = math.pow(0.7,abs(i-j))
rv = multivariate_normal(mean=None, cov=cov_matrix)
Z = rv.rvs(size=(int(n),))

# add noise
mu, sigma = 0, 1
epsilon = np.random.normal(mu, sigma, size=(n,5))

Z += epsilon
np.save('top_z_vals',Z)