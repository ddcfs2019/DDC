import os,sys
import math

import numpy as np

from scipy.stats import multivariate_normal

'''
    the first line is the Y
    and a numpy object for top X values
'''

n = sys.argv[1] # number of instances

seed = 42

np.random.seed(seed)
Y = np.random.binomial(1,0.5,int(n)).tolist()

line = []
fh = open('y_fname','w')
for i in range(len(Y)):
    line.append(str(Y[i]))
fh.write(','.join(line))
fh.close()

# top level X
cov_matrix = np.zeros((5,5))
for i in range(5):
    for j in range(5):
        if i == j:
            cov_matrix[i,j] = 1
        else:
            cov_matrix[i,j] = math.pow(0.7,abs(i-j))
rv = multivariate_normal(mean=None, cov=cov_matrix)
Z = rv.rvs(size=(int(n),), random_state=42)

np.random.seed(seed)
mu, sigma = 0, 2
epsilon = np.random.normal(mu, sigma, size=(int(n),5))

X = np.zeros((int(n),5),dtype=int)
for i in range(int(n)):
    for j in range(5):
        if Z[i,j] + epsilon[i,j] > 0:
            X[i,j] = 1
np.save('top_x_vals',X)
