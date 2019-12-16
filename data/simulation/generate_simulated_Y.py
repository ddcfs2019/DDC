import os,sys

import numpy as np

Y = np.random.binomial(1,0.5,int(n)).tolist()

line = []
fh = open('y_fname','w')
for i in range(len(Y)):
    line.append(str(Y[i]))
fh.write(','.join(line))
fh.close()
