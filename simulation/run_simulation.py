import os,sys

import numpy as np

# a range of values for a specific parameter
# lasso: 0-0.1
# sis: 1 - 50 
# FDR: 0-0.1
# sr-w0 fixed: 0.5, alpha_delta: 0-1
# sr-alpha_delta fixed: 0.5, w0: 0-1
# ddc: 0-1
alg = sys.argv[1]

for i in range(50):
	#fname = 'data_S_100_5500/simulated_data_100_5500_'+str(i)+'.txt'
	fname = 'data_S_1000_170000/simulated_data_1000_170000_'+str(i)+'.txt'

	command = 'python -W ignore '+alg+'.py ' + fname
	os.system(command)
