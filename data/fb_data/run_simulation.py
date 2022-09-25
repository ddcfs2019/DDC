import os,sys

import numpy as np

'''
# generate 10 datasets
os.system('python -W ignore discretize_generate_Y.py 100')
for i in range(10):
	fname = 'data_S_100_5500_fdr/simulated_data_100_5500_'+str(i)+'.txt'
	os.system('python -W ignore discretize_generate_simulated_data.py 100 5500 > ' + fname)

os.system('python -W ignore discretize_generate_Y.py 1000')
for i in range(10):
	fname = 'data_S_1000_170000_fdr/simulated_data_1000_170000_'+str(i)+'.txt'
	os.system('python -W ignore discretize_generate_simulated_data.py 1000 170000 > ' + fname)

'''
# a range of values for a specific parameter
# lasso: 0-0.1
# sis: 1 - 50 
# FDR: 0-0.1
# sr-w0 fixed: 0.5, alpha_delta: 0-1
# sr-alpha_delta fixed: 0.5, w0: 0-1
# ddc: 0-1
alg = sys.argv[1]

for i in range(5):
	#fname = 'data_S_100_5500/simulated_data_100_5500_'+str(i)+'.txt'
	fname = 'data_S_1000_170000/simulated_data_1000_170000_'+str(i)+'.txt'

	command = 'python -W ignore '+alg+'.py ' + fname
	os.system(command)
