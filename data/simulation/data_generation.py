import os,sys

import numpy as np


# generate 500 datasets
os.system('python -W ignore discretize_generate_Y.py 100')
for i in range(500):
	fname = 'data_S_100_5500_fdr/simulated_data_100_5500_'+str(i)+'.txt'
	os.system('python -W ignore discretize_generate_simulated_data.py 100 5500 > ' + fname)

os.system('python -W ignore discretize_generate_Y.py 1000')
for i in range(50):
	fname = 'data_S_1000_170000_fdr/simulated_data_1000_170000_'+str(i)+'.txt'
	os.system('python -W ignore discretize_generate_simulated_data.py 1000 170000 > ' + fname)
