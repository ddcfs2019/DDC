import os,sys

import numpy as np

# generate 500 and 50 datasets for larger and smaller cases, respectively
for i in range(500): # for i in range(50):
	fname = 'simulated_data_100_5500_'+str(i)+'.txt'
	os.system('python -W ignore generate_simulated_data.py 100 5500 > ' + fname)


# a range of values for a specific parameter
# lasso: 0-0.1
# sis: 1 - 50 
# FDR: 0-0.1
# sr-w0 fixed: 0.5, alpha_delta: 0-1
# sr-alpha_delta fixed: 0.5, w0: 0-1
# ddc: 0-1
v = np.linspace(0, 1, 50)
for k in range(1,len(v)-1):
	tprs = []
	fprs = []
	times = []

	for i in range(10):
		fname = 'simulated_data_100_5500_'+str(i)+'.txt'

		command = 'python -W ignore ddc.py ' + fname + ' ' + str(float(v[k])) + ' > line.txt'
		os.system(command) # float or int


		fh = open('line.txt','r')
		data = fh.read().strip().split(',')
		fh.close()
		if os.path.getsize('line.txt') == 0:
			continue


		tprs.append(float(data[0].strip()))
		fprs.append(float(data[1].strip()))
		times.append(float(data[2].strip()))


	mean_tpr = sum(tprs)/float(len(tprs))
	mean_fpr = sum(fprs)/float(len(fprs))
	mean_time = sum(times)/float(len(times))

	print(mean_tpr,',',mean_fpr,',',mean_time)
