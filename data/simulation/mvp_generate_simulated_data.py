import os,sys

import numpy as np

n = sys.argv[1] # 100
p = sys.argv[2] # 5500/5

seed = 42

# re-generating Y using regression
np.random.seed(seed)
mu, sigma = 0, 0.5
v = np.random.normal(mu, sigma, size=int(n))

# run generate one tree 5 times and combine to obtain the final output
for i in range(5):
	temp_fname = 'temp_'+str(i)
	os.system('python -W ignore mvp_generate_simulated_onetree.py ' + n + ' ' + p + ' ' + str(i) + ' y_fname > ' + temp_fname)

# combine
for i in range(int(n)):
	output = []
	fs = [] # features for each instance
	for j in range(5):
		temp_fname = 'temp_'+str(j)
		with open(temp_fname,'r') as fh:
			lines = fh.readlines()
			output.append(lines[i].strip()[:-2])

			arr = lines[i].strip().split(';')
			for k in range(len(arr)):
				fs.append(int(arr[k].split(',')[0].strip()))

	np.random.seed(seed)
	betas = np.random.uniform(-1,1,len(fs))
	Y = 0
	if np.dot(betas,fs) + v[i] > 0:
		Y = 1 
	print(';'.join(output) + ';' + str(Y))

for i in range(5):
	temp_fname = 'temp_'+str(i)
	os.remove(temp_fname)
