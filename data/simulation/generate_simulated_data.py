import os,sys

import numpy as np

n = sys.argv[1] # 100
p = sys.argv[2] # 5500

Y = np.random.binomial(1,0.5,int(n)).tolist()
line = []
fh = open('y_fname','w')
for i in range(len(Y)):
	line.append(str(Y[i]))
fh.write(','.join(line))
fh.close()

# run generate one tree 5 times and combine to obtain the final output
for i in range(5):
	temp_fname = 'temp_'+str(i)
	os.system('python -W ignore generate_simulated_onetree.py ' + n + ' ' + p + ' ' + str(i) + ' y_fname > ' + temp_fname)

# combine
for i in range(int(n)):
	output = []
	for j in range(5):
		temp_fname = 'temp_'+str(j)
		fh = open(temp_fname,'r')
		lines = fh.readlines()
		output.append(lines[i].strip())
		fh.close()
	print(';'.join(output) + ';' + str(Y[i]))
	#print Y[i]


os.remove('y_fname')
for i in range(5):
	temp_fname = 'temp_'+str(i)
	os.remove(temp_fname)
