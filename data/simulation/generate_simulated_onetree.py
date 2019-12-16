import os,sys
import math

import random
import numpy as np
from scipy import stats

# data format
# Y, x_value/level_1,...,x_value/level_1/level_2,...

n = int(sys.argv[1])
p = int(sys.argv[2])/5
ntimes = int(sys.argv[3])
Y_fname = sys.argv[4]


features = [] # store all features, level by level, each level has 2^(l-1) features, each feature is a n-dim vector
structures = [] # store all structures, level by level, each level has 2^(l-1) structures, each structure is a string
ki = [] # store ki, level by level, each level has 2^(l-1) ki's, each ki is a number


#generate Y
fh = open(Y_fname,'r')
line = fh.read()
fh.close()
arr = line.strip().split(',')
Y = []
for k in range(len(arr)):
	Y.append(int(arr[k].strip()))
#Y = np.random.binomial(1,0.5,n).tolist()

# features,structures,ki's for the top level
L = 0

features_0 = []
structures_0 = [str(ntimes)]
ki_0 = [np.random.uniform(-0.25,0.25)]
for idx in range(n):
	prob = 2*ki_0[0] + 1.0
	if Y[idx] == 0:
		prob = 0.5 - 2*ki_0[0]
		
	custm = stats.rv_discrete(values=(np.arange(2),(1-prob,prob)))
	features_0.append(custm.rvs(size=1)[0])

features.append([features_0])
structures.append(structures_0)
ki.append(ki_0)


def generate_fvalues(parent_layer,parent_idx,idx,relevant):
	parent_structure = structures[parent_layer][parent_idx]
	parent_ki = ki[parent_layer][parent_idx]
	cur_ki = np.random.uniform(-abs(parent_ki),abs(parent_ki))
	cur_structure = parent_structure + '/' + str(idx)

	fvals = []
	if relevant == 0:
		fvals = np.random.binomial(1,0.3,n).tolist()
	else:
	# if feature not in relevant feature set
		for m in range(n):
			if features[parent_layer][parent_idx][m] == 0: # if X_p(i) = 0
				fvals.append(0)
				continue

			prob = 2*cur_ki + 1.0
			if Y[m] == 0:
				prob = 0.5 - 2*cur_ki

			custm = stats.rv_discrete(values=(np.arange(2),(1-prob,prob)))
			fvals.append(custm.rvs(size=1)[0])

	return (fvals,cur_structure,cur_ki)


# from the second level
count = 1
flag = 0
while flag == 0: # l: layer
	L += 1 # current layer

	n_features = int(math.pow(2,L*(L+1)/2))
	#print 'layer:',L,',#features:',n_features
	features_l = []
	structures_l = []
	ki_l = []
	for idx in range(n_features):
		relevant = 0
		if idx == 0:
			relevant = 1
		parent_idx = int(idx/math.pow(2,L))
		parent_layer = L-1

		#print 'pl:',parent_layer,', pi:',parent_idx,',cur_idx:',idx

		fvals,cur_structure,cur_ki = generate_fvalues(parent_layer,parent_idx,idx,relevant)
		features_l.append(fvals)
		structures_l.append(cur_structure)
		ki_l.append(cur_ki)

		count += 1
		if count >= p:
			flag = 1
			break

	features.append(features_l)
	structures.append(structures_l)
	ki.append(ki_l)


# outputs
layers = len(features)
for i in range(n):
	output_line = []
	for l in range(layers):
		nf = len(features[l])
		for f in range(nf):
			output_line.append(str(features[l][f][i]) + ',' + structures[l][f])
	print(';'.join(output_line))# + ';' + str(Y[i])
