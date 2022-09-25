import os,sys
import math

import random
import numpy as np
from scipy import stats

# data format
# Y, x_value/level_1,...,x_value/level_1/level_2,...

n = int(sys.argv[1])
p = int(sys.argv[2])/5
ntimes = int(sys.argv[3]) # which branch (5 subtrees in total: 5 top level features)


features = [] # store all features, level by level, each level has 2^(l-1) features, each feature is a n-dim vector
structures = [] # store all structures, level by level, each level has 2^(l-1) structures, each structure is a string


## features and structures for the top level
L = 0

features_0 = []
structures_0 = [str(ntimes)]

top_Z = np.load('top_z_vals.npy')[:,ntimes]
features_0.append(np.ones(n,dtype=int).tolist())
features.append(features_0)
structures.append(structures_0)


rec_length = 10.0 # [-5, +5]

# from the second level
count = 1 # the number of features
flag = 0
while flag == 0: # l: layer
	L += 1 # current layer

	n_features = int(math.pow(2,L))
	#print('layer:',L,',#features:',n_features)
	features_l = []
	structures_l = []

	# create buckets
	buckets = []
	for i in range(n_features):
		r = (-5+i*rec_length/n_features,-5+(i+1)*rec_length/n_features)
		buckets.append(r)

	for i in range(n_features):
		fvals = [0]*n
		b = buckets[i]
		for j in range(n):
			if top_Z[j] >= b[0] and top_Z[j] <= b[1]:
				fvals[j] = 1
		features_l.append(fvals)
		
		# add structures
		parent_idx = int(i/2)
		parent_layer = L-1
		parent_structure = structures[parent_layer][parent_idx]
		cur_structure = parent_structure + '/' + str(i)
		structures_l.append(cur_structure)

		count += 1
		if count >= p:
			flag = 1
			break

	features.append(features_l)
	structures.append(structures_l)
	#print(features,'\n',structures)
	#sys.exit(1)


# formatting outputs
# each instance: feature_value (x),structure_value (layer) ; feature_value (x),structure_value (layer) ; ...
#np.random.seed(seed)
#beta = np.random.uniform(-1,1,count)

layers = len(features)
for i in range(n):
	output_line = []
	fvals = []
	for l in range(layers):
		nf = len(features[l])
		for f in range(nf):
			output_line.append(str(features[l][f][i]) + ',' + structures[l][f])
			fvals.append(features[l][f][i])
	#if np.dot(beta,fvals) > 0:
	#	Y = 1
	#else:
	#	Y = 0
	print(';'.join(output_line))# + ';' + str(Y))
