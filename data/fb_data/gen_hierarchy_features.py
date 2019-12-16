import os,sys
import random

import numpy as np

np.random.seed(2019)

fname = sys.argv[1] # large_features.txt <== generated from raw data activities.txt


fh = open('fid_brandname_category.txt','r')
lines = fh.readlines()
fh.close()
id2category = {}

# different levels of features: global dictionary
featureset = [[],[],[],[]]

for line in lines:
	arr = line.strip().split(',')
	pid = arr[0].strip()
	category = arr[-1].strip()
	id2category[pid] = category

	cat_levels = category.split('/')
	if len(cat_levels) == 1:
		if cat_levels[0].strip() not in featureset[0]:
			featureset[0].append(cat_levels[0].strip())
	elif len(cat_levels) == 2:
		if cat_levels[0].strip() not in featureset[0]:
			featureset[0].append(cat_levels[0].strip())
		if cat_levels[1].strip() not in featureset[1]:
			featureset[1].append(cat_levels[1].strip())
	else:
		if cat_levels[0].strip() not in featureset[0]:
			featureset[0].append(cat_levels[0].strip())
		if cat_levels[1].strip() not in featureset[1]:
			featureset[1].append(cat_levels[1].strip())
		if cat_levels[2].strip() not in featureset[2]:
			featureset[2].append(cat_levels[2].strip())

	featureset[3].append(pid)


# generate features in a specific format: (feature,structure;feature,structure;...)
fh = open(fname,'r')
lines = fh.readlines()
fh.close()
arr = lines[0].strip().split(',')
pid_features = arr[1:-1] # features (a list of pids)

id2structures = {}
for i in range(len(pid_features)):
	pid = pid_features[i].strip()
	categories = id2category[pid]
	cat_levels = categories.split('/')
	#print(pid,':',cat_levels,',',len(cat_levels))

	structures = []
	if len(cat_levels) == 1:
		idx_0 = str(featureset[0].index(cat_levels[0]))
		idx_1 = str(len(featureset[1]))
		idx_2 = str(len(featureset[2]))
	elif len(cat_levels) == 2:
		idx_0 = str(featureset[0].index(cat_levels[0]))
		idx_1 = str(featureset[1].index(cat_levels[1]))
		idx_2 = str(len(featureset[2]))
	elif len(cat_levels) == 3:
		idx_0 = str(featureset[0].index(cat_levels[0]))
		idx_1 = str(featureset[1].index(cat_levels[1]))
		idx_2 = str(featureset[2].index(cat_levels[2]))
	
	structures.append(idx_0)
	structures.append(idx_1)
	structures.append(idx_2)
	structures.append(str(i)) # add index of pid

	id2structures[pid] = structures


for i in range(1,len(lines)):
	arr = lines[i].strip().split(',')
	uid = arr[0].strip()
	label = arr[-1].strip()

	# foreach feature: pid
	feature_structure = {}
	for j in range(len(pid_features)):
		feature_val = arr[1:-1][j]

		pid = pid_features[j].strip()
		structures = id2structures[pid]

		# foreach level
		for m in range(len(structures)):
			key = '/'.join(structures[:m+1])
			if key not in feature_structure:
				feature_structure[key] = feature_val
			else:
				if feature_val == '1':
					feature_structure[key] = '1'
	f_output = []
	for k,v in feature_structure.items():
		f_output.append(v+','+k)
	print(';'.join(f_output)+';'+label)
