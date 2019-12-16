import os,sys
import random

import numpy as np

np.random.seed(2019)


# generate traditional training set: pid, feature_list, label
fh = open('large_features.txt','r')
lines = fh.readlines()
fh.close()

arr = lines[0].strip().split(',')
pid_features = arr[1:-1] # features (a list of pids)

col_idx = list(range(len(pid_features)))
random.shuffle(col_idx)
small_features = col_idx[:int(0.5*len(pid_features))]
# print header
header = []
for i in range(len(small_features)):
	header.append(pid_features[small_features[i]])
print('uid,',','.join(header),',label')

data_lines = lines[1:]
random.shuffle(data_lines)
data = data_lines[:int(0.2*len(data_lines))]
for d in data:
	arr = d.strip().split(',')
	uid = arr[0].strip()
	label = arr[-1].strip()
	arr = arr[1:-1]
	features = []
	for i in range(len(small_features)):
		idx = small_features[i]
		val = arr[idx]
		features.append(val)
	print(uid,',',','.join(features),',',label)
